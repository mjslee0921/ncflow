import torch
import torch.nn as nn
from torch import Tensor
import copy
import torch.nn.functional as F
from boltz.model.modules.utils import center_random_augmentation

class CFMLoss(nn.Module):
    def __init__(self, model: nn.Module, config, eps=1e-3):
        super().__init__()
        self.model = model
        self.config = config
        self.eps = eps

    def forward(self, input_batch) -> Tensor:
        batch = copy.deepcopy(input_batch)
        coords, atom_type, adj, mask = batch['pos'], batch['atom_type'], batch['adj'], batch['mask']
        batch_size = coords.shape[0]

        t = torch.rand([batch_size,1,1], device=coords.device)
        z = torch.randn_like(coords)

        if self.config.model.random_augment:
            coords = center_random_augmentation(coords, mask)

        y = (1 - t) * coords + (self.eps + (1 - self.eps) * t) * z
        u = (1 - self.eps) * z - coords

        y = y * mask[...,None]

        if self.config.data.data == 'plinder':
            ligand_mask = batch['ligand_mask']
            y[ligand_mask == 0] = coords[ligand_mask == 0]
            loss_mask = ligand_mask
        elif self.config.data.data == 'ncaa':
            ncaa_mask = batch['ncaa_mask']
            y[ncaa_mask == 0] = coords[ncaa_mask == 0]
            loss_mask = ncaa_mask

            if self.config.model.add_noise:
                loss_mask, ncaa_mask = batch['loss_mask'], batch['ncaa_mask']
                rand_noise = torch.randn_like(coords) * 0.2
                coords_noised = coords + rand_noise
                z[loss_mask == 1] = coords_noised[loss_mask == 1]  # slightly noise sidechains
                z[ncaa_mask == 1] = torch.randn_like(z)[ncaa_mask == 1]  # fully noise ncaa
                y = (1 - t) * coords + (self.eps + (1 - self.eps) * t) * z
                y[loss_mask == 0] = coords[loss_mask == 0]
                y = y * mask[..., None]
                u = (1 - self.eps) * z - coords

        else:
            loss_mask = mask

        if self.config.model.reset_backbone:
            backbone_mask = batch['backbone_mask']
            y = torch.where(backbone_mask[...,None].bool(), coords, y)
            loss_mask = loss_mask * (1-backbone_mask)

        atom_type = F.one_hot(atom_type, num_classes=17)
        t = t.expand(*atom_type.shape[:2],1)
        node_feat = torch.cat([atom_type, t],dim=-1) * mask[...,None]
        if self.config.model.embed_mask:
            if self.config.data.data == 'plinder':
                mask_embed = batch['ligand_mask']
            elif self.config.data.data == 'ncaa':
                mask_embed = batch['ncaa_mask']
                if self.config.model.add_noise:
                    mask_embed = mask_embed + batch['loss_mask'] # just add so we dont have to retrain ligand model: ncaa is 2, repacked sidechains are 1
            else:
                mask_embed = mask
            node_feat = torch.cat([node_feat, mask_embed.unsqueeze(-1)],dim=-1)

        if self.config.model.embed_aa:
            aa = F.one_hot(batch['aa'], num_classes=21)
            node_feat = torch.cat([node_feat, aa], dim=-1)

        # concat noised edge distances
        pair_mask = mask[:,None] * mask[:,:,None]
        edge_dist = torch.linalg.norm(y[:,None] - y[:,:,None],dim=-1).unsqueeze(-1)
        adj_onehot = F.one_hot(adj.long(), num_classes=5)
        edge_feat = torch.cat([adj_onehot, edge_dist],dim=-1) * pair_mask[...,None]

        x = self.model(node_feat, y, edge_feat, mask)
        if self.config.train.loss_type == 'vf':
            trans_loss = ((x - u) * loss_mask.unsqueeze(-1)).square().sum() / loss_mask.sum().clamp(min=1)
            dist_loss = torch.tensor(0.)
        elif self.config.train.loss_type == 'x0':
            trans_loss = ((x - coords) * loss_mask.unsqueeze(-1)).square().sum() / loss_mask.sum().clamp(min=1)
            pred_dist = torch.cdist(x,x)
            true_dist = torch.cdist(coords,coords)
            dist_loss = ((pred_dist - true_dist) * pair_mask).square().sum() / pair_mask.sum().clamp(min=1)
        else: raise NotImplementedError('wrong loss type')

        return {
            'trans_loss': trans_loss,
            'dist_loss': dist_loss,
        }

def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 8.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_positions[..., None, :]
                - all_atom_positions[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_pred_pos[..., None, :]
                - all_atom_pred_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask.unsqueeze(-1)
        * all_atom_mask.unsqueeze(-2)
        * (1.0 - torch.eye(dmat_true.shape[1], device=all_atom_mask.device).unsqueeze(0))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score

def lddt_loss(
    logits: torch.Tensor,
    pred_pos: torch.Tensor,
    gt_pos: torch.Tensor,
    res_mask: torch.Tensor,
    conditional_mask: torch.Tensor,
    cutoff: float = 15.0,
    no_bins: int = 50,
    eps: float = 1e-10,
    **kwargs,
) -> torch.Tensor:

    score = lddt(
        pred_pos,
        gt_pos,
        res_mask,
        cutoff=cutoff,
        eps=eps
    )

    score = score.detach()

    bin_index = torch.floor(score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))

    lddt_ca_one_hot = torch.nn.functional.one_hot(
        bin_index, num_classes=no_bins
    )

    errors = softmax_cross_entropy(logits, lddt_ca_one_hot)
    conditional_mask = conditional_mask.squeeze(-1)
    loss = torch.sum(errors * conditional_mask, dim=-1) / (
        eps + torch.sum(conditional_mask, dim=-1)
    )

    # Average over the batch dimension
    # loss = torch.mean(loss)

    return loss

def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss