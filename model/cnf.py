import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device
    )
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return pred_lddt_ca * 100

class CNF(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, *args, **kwargs) -> Tensor:
        return self.model(*args, **kwargs)

    def decode_euler(self, z: Tensor, batch, conf_model=None) -> Tensor:
        coords, atom_type, adj, mask = batch['pos'], batch['atom_type'], batch['adj'], batch['mask']
        atom_type = F.one_hot(atom_type, num_classes=17)

        def ode_func(model, t, x):
            t = t.expand(*coords.shape[:2], 1)
            node_feat = torch.cat([atom_type, t], dim=-1) * mask[..., None]
            if self.config.model.embed_aa:
                aa = F.one_hot(batch.aa, num_classes=280)
                node_feat = torch.cat([node_feat, aa], dim=-1)
            if self.config.model.embed_mask:
                if self.config.data.data == 'plinder':
                    mask_embed = batch['ligand_mask']
                elif self.config.data.data == 'ncaa':
                    mask_embed = batch['ncaa_mask']
                else:
                    mask_embed = mask
                if self.config.model.add_noise:
                    mask_embed = mask_embed + batch['loss_mask'] # just add so we dont have to retrain ligand model: ncaa is 2, repacked sidechains are 1
                node_feat = torch.cat([node_feat, mask_embed.unsqueeze(-1)], dim=-1)

            # concat noised edge distances
            pair_mask = mask[:, None] * mask[:, :, None]
            edge_dist = torch.linalg.norm(x[:, None] - x[:, :, None], dim=-1).unsqueeze(-1)
            adj_onehot = F.one_hot(adj.long(), num_classes=5)
            edge_feat = torch.cat([adj_onehot, edge_dist], dim=-1) * pair_mask[..., None]

            out = model(node_feat, x, edge_feat, mask)

            return out

        ts = torch.linspace(
            1.0, 0, self.config.sample.num_timesteps)
        t_1 = ts[0]

        if 'add_noise' not in self.config.model.keys():
            self.config.model.add_noise = False

        if self.config.model.add_noise:
            loss_mask = batch['loss_mask']
            ncaa_mask = batch['ncaa_mask']
            sc_mask = loss_mask * (1-ncaa_mask)
            z[sc_mask==1] = coords[sc_mask==1] + (0.2*z[sc_mask==1])

        if self.config.model.reset_backbone:
            backbone_mask = batch['backbone_mask']
            z[backbone_mask==1] = coords[backbone_mask==1]

        prot_traj, conf_traj = [z], [torch.zeros_like(mask)]
        for t_2 in ts:
            # Run model.
            trans_t_1 = prot_traj[-1]

            if self.config.model.add_noise:
                ligand_mask = batch['loss_mask']
                trans_t_1[ligand_mask==0] = coords[ligand_mask==0]

            else:
                if self.config.data.data == 'plinder':
                    ligand_mask = batch['ligand_mask']
                    trans_t_1[ligand_mask==0] = coords[ligand_mask==0]
                elif self.config.data.data == 'ncaa':
                    ligand_mask = batch['ncaa_mask']
                    trans_t_1[ligand_mask == 0] = coords[ligand_mask == 0]

            if self.config.model.reset_backbone:
                trans_t_1[backbone_mask == 1] = coords[backbone_mask == 1]

            t = torch.ones((z.shape[0], 1, 1), device=z.device) * t_1

            with torch.no_grad():
                out = ode_func(self.model, t, trans_t_1)

            # Take reverse step
            d_t = t_2 - t_1
            if self.config.train.loss_type == 'x0':
                trans_vf = (out - trans_t_1)/ (1-t)
            elif self.config.train.loss_type == 'vf':
                trans_vf = out
            else: raise NotImplementedError('wrong loss type')
            trans_t_2 = trans_t_1 + trans_vf * d_t
            trans_t_2[mask==0] = coords[mask==0]

            if conf_model is not None:
                with torch.no_grad():
                    conf_out = ode_func(conf_model, t, trans_t_1)
                    conf_out = compute_plddt(conf_out)
                    conf_out[ligand_mask==0] = 100.
            else:
                conf_out = torch.zeros_like(mask)

            conf_traj.append(conf_out)

            if not self.config.model.add_noise:
                if self.config.data.data == 'plinder':
                    ligand_mask = batch['ligand_mask']
                    trans_t_2[ligand_mask==0] = coords[ligand_mask==0]
                elif self.config.data.data == 'ncaa':
                    ligand_mask = batch['ncaa_mask']
                    trans_t_2[ligand_mask == 0] = coords[ligand_mask == 0]
            else:
                ligand_mask = batch['loss_mask']
                trans_t_2[ligand_mask == 0] = coords[ligand_mask == 0]

            if self.config.model.reset_backbone:
                trans_t_2[backbone_mask == 1] = coords[backbone_mask == 1]

            prot_traj.append(trans_t_2)
            t_1 = t_2

        return torch.stack(prot_traj,dim=0), torch.stack(conf_traj,dim=0)
