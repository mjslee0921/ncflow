from boltz.model.modules.transformers import AtomTransformer
from boltz.model.modules.trunk import PairformerModule
import torch
from torch import nn

class AF3Transformer(torch.nn.Module):
    def __init__(self, s_in, z_in, s_dim, z_dim, n_pf_layers=6, n_atom_layers=4, heads=8, out_dim=3, **kwargs):
        super().__init__()
        self.s_in = nn.Linear(s_in, s_dim)
        self.z_in = nn.Linear(z_in, z_dim)
        self.c_in = nn.Linear(s_dim, s_dim)
        self.x_in = nn.Linear(3, s_dim)
        self.x_dist_in = nn.Linear(1, z_dim)
        self.pairformer = PairformerModule(s_dim, z_dim, n_pf_layers)
        self.atom_transformer = AtomTransformer(depth=n_atom_layers, heads=heads, dim=s_dim, dim_pairwise=z_dim)
        self.x_out = nn.Linear(s_dim, out_dim)

    def forward(self, s, x, z, mask):
        pair_mask = mask[...,None] * mask[:,None]
        x_pair = torch.cdist(x,x,p=2).unsqueeze(-1)
        s = self.s_in(s) + self.x_in(x)
        z = self.z_in(z) + self.x_dist_in(x_pair)
        s, z = self.pairformer(s, z, mask, pair_mask)
        c = s + self.c_in(s)
        s = self.atom_transformer(q=s, c=c, p=z, mask=mask)
        x_update = self.x_out(s)
        return x_update