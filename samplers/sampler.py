import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.loader import load_seed, load_device, load_ema, load_checkpoint
from utils.logger import Logger, set_log, start_log
from utils.train_utils import count_parameters, recursive_to
from pathlib import Path
from dataloaders.dataset import get_dataloader
import math
from model.af3 import AF3Transformer
from model.cnf import CNF
from utils.structure_utils import create_structure_from_crds, create_chemical_structure
import biotite.structure as bstruct
import biotite.structure.io as bsio

class Sampler(object):
    def __init__(self, config, ddp=False):
        super(Sampler, self).__init__()
        self.config = config
        self.seed = load_seed(self.config.seed)
        self.device = 'cuda'
        self.train_loader, self.test_loader, self.train_sampler, self.test_sampler = get_dataloader(self.config,ddp=ddp,sample=True)

    def train(self, ts, name='test', save_traj=False):
        self.config.exp_name = ts
        self.ckpt = f'{ts}'
        print(f'{self.ckpt}')

        ckpt_dict = torch.load(self.config.ckpt, weights_only=False)
        self.training_cfg = ckpt_dict['config']

        # -------- Load models, optimizers, ema --------
        self.model = AF3Transformer(**self.training_cfg.model).cuda()
        print(f'Number of parameters: {count_parameters(self.model)}')
        self.ema = load_ema(self.model, decay=self.training_cfg.train.ema)
        self.model, self.ema = load_checkpoint(self.model, self.ema, ckpt_dict)
        self.model = CNF(self.model, self.config).eval()
        self.ema.copy_to(self.model.parameters())

        if self.config.conf_ckpt is not None:
            conf_ckpt = torch.load(self.config.conf_ckpt, weights_only=False)
            conf_state_dict, conf_config = conf_ckpt['state_dict'], conf_ckpt['config']
            if 'module' in list(conf_state_dict.keys())[0]:  # if trained with DDP
                conf_state_dict = {k[7:]: v for k, v in conf_state_dict.items()}
            self.conf_model = AF3Transformer(**conf_config.model).cuda().eval()
            self.conf_model.load_state_dict(conf_state_dict)
        else:
            self.conf_model = None

        save_path = Path(f'./samples/{self.training_cfg.data.data}/{self.training_cfg.train.name}/{name}')
        save_path.mkdir(exist_ok=True, parents=True)

        sample_path = save_path.joinpath(ts)
        sample_path.mkdir(exist_ok=True, parents=True)

        output_dict = {}
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                batch = recursive_to(batch,self.device)
                coords, mask = batch['pos'], batch['mask']
                sample_id = batch['id'][0]
                atom_type_str = [i[0] for i in batch['atom_type_str']]
                output_dict[sample_id] = {}
                for sample_idx in range(self.config.sample.n_samples):
                    outPath = sample_path.joinpath(sample_id, f'run_{sample_idx+1}')
                    outPath.mkdir(exist_ok=True,parents=True)
                    z = torch.randn_like(coords)

                    sample_traj, conf_traj = self.model.decode_euler(z, batch, conf_model=self.conf_model)
                    sample = sample_traj[-1]

                    # extract ligand
                    ligand_mask, receptor_mask = batch['ncaa_mask'], batch['complex_mask']
                    ligand_rmsd = ((sample[ligand_mask==1]-coords[ligand_mask==1]).square().mean()*3).sqrt()

                    if self.config.conf_ckpt is not None:
                        b_factor = conf_traj[-1]
                        b_factor_traj = conf_traj[:,0]
                    else:
                        # use bfactor to color ncaa
                        b_factor = ligand_mask*100
                        b_factor_traj = b_factor.expand(len(sample_traj),-1)

                    create_chemical_structure(coords.cpu().numpy(), atom_type_str,
                                              outPath=str(outPath.joinpath('gt.pdb')),b_factor=b_factor)
                    create_chemical_structure(sample.cpu().numpy(), atom_type_str,
                                              outPath=str(outPath.joinpath('sample.pdb')),b_factor=b_factor)
                    create_chemical_structure(sample_traj.cpu().numpy()[:, 0], atom_type_str,
                                              outPath=str(outPath.joinpath('traj.pdb')),b_factor=b_factor_traj)

                    ligand_atoms = np.array(atom_type_str)[ligand_mask[0].cpu().numpy()==1]

                    create_chemical_structure(coords[ligand_mask==1].cpu().numpy(), ligand_atoms, outPath=str(outPath.joinpath('gt_ncaa.pdb')),b_factor=b_factor)
                    create_chemical_structure(sample[ligand_mask==1].cpu().numpy(), ligand_atoms,
                                              outPath=str(outPath.joinpath('ncaa.pdb')),b_factor=b_factor)

                    ligand_gt = bsio.load_structure(str(outPath.joinpath('gt_ncaa.pdb')))
                    ligand_gt.res_id = np.ones(len(ligand_gt))
                    ligand_gt.bonds = bstruct.connect_via_distances(ligand_gt)
                    bsio.save_structure(str(outPath.joinpath('gt_ncaa.sdf')),ligand_gt)

                    ligand_sample = bsio.load_structure(str(outPath.joinpath('ncaa.pdb')))
                    ligand_sample.res_id = np.ones(len(ligand_sample))
                    ligand_sample.bonds = ligand_gt.bonds
                    bsio.save_structure(str(outPath.joinpath('ncaa.sdf')),ligand_sample)

                    # calculate sasa
                    ligand_mask_np = ligand_mask.bool().cpu().numpy()[0]
                    struct = bsio.load_structure(str(outPath.joinpath('gt.pdb')))
                    sasa = bstruct.sasa(struct, atom_filter=ligand_mask_np, vdw_radii='Single')
                    sasa = sasa[ligand_mask_np].mean()

                    output_dict[sample_id][f'run{sample_idx+1}'] = {
                        'rmsd': ligand_rmsd,
                        'plddt': b_factor[ligand_mask==1].mean().item(),
                        'sasa': sasa
                    }

        torch.save(output_dict, sample_path.joinpath('metrics.pth'))


        print(' ')
        return self.ckpt

import yaml
from easydict import EasyDict as edict

def get_config(config, seed):
    config_dir = f'./config/inference/{config}.yaml'
    config = edict(yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader))
    config.seed = seed

    return config

if __name__ == '__main__':
    import argparse

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('name', type=str, default='test')
    parser.add_argument('--save_traj', type=bool, default=False)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    config = get_config(args.config, seed=args.seed)
    trainer = Sampler(config)
    trainer.train(time.strftime('%b%d-%H:%M:%S', time.gmtime()), name=args.name, save_traj=args.save_traj)
    print(time.time() - start_time)
