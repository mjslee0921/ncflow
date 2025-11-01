from dataset_plinder import get_dataloader
import time
from tqdm import tqdm
import torch
from utils.loader import load_seed, load_ema, load_checkpoint
from utils.train_utils import count_parameters, recursive_to
from pathlib import Path
from model.af3 import AF3Transformer
from model.cnf import CNF
from utils.structure_utils import create_chemical_structure
import numpy as np
import biotite.structure.io as bsio
import biotite.structure as bstruct
class Sampler(object):
    def __init__(self, config, ddp=False, device=None):
        super(Sampler, self).__init__()

        self.config = config
        self.ddp = ddp
        self.seed = load_seed(self.config.seed)
        # self.device = 'cpu'
        self.device = device if device is not None else 'cuda'
        self.train_loader, self.test_loader, self.train_sampler, self.test_sampler = get_dataloader(self.config,ddp=ddp, sample=True)

    def train(self, ts, name='test', save_traj=False):
        self.config.exp_name = ts
        self.ckpt = f'{ts}'

        ckpt_dict = torch.load(self.config.ckpt, weights_only=False)
        self.training_cfg = ckpt_dict['config']

        # -------- Load models, optimizers, ema --------
        self.model = AF3Transformer(**self.training_cfg.model).cuda()
        print(f'Number of parameters: {count_parameters(self.model)}')
        self.ema = load_ema(self.model, decay=self.training_cfg.train.ema)
        self.model, self.ema = load_checkpoint(self.model, self.ema, ckpt_dict)
        self.model = CNF(self.model, self.config).eval()
        self.ema.copy_to(self.model.parameters())

        save_path = Path(f'./samples/{self.training_cfg.data.data}/{self.training_cfg.train.name}/{name}')
        save_path.mkdir(exist_ok=True, parents=True)

        sample_path = save_path.joinpath('samples')
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
                    sample_traj, _ = self.model.decode_euler(z, batch)
                    sample = sample_traj[-1]

                    # extract ligand
                    ligand_mask, receptor_mask = batch['ligand_mask'], batch['receptor_mask']
                    ligand_rmsd = (sample[ligand_mask==1]-coords[ligand_mask==1]).square().mean().sqrt()

                    # use bfactor to color ncaa
                    b_factor = ligand_mask*100
                    b_factor_traj = b_factor.expand(len(sample_traj),-1)

                    create_chemical_structure(coords.cpu().numpy(),atom_type_str, outPath=str(outPath.joinpath('gt.pdb')),b_factor=b_factor)
                    create_chemical_structure(sample.cpu().numpy(), atom_type_str,
                                              outPath=str(outPath.joinpath('sample.pdb')),b_factor=b_factor)
                    create_chemical_structure(sample_traj.cpu().numpy()[:,0],atom_type_str,
                                               outPath=str(outPath.joinpath('traj.pdb')),b_factor=b_factor_traj)

                    ligand_atoms = np.array(atom_type_str)[ligand_mask[0].cpu().numpy()==1]

                    create_chemical_structure(coords[ligand_mask==1].cpu().numpy(), ligand_atoms, outPath=str(outPath.joinpath('gt_ligand.pdb')),b_factor=b_factor)
                    create_chemical_structure(sample[ligand_mask==1].cpu().numpy(), ligand_atoms,
                                              outPath=str(outPath.joinpath('ligand.pdb')),b_factor=b_factor)

                    ligand_gt = bsio.load_structure(str(outPath.joinpath('gt_ligand.pdb')))
                    ligand_gt.res_id = np.ones(len(ligand_gt))
                    ligand_gt.bonds = bstruct.connect_via_distances(ligand_gt)
                    bsio.save_structure(str(outPath.joinpath('gt_ligand.sdf')),ligand_gt)

                    ligand_sample = bsio.load_structure(str(outPath.joinpath('ligand.pdb')))
                    ligand_sample.res_id = np.ones(len(ligand_sample))
                    ligand_sample.bonds = ligand_gt.bonds
                    bsio.save_structure(str(outPath.joinpath('ligand.sdf')),ligand_sample)

                    output_dict[sample_id][f'run{sample_idx+1}'] = ligand_rmsd

        torch.save(output_dict, sample_path.joinpath('metrics.pth'))

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
