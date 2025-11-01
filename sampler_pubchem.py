import time
from tqdm import tqdm
import torch
from utils.loader import load_seed, load_ema, load_checkpoint
from utils.train_utils import count_parameters, recursive_to
from pathlib import Path
from dataset_pubchem import get_dataloader as get_dataloader_pubchem, construct_dataloader
from model.af3 import AF3Transformer
from model.cnf import CNF
from utils.structure_utils import create_chemical_structure, kabsch

class Sampler(object):
    def __init__(self, config, ddp=False):
        super(Sampler, self).__init__()
        self.config = config
        self.seed = load_seed(self.config.seed)
        self.device = 'cuda'
        self.train_loader, self.test_loader, self.train_sampler, self.test_sampler = get_dataloader_pubchem(self.config,ddp=ddp,sample=True)

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

        save_path = Path(f'./samples/{self.training_cfg.data.data}/{self.training_cfg.train.name}/{name}')
        save_path.mkdir(exist_ok=True, parents=True)

        sample_path = save_path.joinpath(ts)
        sample_path.mkdir(exist_ok=True, parents=True)

        output_dict = {}
        with torch.no_grad():
            for split_batch in tqdm(self.test_loader):
                dl, _ = construct_dataloader(split_batch, batch_size=config.sample.batch_size)
                for batch in tqdm(dl, total=len(dl)):
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
                        sample_kabsch, rmsd = kabsch(sample[0], coords[0])

                        create_chemical_structure(coords.cpu().numpy(),atom_type_str, outPath=str(outPath.joinpath('gt.pdb')))
                        create_chemical_structure(sample_kabsch.cpu().numpy(), atom_type_str,
                                                  outPath=str(outPath.joinpath('sample.pdb')))
                        create_chemical_structure(sample_traj.cpu().numpy()[:,0],atom_type_str,
                                                   outPath=str(outPath.joinpath('traj.pdb')))

                        output_dict[sample_id][f'run{sample_idx+1}'] = rmsd

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
