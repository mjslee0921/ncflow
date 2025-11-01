import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from utils.train_utils import recursive_to, count_parameters
from utils.loader import load_seed, load_ema
from utils.logger import Logger, start_log, set_log
from dataloaders.dataset import get_dataloader
from loss import CFMLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from model.af3 import AF3Transformer

class Trainer3D(object):
    def __init__(self, config, ddp=False, device=None):
        super(Trainer3D, self).__init__()

        self.config = config
        self.ddp = ddp
        self.seed = load_seed(self.config.seed)
        # self.device = 'cpu'
        self.device = device if device is not None else 'cuda'
        self.train_loader, self.test_loader, self.train_sampler, self.test_sampler = get_dataloader(self.config,ddp=ddp)

    def train(self, ts, resume=False):
        self.config.exp_name = ts
        self.ckpt = f'{ts}'

        # -------- Load models, optimizers, ema --------
        if self.config.model.name == 'af3':
            self.model = AF3Transformer(**self.config.model).to(self.device)
        else:
            raise NotImplementedError('wrong model type')

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.device], find_unused_parameters=False)

        print(f'Number of parameters: {count_parameters(self.model)}')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.train.lr,
                                    weight_decay=self.config.train.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.train.lr_decay)
        self.ema = load_ema(self.model, decay=self.config.train.ema)

        if resume and self.config.ckpt is not None:
            ckpt_dict = torch.load(self.config.ckpt)
            state_dict = ckpt_dict['state_dict']
            if 'module.' in list(ckpt_dict["state_dict"].keys())[0]:
                state_dict = {k[7:]: v for k, v in ckpt_dict["state_dict"].items()}

            self.model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(ckpt_dict['optimizer'])
            self.ema.load_state_dict(ckpt_dict['ema'])
            start_epoch = int(self.config.ckpt.split("_")[-1][:-4])
            print(f'Loaded checkpoint {self.config.ckpt}')
        else:
            start_epoch = 0

        if not self.ddp or dist.get_rank() == 0:
            self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)
            logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
            logger.log(f'{self.ckpt}', verbose=False)
            start_log(logger, self.config)
            writer = SummaryWriter(os.path.join(*['logs_train', 'tensorboard', self.config.data.data,
                                                self.config.train.name, self.config.exp_name]))

        save_path = Path(f'./checkpoints/{self.config.data.data}/{self.config.train.name}/')
        save_path.mkdir(exist_ok=True, parents=True)

        sample_path = save_path.joinpath('samples')
        sample_path.mkdir(exist_ok=True, parents=True)

        self.loss_fn = CFMLoss(self.model, self.config)
        # self.sampling_fn = load_sampling_fn(self.config, self.config.sampler, self.config.sample, self.device,
        #                                     mode=self.config.model.mode)

        # -------- Training --------
        for epoch in trange(start_epoch, (self.config.train.num_epochs), desc = '[Epoch]', position = 1, leave=False):
            train_losses = {i:[] for i in ['total_loss','trans_loss', 'dist_loss']}
            self.model.train()
            if self.ddp:
                self.train_sampler.set_epoch(epoch)
                self.test_sampler.set_epoch(epoch)
            start_time = time.time()

            for _, train_b in enumerate(self.train_loader):
                train_b = recursive_to(train_b, self.device)
                self.optimizer.zero_grad()
                loss_dict = self.loss_fn(train_b)
                total_loss = loss_dict['trans_loss'] + loss_dict['dist_loss']
 
                if total_loss.isnan() or total_loss.isinf():
                    print('inf or nan detected.. skipping')
                    print((train_b['pdb'], train_b['chain']))
                    print(train_b)
                    continue

                total_loss.backward()

                if self.config.train.grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_norm)

                self.optimizer.step()
                
                # -------- EMA update --------
                self.ema.update(self.model.parameters())

                for k,v in loss_dict.items():
                    train_losses[k].append(v.mean().item())
                train_losses['total_loss'].append(total_loss.item())

            if self.config.train.lr_schedule:
                self.scheduler.step()

            self.model.eval()
            test_losses = {i:[] for i in ['total_loss','trans_loss', 'dist_loss']}
            with torch.no_grad():
                for _, test_b in enumerate(self.test_loader):
                    test_b = recursive_to(test_b, self.device)
                    loss_dict = self.loss_fn(test_b)
                    total_loss = loss_dict['trans_loss'] + loss_dict['dist_loss']

                    for k, v in loss_dict.items():
                        test_losses[k].append(v.mean().item())
                    test_losses['total_loss'].append(total_loss.item())

            if not self.ddp or dist.get_rank() == 0:
                log_msg = f'[EPOCH {epoch+1:04d}] | time: {time.time()-start_time:.2f} sec | '
                for k,v in train_losses.items():
                    mean_loss = np.mean(v)
                    writer.add_scalar(f'train_{k}', mean_loss, epoch+1)
                    log_msg += f'train_{k}: {mean_loss:.3e} | '

                for k,v in test_losses.items():
                    mean_loss = np.mean(v)
                    writer.add_scalar(f'test_{k}', mean_loss, epoch+1)
                    log_msg += f'test_{k}: {mean_loss:.3e} | '

                writer.flush()

                # -------- Log losses --------
                logger.log(log_msg, verbose=False)
                if epoch % self.config.train.print_interval == self.config.train.print_interval-1:
                    tqdm.write(log_msg)

                # -------- Save checkpoints --------
                if epoch % self.config.train.save_interval == self.config.train.save_interval-1:
                    save_name = f'_{epoch+1}' if epoch < self.config.train.num_epochs - 1 else ''
                    torch.save({
                        'config': self.config,
                        'state_dict': self.model.state_dict(),
                        'ema': self.ema.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                        }, save_path.joinpath(f'{self.ckpt + save_name}.pth'))

        print(' ')
        return self.ckpt

import yaml
from easydict import EasyDict as edict

def get_config(config, seed):
    config_dir = f'./config/training/{config}.yaml'
    config = edict(yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader))
    config.seed = seed

    return config

if __name__ == '__main__':
    import argparse
    import torch.distributed as dist
    import torch.utils.data.distributed

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--ddp', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')

    args = parser.parse_args()

    current_device = None
    if args.ddp:
        ngpus_per_node = torch.cuda.device_count()

        local_rank = int(os.environ.get("SLURM_LOCALID"))
        rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank

        current_device = local_rank

        torch.cuda.set_device(current_device)

        # init the process group
        dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size,
                                rank=rank)

    config = get_config(args.config, seed=args.seed)
    trainer = Trainer3D(config, ddp=args.ddp, device=current_device)
    trainer.train(time.strftime('%b%d-%H:%M:%S', time.gmtime()),resume=args.resume)
