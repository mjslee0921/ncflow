import torch
import random
import numpy as np
import math
from easydict import EasyDict as edict
from utils.ema import ExponentialMovingAverage


def load_seed(seed):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def load_device():
    if torch.cuda.is_available():
        device = list(range(torch.cuda.device_count()))
    else:
        device = 'cpu'
    return device

def load_ema(model, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    return ema

def load_checkpoint(model, ema, ckpt_dict):
    if 'module' in list(ckpt_dict['state_dict'].keys())[0]:  # if trained with DDP
        ckpt_dict['state_dict'] = {k[7:]: v for k, v in ckpt_dict['state_dict'].items()}
    model.load_state_dict(ckpt_dict['state_dict'])
    ema.load_state_dict(ckpt_dict['ema'])

    return model, ema