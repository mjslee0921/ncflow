import torch
import numpy as np
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data.distributed import DistributedSampler

class ProteinDataset(Dataset):
    def __init__(self, data, **kwargs):
        self.id = data['id'][0]
        self.data = data['data']

        # print(f'Loaded {self.id} with {len(self.data)} structures...')

    def to_tensor(self, d, exclude=[]):
        feat_dtypes = {
            "id": None,
            "chain": None,
            "atom_type": torch.long,
            "aa": torch.long,
            "aa_mask": torch.long,
        }

        for x in exclude:
            del d[x]

        for k,v in d.items():
            if type(v) == dict:
                d[k] = self.to_tensor(v)
            elif type(v) == list or type(v) == np.ndarray:
                if feat_dtypes.get(k, True) is not None:
                    d[k] = torch.tensor(v).to(dtype=feat_dtypes.get(k,torch.float32))

        return d

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = {k:v[0] if k != 'atom_type_str' else v for k,v in sample.items()} # remove batch dimension
        atom_types, coords, adj, atom_type_str = sample['atom_type'], sample['coords'], sample['adj'], sample['atom_type_str']

        coords = coords - coords.mean(0).unsqueeze(0) # remove CoM

        length = atom_types.shape[0]
        res_idx = torch.arange(length)
        mask = torch.ones(length)
        atom_type_str = [i[0] for i in atom_type_str]

        return {
            'id': f'{self.id}_{idx}',
            'res_id': res_idx,
            'pos': coords,
            'atom_type_str': atom_type_str,
            'atom_type': atom_types,
            'atom_name': atom_types,
            'mask': mask,
            'adj': adj
        }

    def __len__(self):
        return len(self.data)

class SplitDataset(Dataset):
    def __init__(self, dataset_path, **kwargs):
        self.structures = [i for i in Path(dataset_path).iterdir() if i.suffix == '.pth']
        print(f'Loaded {len(self.structures)} splits...')

    def __getitem__(self, idx):
        path = self.structures[idx]
        data = torch.load(path)
        return {'id': path.stem, 'data': data}

    def __len__(self):
        return len(self.structures)

def construct_dataloader(data, batch_size=8, ddp=False):
    ds = ProteinDataset(data)
    sampler = DistributedSampler(ds) if ddp else None
    dl = DataLoader(ds, batch_size=batch_size, sampler=sampler, collate_fn=PaddingCollate())
    return dl, sampler

def get_dataloader(config, sample=False, ddp=False):
    if not sample:
        train_ds = SplitDataset(dataset_path=config.data.train_path, **config.data)
    test_ds = SplitDataset(dataset_path=config.data.test_path, **config.data)

    if ddp:
        train_sampler = DistributedSampler(train_ds)
        test_sampler = DistributedSampler(test_ds)
        train_dl = DataLoader(train_ds, batch_size=1, sampler=train_sampler)
        test_dl = DataLoader(test_ds, batch_size=1, sampler=test_sampler)
        return train_dl, test_dl, train_sampler, test_sampler
    else:
        if not sample:
            train_dl = DataLoader(train_ds, batch_size=1, num_workers=0, shuffle=True)
        else:
            train_dl = None
        test_dl = DataLoader(test_ds, batch_size=1, num_workers=0, shuffle=True)
        return train_dl, test_dl, None, None

from torch.utils.data._utils.collate import default_collate

class PaddingCollate(object):

    def __init__(self, max_len=None):
        super().__init__()
        self.max_len = max_len

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x

            if len(x.shape) == 2 and x.size(0) == x.size(1):
                pad_size = n - x.size(0)
                return F.pad(x, (0,pad_size,0,pad_size))
            else:
                pad_size = [n - x.size(0)] + list(x.shape[1:])
                pad = torch.full(pad_size, fill_value=value).to(x)
                return torch.cat([x, pad], dim=0)
        elif isinstance(x, str):
            pad = value * (n - len(x))
            return x + pad
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_value(k):
        if k == 'id' or k == 'atom_type_str':
            return ''
        else:
            return 0

    def __call__(self, data_list):
        max_length = self.max_len if self.max_len else max([len(data["pos"]) for data in data_list])
        data_list_padded = []
        for data in data_list:
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_value(k)) for k,v in data.items()
            }
            data_list_padded.append(data_padded)
        return default_collate(data_list_padded)