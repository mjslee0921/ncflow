import torch
import numpy as np
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
import biotite.structure as bstruct
import biotite.structure.io as bsio
import torch.nn.functional as F
from pathlib import Path
from utils.constants import heavyatom_to_label
from torch.utils.data.distributed import DistributedSampler
from sklearn.neighbors import KDTree
import warnings

class PlinderDataset(Dataset):
    def __init__(self, dataset_path, **kwargs):
        warnings.filterwarnings("ignore", category=UserWarning)
        self.dataset_path = list(Path(dataset_path).iterdir())
        # self.data = process_map(self.extract_features, self.dataset_path,chunksize=10)
        # self.data = [i for i in self.data if i is not None]
        print(f'Loaded {len(self.dataset_path)} structures...')

    def to_tensor(self, d, exclude=[]):
        feat_dtypes = {
            "id": None,
            "atom_type_str": None,
            "atom_type": torch.long,
            "aa": torch.long,
            'mask': torch.long,
            'ligand_mask': torch.long,
            'receptor_mask': torch.long,
            'adj': torch.long,

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

    def extract_features(self, path):
        pdb, model, receptor_chain, ligand_chains = path.stem.split('__')
        structure = bsio.load_structure(path)
        # NOTE: THIS IS A PROBLEM WITH BIOTITE 0.42, WHERE IT DOESNT SEEM TO RECOGNIZE CHAIN_IDS LONGER THAN LENGTH 4
        # HACKY IMPLEMENTATION SINCE I CAN'T INSTALL BIOTITE > 1.0 ON ALLIANCE CANADA
        if len(receptor_chain) > 4:
            receptor_chain = receptor_chain[:4]
        receptor = structure[structure.chain_id == receptor_chain]
        ligand_len = 0
        for i in ligand_chains.split('_'):
            if len(i) > 4:
                i = i[:4]
            ligand = structure[structure.chain_id == i]
            if len(ligand) > ligand_len:
                ligand_len = len(ligand)
                longest_ligand = ligand
                longest_ligand_chain = i

        # filter to up to 128 receptor atoms
        tree = KDTree(receptor.coord)
        ligand_centroid = longest_ligand.coord.mean(0, keepdims=True)
        dist, indices = tree.query(ligand_centroid, k=min(len(receptor),(128 - len(longest_ligand))))
        indices = np.sort(indices)
        selected = receptor[indices[0]]

        struct_filtered = concatenate([selected, longest_ligand])

        bond_type = bstruct.connect_via_residue_names(struct_filtered, inter_residue=True).bond_type_matrix()
        bond_type[bond_type == 0] = 1  # any type to single
        bond_type[bond_type == -1] = 0  # -1 is empty, set to 0
        bond_type[bond_type == 4] = 3  # just set quadruple bonds to triple (these rarely exist)
        bond_type[bond_type >= 5] = 4  # set aromatic to 4

        atom_type = torch.LongTensor(
            [heavyatom_to_label.get(i, heavyatom_to_label['X']) for i in struct_filtered.element])
        bond_type = torch.LongTensor(bond_type)

        atom_name = struct_filtered.atom_name
        coords = struct_filtered.coord
        receptor_mask = torch.LongTensor(struct_filtered.chain_id == receptor_chain)
        ligand_mask = torch.LongTensor(struct_filtered.chain_id == longest_ligand_chain)

        coords = coords - ligand_centroid  # remove CoM of ligand

        length = atom_type.shape[0]
        res_idx = torch.arange(length)
        mask = torch.ones(length)

        data = {
            'id': path.name,
            'res_id': res_idx,
            'pos': coords,
            'atom_type_str': atom_name.tolist(),
            'atom_type': atom_type,
            'mask': mask,
            'ligand_mask': ligand_mask,
            'receptor_mask': receptor_mask,
            'adj': bond_type,
        }

        return data

    def __getitem__(self, idx):
        sample_path = self.dataset_path[idx]
        # data = self.extract_features(sample_path)
        data = torch.load(sample_path, weights_only=False)
        return self.to_tensor(data)

    def __len__(self):
        return len(self.dataset_path)

def get_dataloader(config, sample=False, ddp=False):
    if not sample:
        train_ds = PlinderDataset(dataset_path=config.data.train_path, **config.data)
    test_ds = PlinderDataset(dataset_path=config.data.test_path, **config.data)

    if ddp:
        train_sampler = DistributedSampler(train_ds)
        test_sampler = DistributedSampler(test_ds)
        train_dl = DataLoader(train_ds, batch_size=config.train.batch_size, sampler=train_sampler, collate_fn=PaddingCollate(128))
        test_dl = DataLoader(test_ds, batch_size=config.train.batch_size, sampler=test_sampler, collate_fn=PaddingCollate(128))
        return train_dl, test_dl, train_sampler, test_sampler
    else:
        if not sample:
            train_dl = DataLoader(train_ds, batch_size=config.train.batch_size, num_workers=0, shuffle=True, collate_fn=PaddingCollate(128))
        else:
            train_dl = None
        test_dl = DataLoader(test_ds, batch_size=config.train.batch_size, num_workers=0, shuffle=True, collate_fn=PaddingCollate(128))
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