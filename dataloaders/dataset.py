import torch
import numpy as np
import random
from torch.distributions import Categorical
from utils.constants import three_to_one_letter, letter_to_num, heavyatom_to_label
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from biotite.structure import connect_via_residue_names, filter_peptide_backbone
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

class ProteinDataset(Dataset):
    def __init__(self, dataset_path, code_to_idx=None, use_ncaa_splits=False, **kwargs):
        self.structures = torch.load(dataset_path, weights_only=False)

        self.code_to_struct = {}
        for i in self.structures:
            aa = i['res_aa']
            if aa not in self.code_to_struct.keys():
                self.code_to_struct[aa] = []
            self.code_to_struct[aa].append(i)

        if code_to_idx is not None:
            self.code_to_idx = torch.load(code_to_idx, weights_only=False)
            self.idx_to_code = {v:k for k,v in self.code_to_idx.items()}
            self.use_ncaa_splits = use_ncaa_splits
            self.idx_to_codeidx = {idx: self.code_to_idx[aa] for idx, aa in enumerate(self.code_to_struct.keys())}
        else:
            self.use_ncaa_splits = False

        uniform_freq = torch.tensor([1/len(self.code_to_struct) for _ in range(len(self.code_to_struct))])
        self.dist_sample = Categorical(probs=uniform_freq)

        print(f'Loaded {len(self.structures)} structures containing {len(self.code_to_struct)} NCAAs...')

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
        if self.use_ncaa_splits:
            sample_idx = self.dist_sample.sample((1,)).item()
            structure = random.choice(self.code_to_struct[self.idx_to_code[self.idx_to_codeidx[sample_idx]]])
        else:
            structure = self.structures[idx]

        data = structure['struct_data']
        res_idx = structure['res_idx']
        target_res = structure['res_aa']
        chain = structure['chain']
        ins_code = structure['ins_code']
        loss_mask = structure['loss_mask'] if 'loss_mask' in structure.keys() else None

        # remove all hydrogens
        h_mask = [True if i != 'H' else False for i in data.element]
        data = data[h_mask]
        loss_mask = loss_mask[h_mask]

        # remove all nans
        # nan_mask = np.isnan(data.coord)
        # data = data[~nan_mask]

        res_id_mask = data.res_id == res_idx
        chain_mask = data.chain_id == chain
        ins_code_mask = data.ins_code == ins_code
        target_mask = res_id_mask * ins_code_mask * chain_mask
        unique_res = np.unique(data[target_mask].res_name)
        if len(unique_res) > 1:
            remove_mask = (data.res_id == res_idx) * (data.res_name != target_res) * chain_mask
            data = data[~remove_mask]
            loss_mask = loss_mask[~remove_mask]

        backbone_mask = torch.LongTensor(np.isin(data.atom_name, ['N','CA','C','O']))

        coords = torch.tensor(data.coord)
        # aa = torch.LongTensor([self.code_to_idx.get(i,self.code_to_idx['UNK']) for i in data.res_name])
        atom_type = torch.LongTensor([heavyatom_to_label.get(i,heavyatom_to_label['X']) for i in data.element])
        ncaa_mask = torch.LongTensor((data.res_id == res_idx) * (data.ins_code == ins_code) * (data.chain_id == chain))
        complex_mask = 1-ncaa_mask
        # center coords on ca of target
        # other_mask = (1-mask).unsqueeze(-1)
        # if other_mask.sum() == 0:
        #     other_mask = torch.ones_like(other_mask) # just center to origin if no context
        # origin = (coords * other_mask).sum(0) / other_mask.sum(0).clamp(min=1)
        # coords = coords - origin.view(1,-1)
        chain_mask = torch.Tensor(data.chain_id == chain)
        target_ca_mask = torch.Tensor(data.atom_name == 'CA') * ncaa_mask * chain_mask
        target_ca = coords[target_ca_mask==1]

        if target_ca.shape[0] != 1:
            print(data)
            print(target_ca)
            print(backbone_mask)
            print(chain_mask)
            print(data.atom_name == 'CA')
            print(data.res_id)
            print(data.ins_code)
            assert False
        coords = coords - target_ca

        bond_type = connect_via_residue_names(data, inter_residue=True).bond_type_matrix()
        bond_type[bond_type == 0] = 1  # any type to single
        bond_type[bond_type == -1] = 0  # -1 is empty, set to 0
        bond_type[bond_type == 4] = 3  # just set quadruple bonds to triple (these rarely exist)
        bond_type[bond_type >= 5] = 4  # set aromatic to 4
        bond_type = torch.LongTensor(bond_type)



        # res_idx = torch.LongTensor(data.res_id)
        # seq_relpos = torch.clamp(res_idx[edges[0]] - res_idx[edges[1]], min=-32,max=32) + 32
        # seq_relpos = torch.where(chain_mask, seq_relpos, 65)
        # seq_relpos = F.one_hot(seq_relpos, num_classes=66)
        #
        # tok_idx = torch.LongTensor(structure['tok_idx'])
        # tok_relpos = torch.clamp(tok_idx[edges[0]] - tok_idx[edges[1]], min=-32,max=32) + 32
        # tok_relpos = torch.where(chain_mask, tok_relpos, 65)
        # tok_relpos = F.one_hot(tok_relpos, num_classes=66)
        #
        # edge_attr = torch.cat([seq_relpos, tok_relpos, edge_bond_attr, chain_mask[...,None].long()],dim=-1)
        # ca_mask = torch.Tensor(data.atom_name == 'CA').float()

        mask = torch.ones(len(data))

        out = {
            'id': f'{structure["pdb"]}_{chain}_{target_res}{res_idx}',
            'res_id': str(res_idx),
            'pos': coords,
            'atom_type_str': data.atom_name.tolist(),
            'atom_type': atom_type,
            'mask': mask,
            'complex_mask': complex_mask,
            'ncaa_mask': ncaa_mask,
            'adj': bond_type,
            'backbone_mask': backbone_mask,
            'loss_mask': loss_mask,
            'target_ca_mask': target_ca_mask
        }

        return out

    def __len__(self):
        if self.use_ncaa_splits:
            return len(self.code_to_idx)
        else:
            return len(self.structures)

def get_dataloader(config, sample=False, ddp=False):
    batch_size = config.sample.batch_size if sample else config.train.batch_size

    if not sample:
        train_ds = ProteinDataset(dataset_path=config.data.train_path, **config.data)
    test_ds = ProteinDataset(dataset_path=config.data.test_path, **config.data, test=True)

    if ddp:
        train_sampler = DistributedSampler(train_ds)
        test_sampler = DistributedSampler(test_ds)
        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                              collate_fn=PaddingCollate())
        test_dl = DataLoader(test_ds, batch_size=batch_size, sampler=test_sampler,
                             collate_fn=PaddingCollate())
        return train_dl, test_dl, train_sampler, test_sampler
    else:
        if not sample:
            train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True,
                                  collate_fn=PaddingCollate())
        else:
            train_dl = None
        test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=0, shuffle=True,
                             collate_fn=PaddingCollate())
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
                return F.pad(x, (0, pad_size, 0, pad_size))
            else:
                pad_size = [n - x.size(0)] + list(x.shape[1:])
                pad = torch.full(pad_size, fill_value=value).to(x)
                return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_value(k):
        if k in ['id','atom_type_str','res_idx']:
            return ''
        else:
            return 0

    def __call__(self, data_list):
        max_length = self.max_len if self.max_len else max([len(data["pos"]) for data in data_list])
        data_list_padded = []
        for data in data_list:
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_value(k)) for k, v in data.items()
            }
            data_list_padded.append(data_padded)
        return default_collate(data_list_padded)