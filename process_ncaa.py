import torch
from utils.constants import three_to_one_letter, letter_to_num
import requests
import copy
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import biotite.structure as bstruct
import biotite.structure.io as bsio
from scipy.spatial import KDTree
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
import random

path = Path('./data/ncaa/raw')

canonical = list(three_to_one_letter.keys())
contains_ncaa = []
metrics = {}
list_of_paths = [p2 for p1 in path.iterdir() for p2 in p1.iterdir()]

def run_analysis(path):
    metrics = {}
    structure = bsio.load_structure(path, model=1)
    structure = structure[bstruct.filter_amino_acids(structure)]
    if not structure: return None
    _, aa = bstruct.get_residues(structure)
    ncaa = False
    for i in aa:
        if i not in canonical:
            ncaa = True
            if i not in metrics.keys():
                metrics[i] = 0
            metrics[i] += 1
    if ncaa:
        return {'path': path, 'metrics':metrics}
    else:
        return None

out = process_map(run_analysis, list_of_paths, chunksize=100)
out = [i for i in out if i]

all_metrics = {}
total_count = 0
for i in out:
    for k,v in i['metrics'].items():
        if k not in all_metrics.keys():
            all_metrics[k] = 0
        all_metrics[k] += v
        total_count += v

code_data = {}
for code in tqdm(list(all_metrics.keys())):
    result = requests.get(f'https://www.ebi.ac.uk/pdbe/api/pdb/compound/summary/{code}')
    result = result.json()[code][0]
    code_data[code] = {
        'name': result['name'],
        'smiles': result['smiles'][-1],
    }

torch.save(code_data, './data/ncaa/processed/code_data.pth')

exclude = []
exclude += ['MSE','SEC']
# exclude += [i for i in code_data.keys() if all_metrics[i] < 10] # for nonredundant test set
exclude += [i for i in code_data.keys() if all_metrics[i] >= 10]

out_filtered = copy.deepcopy(out)
for i in out_filtered:
    compounds = list(i['metrics'].keys())
    for c in compounds:
        if c in exclude:
            i['metrics'].pop(c)

filtered = {}
total_count = 0
for i in out_filtered:
    for k,v in i['metrics'].items():
        if k not in filtered.keys():
            filtered[k] = 0
        filtered[k] += v
        total_count += v

code_data_filtered = {k:v for k,v in code_data.items() if k in filtered.keys()}
torch.save(code_data_filtered,'./data/ncaa/processed/code_data_filtered.pth')

outPath = Path('./data/ncaa/processed/structures')
outPath.mkdir(exist_ok=True)
for i in out_filtered:
    shutil.copy(i['path'], outPath)

canonical = list(three_to_one_letter.keys())

def process_structures_knn(p):
    all_structures = []
    structure = bsio.load_structure(p,model=1)
    structure = structure[bstruct.filter_amino_acids(structure)]
    h_mask = [True if i != 'H' else False for i in structure.element]
    structure = structure[h_mask]
    if not structure: return

    _, seq = bstruct.get_residues(structure)
    tree = KDTree(structure.coord)

    # get token indices
    tok_id = []
    for res_idx, res in enumerate(bstruct.residue_iter(structure)):
        for atom_idx, atom in enumerate(res):
            tok_id.append(atom_idx)
    tok_id = np.array(tok_id)

    assert len(tok_id) == len(structure)

    for res_idx, res in enumerate(bstruct.residue_iter(structure)):
        res_aa = res.res_name[0]
        # if res_idx < 5 or res_idx > length-5: continue # no ncaas in n/c terminus
        if len(res.coord) <= 4: continue # if only backbone coordinates
        res_ca = res[res.atom_name == 'CA'].coord
        res_id = res.res_id[0]
        ins_code = res.ins_code[0]
        chain_id = res.chain_id[0]
        if res_ca.size == 0: continue
        if res_aa in code_data.keys():
            sc_coords = res[~bstruct.filter_peptide_backbone(res)]

            # identify interacting atoms with ncAA
            indices = tree.query_ball_point(sc_coords.coord,r=3.5)
            unique_res_indices = []
            for i in indices:
                unique_res_indices += list(i)
            unique_res_indices = np.unique(unique_res_indices)
            res_pos = bstruct.get_residue_positions(structure,unique_res_indices)
            res_starts = bstruct.get_residue_starts(structure,add_exclusive_stop=True)
            full_indices = []
            for i in res_pos:
                full_indices.append(
                    np.arange(res_starts[i],res_starts[i+1])
                )
            full_indices = np.unique(np.concatenate(full_indices))
            loss_mask = np.zeros(len(structure))
            loss_mask[full_indices] = 1

            # isolate nearest 200 atoms
            k = min(200, len(structure.coord))
            dists, indices = tree.query(res_ca,k=k)
            if (dists<0.1).sum() != 1: continue
            indices = np.unique(np.concatenate([indices[0], full_indices]))
            selected = structure[indices]

            # loss_mask
            loss_mask = loss_mask[indices].astype(bool)

            tok_id_selected = tok_id[indices]
            d = code_data[res_aa]

            # adj_matrix = bonds.adjacency_matrix()

            all_structures.append({
                'pdb': p.stem,
                'chain': chain_id,
                'res_aa': res_aa,
                'res_idx': res_id,
                # 'tok_idx': tok_id_selected,
                'ins_code': ins_code,
                'smiles': d['smiles'],
                'name': d['name'],
                'struct_data': selected,
                'loss_mask': loss_mask
            })

    return all_structures

data = process_map(process_structures_knn, list(Path('./data/ncaa/processed/structures').iterdir()),chunksize=10)

data_flattened = []
for i in data:
    if not i: continue
    data_flattened += i

torch.save(data_flattened, './data/ncaa/processed/data_256.pth')

random.shuffle(data_flattened)
num_train = int(0.9*len(data_flattened))
train_data, test_data = data_flattened[:num_train], data_flattened[num_train:]
torch.save(train_data,'./data/ncaa/processed/train_data_256.pth')
torch.save(test_data,'./data/ncaa/processed/test_data_256.pth')

code_to_struct = {}
for i in data_flattened:
    aa = i['res_aa']
    if aa not in code_to_struct.keys():
        code_to_struct[aa] = []
    code_to_struct[aa].append(i)

idx_to_code = {idx:k for idx,k in enumerate(code_to_struct.keys())}

# construct code to label
num_to_three = {(letter_to_num[v]+len(idx_to_code)):k for k,v in three_to_one_letter.items()}
idx_to_code = {**idx_to_code, **num_to_three}
code_to_idx = {v:k for k,v in idx_to_code.items()}
torch.save(code_to_idx, './data/ncaa/processed/code_to_idx.pth')