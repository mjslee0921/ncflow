from pathlib import Path
from tqdm.auto import tqdm
import zipfile
import shutil
import biotite.structure as bstruct
import biotite.structure.io as bsio
from utils.constants import heavyatom_to_label
import torch
from sklearn.neighbors import KDTree
import numpy as np
from tqdm.contrib.concurrent import process_map, thread_map
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

#### CHANGE HARDCODED INPUT/OUTPUT PATHS BELOW
INPUT_PATH = Path('./data/plinder/raw')
OUTPUT_PATH = Path('./data/plinder/processed')

#### THE REST IS FINE

struct_path = OUTPUT_PATH.joinpath('structures')
pth_path = OUTPUT_PATH.joinpath('pth')
struct_path.mkdir(parents=True, exist_ok=True)
pth_path.mkdir(parents=True, exist_ok=True)

tmp_path = INPUT_PATH.joinpath('tmp')
tmp_path.mkdir(exist_ok=True)

all_zips = [i for i in list(INPUT_PATH.iterdir()) if i.suffix == '.zip']

def unzip(path):
    tmp = tmp_path.joinpath(path.stem)
    tmp.mkdir(exist_ok=True)
    with zipfile.ZipFile(path, 'r') as zip:
        zip.extractall(tmp)

    for p2 in tmp.iterdir():
        id = p2.name
        outpath = struct_path.joinpath(id)
        if outpath.exists(): continue
        outpath.mkdir(exist_ok=True)
        system_path = p2.joinpath('system.cif')
        ligand_path = p2.joinpath('ligand_files')
        if not system_path.exists() or not ligand_path.exists(): continue
        shutil.move(p2.joinpath('system.cif'), outpath.joinpath('system.cif'))
        shutil.move(p2.joinpath('ligand_files'), outpath.joinpath('ligand_files'))
    shutil.rmtree(tmp)

_ = thread_map(unzip, all_zips)

def extract_features(path):
    pdb, model, receptor_chain, ligand_chains = path.name.split('__')
    try:
        structure = bsio.load_structure(path.joinpath('system.cif'))
    except FileNotFoundError:
        return None
    receptor = structure[structure.chain_id == receptor_chain]
    ligand_len = 0
    for i in ligand_chains.split('_'):
        ligand = structure[structure.chain_id == i]
        if len(ligand) > ligand_len:
            ligand_len = len(ligand)
            longest_ligand = ligand
            longest_ligand_chain = i

    if len(longest_ligand) < 5 or len(longest_ligand) > 64: return None
    if len(receptor) < 30: return None

    # filter to up to 128 receptor atoms
    tree = KDTree(receptor.coord)
    ligand_centroid = longest_ligand.coord.mean(0, keepdims=True)
    dist, indices = tree.query(ligand_centroid, k=min(len(receptor), (128 - len(longest_ligand))))
    indices = np.sort(indices)
    selected = receptor[indices[0]]

    struct_filtered = bstruct.concatenate([selected, longest_ligand])

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
        'adj': bond_type
    }

    torch.save(data, pth_path.joinpath(f'{path.stem}.pth'))
    return path

_ = process_map(extract_features, list(struct_path.iterdir()), chunksize=10)