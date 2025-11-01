from utils.constants import heavyatom_to_label
import torch
from rdkit import Chem
import gzip
from pathlib import Path
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map,thread_map
from utils.rdkit_utils import get_coordinates, get_bond_matrix
import itertools
import numpy as np

CHUNKSIZE = 1000

def write_coords(p, outpath):
   inf = gzip.open(str(p))
   idx = 0
   with Chem.ForwardSDMolSupplier(inf) as gzsuppl:
      for mol in tqdm(gzsuppl):
         if not mol: continue
         atom_type_str = [atom.GetSymbol().upper() for atom in mol.GetAtoms()]
         atom_types = torch.LongTensor([heavyatom_to_label.get(a,16) for a in atom_type_str])
         if len(atom_types) <= 5: continue
         adj = get_bond_matrix(mol).float()
         coords = get_coordinates(mol)
         idx += 1

         data = {
             'atom_type_str': atom_type_str,
             'atom_type': atom_types,
             'adj': adj,
             'coords': coords
         }

         torch.save(data, outpath.joinpath(f'{p.stem.split(".")[0]}_{idx}.pth'))

def chunk_files(path, outpath):
    idx, path = path
    outpath = outpath.joinpath(f'split{idx+1}.pth')
    if Path(outpath).exists(): return
    out = [torch.load(p) for p in path]
    torch.save(out,outpath)

def main(args):
    path = Path(args.path)
    outpath = Path(args.outpath)
    outpath_raw = outpath.joinpath('raw')
    outpath_raw.mkdir(parents=True, exist_ok=True)
    all_paths = [p for p in path.iterdir() if p.suffix == '.gz']
    _ = process_map(write_coords, all_paths, itertools.repeat(outpath_raw), chunksize=1)

    num_files = len(list(outpath_raw.iterdir()))
    paths_split = np.array_split(list(outpath_raw.iterdir()), num_files//CHUNKSIZE)
    paths_split = [[idx, i.tolist()] for idx, i in enumerate(paths_split)]

    _ = process_map(chunk_files, paths_split, itertools.repeat(outpath), chunksize=1)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('outpath', type=str)
    args = parser.parse_args()
    main(args)