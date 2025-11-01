import pandas as pd
from rdkit import Chem
from utils.rdkit_utils import reorder_mol, get_bond_matrix, get_bond_list, get_bond_list_kek, get_coordinates, construct_biotite
from utils.constants import three_to_one_letter
import torch
import numpy as np

amino_acids = [
    {"name": "Homocysteine", "three_letter": "HCY", 'smiles': 'C(CS)[C@@H](C(=O)O)N'}, # L
    {"name": "Homoserine", "three_letter": "HSE", 'smiles': 'C(CO)[C@@H](C(=O)O)N'},
    {"name": "Norleucine", "three_letter": "NLE", 'smiles': 'CCCC[C@@H](C(=O)O)N'},
    {"name": "Norvaline", "three_letter": "NVA", 'smiles': 'CCC[C@@H](C(=O)O)N'},
]

df = pd.DataFrame(amino_acids)

out = {}
for k,v in zip(df['three_letter'],df['smiles']):
    mol = Chem.MolFromSmiles(v)
    mol = reorder_mol(mol,use_fmoc=False)
    if not mol:
        print(k)
        print(v)
        continue
    mol = Chem.RWMol(mol)
    mol.RemoveAtom(4) # Remove O

    atom_types = [atom.GetSymbol().upper() for atom in mol.GetAtoms()]
    bond_matrix = get_bond_matrix(mol)
    atoms_no_bonds = bond_matrix.sum(-1) == 0
    if torch.any(atoms_no_bonds).item(): continue # Remove any ncaas that have nonconnected atoms for some reason
    bond_list = get_bond_list(mol)

    bond_list_biotite = np.array(get_bond_list_kek(mol))

    smiles_new = Chem.MolToSmiles(mol)
    if '.' in smiles_new: continue # if there are more than one fragment now

    biotite_array = construct_biotite(atom_types,bond_list_biotite)
    for res in three_to_one_letter.keys():
        if res not in out.keys():
            out[res] = []

        out[res].append({
            'id': k,
            'smiles': smiles_new,
            'biotite': biotite_array,
            'atom_types': atom_types,
            'bond_matrix': bond_matrix,
        })

torch.save(out, './data/ncaa/test.pth')