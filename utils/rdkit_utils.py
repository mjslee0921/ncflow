import torch
from rdkit import Chem
import biotite.structure as bstruct
import biotite.structure.io as bsio
import numpy as np

def get_coordinates(mol):
  coordinates = []
  for i, atom in enumerate(mol.GetAtoms()):
      positions = mol.GetConformer().GetAtomPosition(i)
      coordinates.append([positions.x, positions.y, positions.z])

  return torch.Tensor(coordinates)

def get_bond_matrix(mol):
    N = mol.GetNumAtoms()
    adjs = torch.zeros((N, N))

    bond_type_to_channel = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4
    }
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        val = bond_type_to_channel.get(bond_type,0)
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjs[i, j] = val
        adjs[j, i] = val
    return adjs

def get_bond_list(mol):
    bond_list = []
    bond_type_to_channel = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4
    }
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        val = bond_type_to_channel.get(bond_type,0)
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_list.append((i,j,val))
    return bond_list

def get_bond_list_kek(mol):
    bond_list = []
    bond_type_to_channel = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
    }
    Chem.Kekulize(mol)
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        val = bond_type_to_channel.get(bond_type,0)
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if bond.GetIsAromatic() and val != 0:
            val += 4
        bond_list.append((i,j,val))
    return bond_list

def reorder_mol(mol, use_fmoc=True):
    if use_fmoc:
        substruct = Chem.MolFromSmiles('OC(=O)CNC(=O)OCC1c2ccccc2-c2ccccc12')
    else:
        substruct = Chem.MolFromSmiles('NCC(=O)O')
    matches = mol.GetSubstructMatches(substruct)

    if len(matches) != 1: return None
    matches = list(matches[0])

    for i in range(mol.GetNumAtoms()):
        if i in matches: continue
        matches += [i]

    mol_renumbered = Chem.RenumberAtoms(mol, matches)
    return mol_renumbered

def reorder_mol_by_another(mol1, mol2):
    matches = mol1.GetSubstructMatches(mol2)

    if len(matches) != 1: return None
    matches = list(matches[0])

    for i in range(mol1.GetNumAtoms()):
        if i in matches: continue
        matches += [i]

    mol_renumbered = Chem.RenumberAtoms(mol1, matches)
    return mol_renumbered

def construct_biotite(atom_type, bond_list):
    atom_array = bstruct.AtomArray(len(atom_type))
    for idx,i in enumerate(atom_type):
        if idx == 1:
            atom_name = 'CA'
        elif idx < 4:
            atom_name = i
        else:
            atom_name = f'{i}{idx}'
        atom = bstruct.Atom([0,0,0], chain_id='A', element=i, atom_name=atom_name, hetero=True)
        atom_array[idx] = atom
        bond_list_in = bstruct.BondList(atom_count=len(atom_type), bonds=bond_list)
        atom_array.bonds = bond_list_in
    return atom_array