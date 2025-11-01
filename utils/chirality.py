# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Chirality detection and comparison."""

from collections.abc import Mapping

from absl import logging
# from alphafold3 import structure
# from alphafold3.constants import chemical_components
# from alphafold3.data.tools import rdkit_utils
import rdkit.Chem as rd_chem

_CHIRAL_ELEMENTS = frozenset({'C', 'S'})


def _find_chiral_centres(mol: rd_chem.Mol) -> dict[str, str]:
  """Find chiral centres and detect their chirality.

  Only elements listed in _CHIRAL_ELEMENTS are considered as centres.

  Args:
    mol: The molecule for which to detect chirality.

  Returns:
    Map from chiral centre atom names to identified chirality.
  """
  chiral_centres = rd_chem.FindMolChiralCenters(
      mol, force=True, includeUnassigned=False, useLegacyImplementation=True
  )
  atom_name_by_idx = {
      atom.GetIdx(): atom.GetProp('atom_name') for atom in mol.GetAtoms()
  }
  atom_chirality_by_name = {atom_name_by_idx[k]: v for k, v in chiral_centres}
  return {
      k: v
      for k, v in atom_chirality_by_name.items()
      if any(k[: len(el)].upper() == el for el in _CHIRAL_ELEMENTS)
  }


def _chiral_match(mol1: rd_chem.Mol, mol2: rd_chem.Mol) -> bool:
  """Compares chirality of two Mols. Mol1 can match a subset of mol2."""

  mol1_atom_names = {a.GetProp('atom_name') for a in mol1.GetAtoms()}
  mol2_atom_names = {a.GetProp('atom_name') for a in mol2.GetAtoms()}
  if mol1_atom_names != mol2_atom_names:
    if not mol1_atom_names.issubset(mol2_atom_names):
      raise ValueError('Mol1 atoms are not a subset of mol2 atoms.')

  mol1_chiral_centres = _find_chiral_centres(mol1)
  mol2_chiral_centres = _find_chiral_centres(mol2)
  if set(mol1_chiral_centres) != set(mol2_chiral_centres):
    if not set(mol1_chiral_centres).issubset(mol2_chiral_centres):
      return False
  chirality_matches = {
      centre_atom: chirality1 == mol2_chiral_centres[centre_atom]
      for centre_atom, chirality1 in mol1_chiral_centres.items()
      if '?' != mol2_chiral_centres[centre_atom]
  }
  return all(chirality_matches.values())