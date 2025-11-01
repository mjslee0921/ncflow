import torch

three_to_one_letter = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 'UNK': 'X'}

one_to_three_letter = {v:k for k,v in three_to_one_letter.items()}

letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                       'N': 2, 'Y': 18, 'M': 12, 'X': 20}

num_to_letter = {v:k for k, v in letter_to_num.items()}

restype_to_heavyatom_names = {
    "ALA": ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],
    "ARG": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],
    "ASN": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],
    "ASP": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],
    "CYS": ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],
    "GLN": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],
    "GLU": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],
    "GLY": ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    "HIS": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],
    "ILE": ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],
    "LEU": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],
    "LYS": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],
    "MET": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],
    "PHE": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],
    "PRO": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],
    "SER": ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],
    "THR": ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    "TRP": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    "TYR": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],
    "VAL": ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    "UNK": ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
}

heavyatoms = ['C','N','O','S','P','F','B','BR','AS','SE','CL','BE','D','HG','I','H','X']
heavyatom_to_label = {i:idx for idx,i in enumerate(heavyatoms)}
label_to_heavyatom = {v:k for k,v in heavyatom_to_label.items()}
label_to_heavyatom = {v:k for k,v in heavyatom_to_label.items()}

atom_to_number = {
    'C': 6,
    'N': 7,
    'O': 8,
    'P': 15,
    'F': 9,
    'B': 5,
    'BR': 35,
    'AS': 33,
    'SE': 34,
    'CL': 17,
    'BE': 4,
    'D': 1,
    'I': 53,
    'H': 1,
    'X': 0,
    'S': 16
}

van_der_waals_radius = {
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "S": 1.8,
}

vdw_tensor = torch.tensor([1.7, 1.55, 1.52, 1.8, 0.0]) # 0.0 for 'X'

max_num_heavy_atoms = len(restype_to_heavyatom_names["ALA"])

atom_types = [
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37.

atom37_to_14_mask = torch.zeros(21,37)
atom14_mask = torch.zeros(21,14)

for aa,atom_list in restype_to_heavyatom_names.items():
    aa_idx = letter_to_num[three_to_one_letter[aa]]
    for atom_idx,atom in enumerate(atom_list):
        if atom == '': continue
        atom37_to_14_mask[aa_idx,atom_types.index(atom)] = 1
        atom14_mask[aa_idx, atom_idx] = 1

canonical_smiles = {
    'ALA': 'C[C@@H](C(=O)O)N',
    'CYS': 'C([C@@H](C(=O)O)N)S',
    'TYR': 'C1=CC(=CC=C1C[C@@H](C(=O)O)N)O',
    'ASP': 'C([C@@H](C(=O)O)N)C(=O)O',
    'GLN': 'C(CC(=O)N)[C@@H](C(=O)O)N',
    'GLU': 'C(CC(=O)O)[C@@H](C(=O)O)N',
    'HIS': 'C1=C(NC=N1)C[C@@H](C(=O)O)N',
    'SER': 'C([C@@H](C(=O)O)N)O',
    'LYS': 'C(CCN)C[C@@H](C(=O)O)N',
    'GLY': 'C(C(=O)O)N',
    'ILE': 'CC[C@H](C)[C@@H](C(=O)O)N',
    'LEU': 'CC(C)C[C@@H](C(=O)O)N',
    'ASN': 'C([C@@H](C(=O)O)N)C(=O)N',
    'MET': 'CSCC[C@@H](C(=O)O)N',
    'PHE': 'C1=CC=C(C=C1)C[C@@H](C(=O)O)N',
    'VAL': 'CC(C)[C@@H](C(=O)O)N',
    'PRO': 'C1C[C@H](NC1)C(=O)O',
    'THR': 'C[C@H]([C@@H](C(=O)O)N)O',
    'ARG': 'C(C[C@@H](C(=O)O)N)CN=C(N)N',
    'TRP': 'C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N'
}