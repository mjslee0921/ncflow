from Bio.PDB import StructureBuilder
from Bio.PDB.PDBIO import PDBIO
from utils.constants import num_to_letter, one_to_three_letter, restype_to_heavyatom_names, label_to_heavyatom
import warnings
import torch

def create_structure_from_crds(aa,crds,res_id,atom_type,ins_code,chain_id=None,outPath="test.pdb", save_traj=False):
    warnings.filterwarnings("ignore", ".*Used element.*")
    structure_builder = StructureBuilder.StructureBuilder()
    structure_builder.init_structure(0)
    if len(crds.shape) == 2:
        crds = crds[None]
    for model_idx in range(crds.shape[0]):
        structure_builder.init_model(model_idx)
        if chain_id is not None:
            unique_chains = np.unique(chain_id)
            if len(unique_chains) == 1 and unique_chains[0] == '':
                unique_chains = ['A']
                chain_id = np.array(['A'] * len(aa))
            mask_long_chains = np.array([len(i) > 1 for i in unique_chains])
            unique_chains = unique_chains[mask_long_chains == False]
        else:
            chain_id = np.array(['A'] * len(aa))
            unique_chains = ['A']

        for i in unique_chains:
            structure_builder.init_chain(i)
            structure_builder.init_seg(' ')

            aa_chain = np.array(list(aa))[chain_id == i]
            crds_chain = crds[model_idx][chain_id == i]
            res_id_chain = res_id[chain_id == i]
            ins_code_chain = ins_code[chain_id==i]

            res_id_unique = np.unique(res_id_chain)
            atom_type_chain = atom_type[chain_id == i]
            for idx, r in enumerate(res_id_unique):
                ins_code_resid = ins_code_chain[res_id_chain == r]
                for icode in np.unique(ins_code_resid):
                    res_mask = (res_id_chain == r) * (ins_code_chain == icode)
                    res = aa_chain[res_mask]
                    assert len(np.unique(res)) == 1
                    res = res[0]
                    crds_res = crds_chain[res_mask]
                    atom_type_res = atom_type_chain[res_mask]
                    hetero = "H" if res not in one_to_three_letter.values() else " "
                    icode_in = " " if icode == "" else icode
                    structure_builder.init_residue(res, hetero, r, icode_in)
                    for i, atom_name in enumerate(atom_type_res):
                        if atom_name == '': continue
                        if len(atom_name) == 1:
                            fullname = f' {atom_name}  '
                        elif len(atom_name) == 2:
                            fullname = f' {atom_name} '
                        elif len(atom_name) == 3:
                            fullname = f' {atom_name}'
                        else:
                            fullname = atom_name  # len == 4
                        structure_builder.init_atom(name=atom_name, coord=crds_res[i], b_factor=100.0, occupancy=1.0,
                                                    altloc=" ", fullname=fullname)

    st = structure_builder.get_structure()
    io = PDBIO()
    io.set_structure(st)
    io.save(outPath)

def create_chemical_structure(crds,atom_type,outPath="test.pdb", b_factor=None):
    warnings.filterwarnings("ignore", ".*Used element.*")
    structure_builder = StructureBuilder.StructureBuilder()
    structure_builder.init_structure(0)
    if len(crds.shape) == 2:
        crds = crds[None]
    for model_idx in range(crds.shape[0]):
        structure_builder.init_model(model_idx)

        structure_builder.init_chain('A')
        structure_builder.init_seg(' ')
        # structure_builder.init_residue('UNK', "H", 1, " ")
        crds_one = crds[model_idx]
        if b_factor is not None:
            b_factor_one = b_factor[model_idx]

        for idx, atom_name in enumerate(atom_type):
            structure_builder.init_residue('UNK', "H", idx+1, " ")
            if atom_name == '': continue
            atom_name = f'{atom_name}'
            if len(atom_name) == 1:
                fullname = f' {atom_name}  '
            elif len(atom_name) == 2:
                fullname = f' {atom_name} '
            elif len(atom_name) == 3:
                fullname = f' {atom_name}'
            else:
                fullname = atom_name  # len == 4

            if b_factor is not None:
                b_factor_atom = b_factor_one[idx]
            else:
                b_factor_atom = 100.0

            structure_builder.init_atom(name=atom_name, coord=crds_one[idx], b_factor=b_factor_atom, occupancy=1.0,
                                        altloc=" ", fullname=fullname)

    st = structure_builder.get_structure()
    io = PDBIO()
    io.set_structure(st)
    io.save(outPath)

from Bio import PDB
import numpy as np

# Atomic radii for various atom types.
# You can comment out the ones you don't care about or add new ones
atom_radii = {
#    "H": 1.20,  # Who cares about hydrogen??
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    # "F": 1.47,
    # "P": 1.80,
    # "CL": 1.75,
    # "MG": 1.73,
}

def count_clashes(path, clash_cutoff=0.4):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(0, path)

    # Set what we count as a clash for each pair of atoms
    clash_cutoffs = {i + "_" + j: (clash_cutoff * (atom_radii[i] + atom_radii[j])) for i in atom_radii for j in atom_radii}
    # Extract atoms for which we have a radii
    atoms = [x for x in structure.get_atoms() if x.element in atom_radii]
    coords = np.array([a.coord for a in atoms], dtype="d")
    # Build a KDTree (speedy!!!)
    kdt = PDB.kdtrees.KDTree(coords)
    # Initialize a list to hold clashes
    clashes = []
    # Iterate through all atoms
    for atom_1 in atoms:
        # Find atoms that could be clashing
        kdt_search = kdt.search(np.array(atom_1.coord, dtype="d"), max(clash_cutoffs.values()))
        # Get index and distance of potential clashes
        potential_clash = [(a.index, a.radius) for a in kdt_search]
        for ix, atom_distance in potential_clash:
            atom_2 = atoms[ix]
            # Exclude clashes from atoms in the same residue
            if atom_1.parent.id == atom_2.parent.id:
                continue
            # Exclude clashes from peptide bonds
            elif (atom_2.name == "C" and atom_1.name == "N") or (atom_2.name == "N" and atom_1.name == "C"):
                continue
            # Exclude clashes from disulphide bridges
            elif (atom_2.name == "SG" and atom_1.name == "SG") and atom_distance > 1.88:
                continue
            if atom_distance < clash_cutoffs[atom_2.element + "_" + atom_1.element]:
                clashes.append((atom_1, atom_2))
    return len(clashes) // 2

# from https://hunterheidenreich.com/posts/kabsch_algorithm/
def kabsch(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.
    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = torch.mean(P, dim=0)
    centroid_Q = torch.mean(Q, dim=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(0, 1), q)

    # SVD
    U, S, Vt = torch.linalg.svd(H)

    # Validate right-handed coordinate system
    if torch.det(torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))) < 0.0:
        Vt[:, -1] *= -1.0

    # Optimal rotation
    R = torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))

    # RMSD
    P_optim = torch.matmul(p, R.transpose(0, 1))
    rmsd = torch.sqrt(torch.sum(torch.square(P_optim - q)) / P.shape[0])

    return P_optim, rmsd