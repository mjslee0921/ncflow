import copy
import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch

from utils.loader import load_seed, load_device, load_ema, load_checkpoint
from utils.train_utils import count_parameters, recursive_to
from pathlib import Path
from dataset import get_dataloader
from model.af3 import AF3Transformer
from model.cnf import CNF
from utils.structure_utils import create_structure_from_crds, create_chemical_structure
import yaml
from easydict import EasyDict as edict
import biotite.structure as bstruct
import biotite.structure.io as bsio
from scipy.spatial import KDTree
from utils.constants import three_to_one_letter, letter_to_num, heavyatom_to_label, atom_to_number, canonical_smiles
from biotite.structure import connect_via_residue_names, filter_peptide_backbone
import hydride
from scipy.spatial.distance import cdist
from rdkit import Chem
from utils.rdkit_utils import reorder_mol_by_another

ncaa_dict = torch.load('./data/ncaa/wuxi.pth')

def read_pdb(fileobj, index=-1):
    """Read PDB files.

    The format is assumed to follow the description given in
    http://www.wwpdb.org/documentation/format32/sect9.html."""
    if isinstance(fileobj, str):
        fileobj = open(fileobj)

    images = []
    atoms = Atoms()
    for line in fileobj.readlines():
        if line.startswith('ATOM') or line.startswith('HETATM'):
            try:
                # Atom name is arbitrary and does not necessarily contain the element symbol.
                # The specification requires the element symbol to be in columns 77+78.
                symbol = line[76:78].strip().lower().capitalize()
                words = line[30:55].split()
                position = np.array([float(words[0]),
                                     float(words[1]),
                                     float(words[2])])
                atoms.append(Atom(symbol, position))
            except:
                pass
        if line.startswith('ENDMDL'):
            images.append(atoms)
            atoms = Atoms()
    if len(images) == 0:
        images.append(atoms)
    return images[index]


def write_pdb(fileobj, images):
    """Write images to PDB-file.

    The format is assumed to follow the description given in
    http://www.wwpdb.org/documentation/format32/sect9.html."""
    if isinstance(fileobj, str):
        fileobj = paropen(fileobj, 'w')

    if not isinstance(images, (list, tuple)):
        images = [images]

    # if images[0].get_pbc().any():
    #     from ase.lattice.spacegroup.cell import cell_to_cellpar
    #     cellpar = cell_to_cellpar(images[0].get_cell())
    #     # ignoring Z-value, using P1 since we have all atoms defined explicitly
    #     format = 'CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1\n'
    #     fileobj.write(format % (cellpar[0], cellpar[1], cellpar[2], cellpar[3], cellpar[4], cellpar[5]))

    #         1234567 123 6789012345678901   89   67   456789012345678901234567 890
    format = 'ATOM  %5d %4s MOL     1    %8.3f%8.3f%8.3f  1.00  0.00          %2s  \n'

    # RasMol complains if the atom index exceeds 100000. There might
    # be a limit of 5 digit numbers in this field.
    MAXNUM = 100000

    symbols = images[0].get_chemical_symbols()
    natoms = len(symbols)

    for n, atoms in enumerate(images):
        fileobj.write('MODEL     ' + str(n + 1) + '\n')
        p = atoms.get_positions()
        for a in range(natoms):
            x, y, z = p[a]
            fileobj.write(format % (a % MAXNUM, symbols[a], x, y, z, symbols[a].rjust(2)))
        fileobj.write('ENDMDL\n')

def get_config(config, seed):
    config_dir = f'./config/inference/{config}.yaml'
    config = edict(yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader))
    config.seed = seed

    return config

def add_hydrogens(structure):
    # remove hydrogens if they exist
    h_mask = [True if i != 'H' else False for i in structure.element]
    structure = structure[h_mask]

    if not hasattr(structure,'charge'):
        structure.charge = np.zeros(len(structure))
    if structure.bonds is None:
        structure.bonds = bstruct.connect_via_residue_names(structure)

    structure, _ = hydride.add_hydrogen(structure)
    # structure.coord = hydride.relax_hydrogen(structure)

    return structure

def load_structure(args):
    structure = bsio.load_structure(args.pdb_path, model=1)
    # remove all hydrogens
    h_mask = [True if i != 'H' else False for i in structure.element]
    structure = structure[h_mask]
    # structure = structure[bstruct.filter_amino_acids(structure)]
    avail_chains = bstruct.get_chains(structure)
    assert args.peptide_chain in avail_chains, 'peptide chain not found'

    chains = args.receptor_chain.split(',')
    assert all([i in avail_chains for i in chains]), 'receptor chain not found'
    receptor = []
    for c in chains:
        receptor.append(structure[structure.chain_id == c])
    receptor = bstruct.concatenate(receptor)

    peptide = structure[structure.chain_id == args.peptide_chain]
    peptide.bonds = bstruct.connect_via_residue_names(peptide)
    peptide.bonds.remove_aromaticity()

    if args.cyclic:
        bb_mask = np.isin(peptide.atom_name, ['N', 'CA', 'C', 'O'])
        bb_index = bb_mask.nonzero()[0]
        peptide.bonds.add_bond(bb_index[0], bb_index[-2], 1)

    # if args.custom:
    #     bb_mask = np.isin(peptide.atom_name, ['N', 'CA', 'C', 'O'])
    #     bb_index = bb_mask.nonzero()[0]
    #     n_index = (peptide.atom_name == 'N01').nonzero()[0][0]
    #     c_index = (peptide.atom_name == 'C09').nonzero()[0][0]
    #     peptide.bonds.add_bond(n_index, bb_index[-2], 1)
    #     peptide.bonds.add_bond(bb_index[0], c_index, 1)

    # we're just going to autodetect disulfide bridges < 4A
    if args.disulfide:
        cys_mask = peptide.res_name == 'CYS'
        s_mask = peptide.atom_name == 'SG'
        dist = cdist(peptide.coord, peptide.coord) < 4.0
        disulfide_mask = dist * (cys_mask[None] * cys_mask[..., None]) * (s_mask[None] * s_mask[..., None]) * (1-np.eye(len(dist)))
        indices = disulfide_mask.nonzero()
        for i,j in zip(indices[0],indices[1]):
            peptide.bonds.add_bond(i, j, 1)

    data = {
        'id': f'{Path(args.pdb_path).stem}',
        'peptide': peptide,
        'receptor': receptor
    }

    return data

def extract_features(peptide, receptor, idx, ncaa, bond_matrix, d_aa=False):
    peptide_chain = peptide.chain_id[0]
    complex = bstruct.concatenate([receptor, peptide])
    res_idx_to_modify = (complex.chain_id == peptide_chain) * (complex.res_id == idx)
    res_to_modify = complex[res_idx_to_modify]
    ca_to_modify = res_idx_to_modify * (complex.atom_name == 'CA')
    ca_coord = complex[ca_to_modify].coord
    parent_res = complex[ca_to_modify].res_name[0]
    assert len(ca_coord) == 1

    complex_tree = KDTree(complex.coord)
    # find sc near native sidechain
    sc_coords = res_to_modify[~bstruct.filter_peptide_backbone(res_to_modify)]
    indices = complex_tree.query_ball_point(sc_coords.coord, r=3.5)
    unique_res_indices = []
    for i in indices:
        unique_res_indices += list(i)
    unique_res_indices = np.unique(unique_res_indices)
    res_pos = bstruct.get_residue_positions(complex, unique_res_indices)
    res_starts = bstruct.get_residue_starts(complex, add_exclusive_stop=True)
    full_indices = []
    for i in res_pos:
        full_indices.append(
            np.arange(res_starts[i], res_starts[i + 1])
        )
    full_indices = np.unique(np.concatenate(full_indices))
    loss_mask = torch.zeros(len(complex))
    loss_mask[full_indices] = 1


    # # find the indices to slice complex
    # res_nonzero = res_idx_to_modify.nonzero()[0]
    # start_idx, end_idx = res_nonzero[0], res_nonzero[-1]+1
    # complex_m1 = complex[~res_idx_to_modify]
    # tree = KDTree(complex_m1.coord)

    # k = min(200, len(complex_m1.coord))
    # dists, indices = tree.query(ca_coord, k=k)
    # indices = np.sort(indices)[0]
    # selected = complex_m1[indices]
    # num_res_before = sum(indices<start_idx)

    res_nonzero = res_idx_to_modify.nonzero()[0]
    start_idx, end_idx = res_nonzero[0], res_nonzero[-1]+1
    k = min(200, len(complex.coord))
    dists, indices = complex_tree.query(ca_coord, k=k)

    indices = np.unique(np.concatenate([indices[0], full_indices]))
    selected = complex[indices]
    num_res_before = sum(indices<start_idx)
    loss_mask = loss_mask[indices].to(dtype=torch.bool)

    # Load ncAA structure and modify to fit structure
    ncaa_structure = ncaa
    ncaa_structure.chain_id = [peptide_chain] * len(ncaa_structure)
    ncaa_structure.res_id = [idx] * len(ncaa_structure)
    ncaa_structure.coord = np.zeros_like(ncaa_structure.coord)
    ncaa_structure.coord[:4] = res_to_modify.coord[:4] # reset backbone coords

    final_structure = bstruct.concatenate([selected[:num_res_before],ncaa_structure,selected[num_res_before+len(res_to_modify):]])
    loss_mask = torch.cat([loss_mask[:num_res_before],torch.ones(len(ncaa_structure)),loss_mask[num_res_before+len(res_to_modify):]])

    ncaa_mask = np.zeros(len(final_structure))
    ncaa_mask[num_res_before:num_res_before+len(ncaa_structure)] = 1

    coords = torch.tensor(final_structure.coord)
    # aa = torch.LongTensor([self.code_to_idx.get(i,self.code_to_idx['UNK']) for i in data.res_name])
    atom_type = torch.LongTensor([heavyatom_to_label.get(i, heavyatom_to_label['X']) for i in final_structure.element])
    complex_mask = 1 - ncaa_mask

    coords = coords - ca_coord # remove CoM

    bond_type = connect_via_residue_names(final_structure, inter_residue=True).bond_type_matrix()
    bond_type[bond_type == 0] = 1  # any type to single
    bond_type[bond_type == -1] = 0  # -1 is empty, set to 0
    bond_type[bond_type == 4] = 3  # just set quadruple bonds to triple (these rarely exist)
    bond_type[bond_type >= 5] = 4  # set aromatic to 4
    bond_type = torch.LongTensor(bond_type)
    ncaa_pair_mask = ncaa_mask[...,None] * ncaa_mask[None]
    bond_type[ncaa_pair_mask==1] = bond_matrix.long().flatten()

    # add peptide bond
    c_before = (final_structure.chain_id == peptide_chain) * (final_structure.res_id == idx-1) * \
                (final_structure.atom_name == 'C') * bstruct.filter_peptide_backbone(final_structure)
    c_before = c_before.nonzero()[0]

    n_after = (final_structure.chain_id == peptide_chain) * (final_structure.res_id == idx+1) * \
                (final_structure.atom_name == 'N') * bstruct.filter_peptide_backbone(final_structure)
    n_after = n_after.nonzero()[0]

    ncaa_indices = ncaa_mask.nonzero()[0]
    ncaa_c = ncaa_indices[2]
    ncaa_n = ncaa_indices[0]
    if len(c_before) == 1:
        bond_type[ncaa_n,c_before[0]] = 1
        bond_type[c_before[0], ncaa_n] = 1
    if len(n_after) == 1:
        bond_type[ncaa_c, n_after[0]] = 1
        bond_type[n_after[0], ncaa_c] = 1

    mask = torch.ones(len(final_structure))
    if d_aa:
        backbone_mask = torch.LongTensor(np.isin(final_structure.atom_name, ['N', 'C', 'O']))
    else:
        backbone_mask = torch.LongTensor(np.isin(final_structure.atom_name, ['N','CA','C','O']))

    out = {
        'pos': coords,
        'atom_type_str': final_structure.atom_name.tolist(),
        'atom_type': atom_type,
        'mask': mask,
        'complex_mask': torch.LongTensor(complex_mask),
        'ncaa_mask': torch.LongTensor(ncaa_mask),
        'adj': bond_type,
        'backbone_mask': backbone_mask,
        'loss_mask': loss_mask
    }

    # add batch dimension
    out = {k:v[None] if type(v) == torch.Tensor else v for k,v in out.items()}

    return out, ncaa_structure, ca_coord, parent_res

def count_clashes(structure):
    # very rough estimate given vdw radius of 1.5
    bond_matrix = connect_via_residue_names(structure).bond_type_matrix()
    coords = torch.Tensor(structure.coord)
    bond_matrix = torch.Tensor(bond_matrix)
    dist = torch.cdist(coords,coords)
    clashes = (dist < 1.5) * (bond_matrix == -1)
    return clashes.sum().item()

def sample(args):
    config = get_config(args.config, args.seed)

    ckpt_dict = torch.load(config.ckpt)
    training_cfg = ckpt_dict['config']

    # -------- Load models, optimizers, ema --------
    model = AF3Transformer(**training_cfg.model).cuda()
    ema = load_ema(model, decay=training_cfg.train.ema)
    model, ema = load_checkpoint(model, ema, ckpt_dict)
    model = CNF(model, config).eval()
    ema.copy_to(model.parameters())

    if config.conf_ckpt is not None:
        conf_ckpt = torch.load(config.conf_ckpt)
        conf_state_dict, conf_config = conf_ckpt['state_dict'], conf_ckpt['config']
        if 'module' in list(conf_state_dict.keys())[0]:  # if trained with DDP
            conf_state_dict = {k[7:]: v for k, v in conf_state_dict.items()}
        conf_model = AF3Transformer(**conf_config.model).cuda().eval()
        conf_model.load_state_dict(conf_state_dict)
    else:
        conf_model = None

    save_path = Path(f'./designs/{args.name}')
    save_path.mkdir(exist_ok=True, parents=True)

    data = load_structure(args)
    pdb = Path(args.pdb_path).stem

    out_dict = {}
    with torch.no_grad():
        peptide_gt, receptor_gt = data['peptide'], data['receptor']
        complex_gt = bstruct.concatenate([receptor_gt,peptide_gt])
        clashes_gt = count_clashes(complex_gt)
        bsio.save_structure(str(save_path.joinpath('gt.pdb')), complex_gt)
        bsio.save_structure(str(save_path.joinpath('peptide.pdb')), peptide_gt)
        bsio.save_structure(str(save_path.joinpath('receptor.pdb')), receptor_gt)

        peptide_gt_h = add_hydrogens(peptide_gt)
        bsio.save_structure(str(save_path.joinpath('peptide.sdf')), peptide_gt_h)
        bsio.save_structure(str(save_path.joinpath('peptide_addh.pdb')), peptide_gt_h)

        complex_gt_h = add_hydrogens(complex_gt)
        bsio.save_structure(str(save_path.joinpath('gt_addh.pdb')), complex_gt_h)

        receptor_gt_h = add_hydrogens(receptor_gt)
        bsio.save_structure(str(save_path.joinpath('receptor_addh.pdb')), receptor_gt_h)

        # extract interface residues
        # peptide_tree = KDTree(peptide_gt.coord)
        # interface = peptide_tree.query_ball_point(receptor_gt.coord, r=12.0)
        # all = []
        # for i in interface:
        #     all += i
        # unique_indices = np.unique(all)
        # res_id = np.unique(peptide_gt[unique_indices].res_id)
        res_id = np.unique(peptide_gt.res_id)

        for res_idx in tqdm(res_id):
            parent_res = peptide_gt[peptide_gt.res_id == res_idx].res_name[0]
            try:
                ncaa_pool = ncaa_dict[parent_res]
            except KeyError:
                continue
            for ncaa_data in tqdm(ncaa_pool):
                template, bond_matrix = ncaa_data['biotite'], ncaa_data['bond_matrix']
                # d_aa = ncaa_data.get('d_aa',False) # assume L-aa if doesn't exist
                batch, ncaa_template, ca_coord, parent_res = extract_features(peptide_gt, receptor_gt, res_idx, template, bond_matrix, d_aa=False)
                batch = recursive_to(batch,"cuda")
                coords, mask = batch['pos'], batch['mask']
                sample_id = f'{pdb}_receptor{args.receptor_chain}_peptide{args.peptide_chain}_residx{res_idx}'
                atom_type_str = [i[0] for i in batch['atom_type_str']]
                for sample_idx in range(config.sample.n_samples):
                    peptide, receptor = copy.deepcopy(peptide_gt), copy.deepcopy(receptor_gt)

                    outPath = save_path.joinpath(sample_id, f'{parent_res}_{ncaa_data["id"]}', f'run_{sample_idx+1}')
                    outPath.mkdir(exist_ok=True,parents=True)
                    z = torch.randn_like(coords)

                    sample_traj, conf_traj = model.decode_euler(z, batch, conf_model=conf_model)
                    sample = sample_traj[-1]
                    plddt = conf_traj[-1]

                    # extract ncaa
                    ncaa_mask = batch['ncaa_mask']
                    ncaa_coords = sample[ncaa_mask==1].detach().cpu().numpy()
                    ncaa_template.coord = ncaa_coords

                    peptide_res_mask = peptide.res_id == res_idx
                    peptide_res_mask_indices = peptide_res_mask.nonzero()[0]
                    peptide_start, peptide_end = peptide_res_mask_indices[0], peptide_res_mask_indices[-1]

                    # remove CoM on Ca of residue
                    peptide.coord = peptide.coord - ca_coord
                    receptor.coord = receptor.coord - ca_coord
                    peptide_final = bstruct.concatenate([peptide[:peptide_start], ncaa_template, peptide[peptide_end+1:]])
                    complex_final = bstruct.concatenate([receptor, peptide_final])
                    peptide_final.coord = peptide_final.coord + ca_coord
                    complex_final.coord = complex_final.coord + ca_coord

                    clashes = count_clashes(complex_final)
                    complex_final.add_annotation('b_factor',dtype=float)
                    b_factor = np.zeros(len(complex_final)) + 100
                    b_start = len(receptor) + peptide_start
                    if conf_model is not None:
                        b_factor[b_start:b_start + len(ncaa_template)] = plddt[ncaa_mask==1].detach().cpu().numpy()
                        b_factor_sample = plddt
                        avg_plddt = plddt[ncaa_mask==1].mean().item()
                    else:
                        b_factor[b_start:b_start+len(ncaa_template)] = 0.
                        b_factor_sample = ncaa_mask * 100
                        avg_plddt = 100.

                    complex_minus_ncaa = bstruct.concatenate([receptor, peptide[:peptide_start], peptide[peptide_end+1:]])

                    ncaa_bonds = peptide_final.bonds.as_array()
                    canonical_bonds = bstruct.connect_via_residue_names(peptide_final).as_array()
                    all_bonds = np.concatenate([ncaa_bonds,canonical_bonds])
                    all_bonds = bstruct.BondList(len(peptide_final),all_bonds)

                    # add peptide bond
                    c_before = (peptide_final.res_id == res_idx - 1) * \
                               (peptide_final.atom_name == 'C') * bstruct.filter_peptide_backbone(peptide_final)
                    c_before = c_before.nonzero()[0]

                    n_after = (peptide_final.res_id == res_idx + 1) * \
                              (peptide_final.atom_name == 'N') * bstruct.filter_peptide_backbone(peptide_final)
                    n_after = n_after.nonzero()[0]

                    ncaa_indices = (peptide_final.res_id == res_idx).nonzero()[0]
                    ncaa_c = ncaa_indices[2]
                    ncaa_n = ncaa_indices[0]
                    if len(c_before) == 1:
                        all_bonds.add_bond(ncaa_n, c_before[0], 1)
                    if len(n_after) == 1:
                        all_bonds.add_bond(ncaa_c, n_after[0], 1)

                    # add cyclic offset if exists
                    if args.cyclic:
                        bb_mask = np.isin(peptide_final.atom_name, ['N','CA','C','O'])
                        bb_index = bb_mask.nonzero()[0]
                        all_bonds.add_bond(bb_index[0],bb_index[-2],1)

                    # if args.custom:
                    #     bb_mask = np.isin(peptide_final.atom_name, ['N', 'CA', 'C', 'O'])
                    #     bb_index = bb_mask.nonzero()[0]
                    #     n_index = (peptide_final.atom_name == 'N01').nonzero()[0][0]
                    #     c_index = (peptide_final.atom_name == 'C09').nonzero()[0][0]
                    #     all_bonds.add_bond(n_index, bb_index[-2], 1)
                    #     all_bonds.add_bond(bb_index[0], c_index, 1)

                    # we're just going to autodetect disulfide bridges < 4A
                    if args.disulfide:
                        cys_mask = peptide_final.res_name == 'CYS'
                        s_mask = peptide_final.atom_name == 'SG'
                        dist = cdist(peptide_final.coord, peptide_final.coord) < 4.0
                        disulfide_mask = dist * (cys_mask[None] * cys_mask[..., None]) * (
                                    s_mask[None] * s_mask[..., None]) * (1 - np.eye(len(dist)))
                        indices = disulfide_mask.nonzero()
                        for i, j in zip(indices[0], indices[1]):
                            all_bonds.add_bond(i, j, 1)

                    peptide_final.bonds = all_bonds

                    complex_final.b_factor = b_factor
                    bsio.save_structure(str(outPath.joinpath('complex.pdb')), complex_final)
                    bsio.save_structure(str(outPath.joinpath('peptide.pdb')), peptide_final)

                    peptide_final_h = add_hydrogens(peptide_final)
                    peptide_final_h.bonds.remove_aromaticity()
                    bsio.save_structure(str(outPath.joinpath('peptide.sdf')), peptide_final_h)

                    create_chemical_structure(sample.cpu().numpy(), atom_type_str,
                                              outPath=str(outPath.joinpath('sample.pdb')),b_factor=b_factor_sample)
                    bsio.save_structure(str(outPath.joinpath('ncaa.mol')), ncaa_template)
                    bsio.save_structure(str(outPath.joinpath('complex_no_ncaa.pdb')), complex_minus_ncaa)

                    complex_final_h = add_hydrogens(complex_final)
                    bsio.save_structure(str(outPath.joinpath('complex_addh.pdb')), complex_final_h)

                    # check chirality
                    try:
                        mol = Chem.MolFromMolFile(outPath.joinpath('ncaa.mol'))
                        gt_mol = Chem.MolFromSmiles(ncaa_data['smiles'])
                        chiral = Chem.FindMolChiralCenters(mol)
                        gt_mol = reorder_mol_by_another(gt_mol, mol)
                        gt_chiral = Chem.FindMolChiralCenters(gt_mol)
                        chirality_match = (chiral == gt_chiral)

                        out_dict[f'{sample_id}_{ncaa_data["id"]}_run{sample_idx+1}'] = {
                            'parent_aa': parent_res,
                            'ncaa_length': len(ncaa_template),
                            'peptide_length': len(peptide_final),
                            'complex_length': len(complex_final),
                            'clashes_gt': clashes_gt,
                            'clashes': clashes,
                            'plddt': avg_plddt,
                            'chirality': chirality_match
                        }
                    except: continue

    torch.save(out_dict,save_path.joinpath('energy.pth'))

if __name__ == '__main__':
    import argparse

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('pdb_path', type=str)
    parser.add_argument('receptor_chain', type=str)
    parser.add_argument('peptide_chain', type=str)
    parser.add_argument('name', type=str, default='test')
    parser.add_argument('--residue_index', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cyclic',action='store_true')
    parser.add_argument('--disulfide', action='store_true')
    # parser.add_argument('--custom', action='store_true')

    args = parser.parse_args()
    sample(args)
