# Codebase for NCFlow

This repository contains the training and inference code for NCFlow, a flow matching model that can predict the 3D conformation
of arbitrary ncAAs in proteins.

Reference: Jin Sub Lee and Philip M Kim. “Design of peptides with non-canonical amino acids
using flow matching”. In: bioRxiv (2025), pp. 2025–07 https://www.biorxiv.org/content/10.1101/2025.07.31.667780v1

## Inference

Sampling can be done on a preprocessed test set with the following command:

`python sampler.py ncaa`

where `ncaa` refers to the YAML file in `config/inference/ncaa.yaml`.

For design via deep mutational scanning, you can run the following command:

`python design_peptides_smiles.py <config> <cif file> <comma-separated receptor chain IDs> <peptide chain> <name for labeling>`

For instance:

`python design_peptides_smiles.py ncaa ./data/mdm2.cif A,C D mdm2`

By default the ncAA pool used for sampling are taken from Wuxi Apptec, but a custom ncAA list can be generated easily
by adapting the script `process_ncaa_pool_for_design.py`.

The flags `--cyclic` should be added if the peptide is head-to-tail cyclic, and `--disulfide` if there are disulfide bridges. 
This is to ensure that these bonds are included in the output SDF files.

For Pubchem and Plinder sampling:

`python sampler_pubchem.py pubchem`

`python sampler_plinder.py plinder`

### Scoring

[AEV-PLIG](https://github.com/isakvals/AEV-PLIG) and [ATM](https://github.com/Gallicchio-Lab/alchemical-transfer-method) is used to score the designed variants and identify strongest predicted binders. Please use the respective
codebases for installation and inference.

Note that we use the scripts from [Quantumbind RBFE](https://github.com/Acellera/quantumbind_rbfe) for running ATM, which
has a more intuitive Python wrapper for running ATM.

The modified scripts for running AEV-PLIG and ATM directly on designed peptide variants will be uploaded shortly.

## Training

### Stage 1: PubChem

First, download the PubChem3D SDF files from here: https://ftp.ncbi.nlm.nih.gov/pubchem/Compound_3D/01_conf_per_cmpd/SDF/

Then, you can run the following command:

`python process_pubchem.py <path to folder with .sdf.gz> <outpath>`

This will preprocess the SDF files into .pth files containing features necessary for pretraining.

Then, you can run the command:

`python trainer_pubchem.py pubchem`

where `pubchem` refers to the `config/training/pubchem.yaml` file.

### Stage 2: Plinder

First, download the Plinder dataset (https://www.plinder.sh/) - the 2024/06 release is used for NCFlow. 
You only need the `systems` folder from Plinder to process the dataset.

Then, you can run the following command: `python process_plinder.py`

Note that the input/output paths in the processing script is hard-coded, so you will need to revise accordingly.

Then, you can train on the processed `pth` files with the command:

`python trainer_plinder.py plinder`

where `plinder` refers to the `config/training/plinder.yaml` file.

When finetuning from the PubChem checkpoint, you simply need to add the path to the checkpoint under `ckpt` in 
the YAML file and add the flag `--resume`.

### Stage 3: NCAAs

First, you must download a snapshot of the PDB using conventional tools (ex. Biotite's `rcsb` class).

Then you can run the processing script with `python process_ncaa.py`

Training is performed similarly to previous stages:

`python trainer.py ncaa`

### Contact

If there are any comments or questions, feel free to email me at mjslee0921@gmail.com.



