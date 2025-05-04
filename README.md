# DirectMultiStep: Direct Route Generation for Multi-Step Retrosynthesis

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/batistagroup/DirectMultiStep/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2405.13983-b31b1b.svg)](https://arxiv.org/abs/2405.13983)
[![image](https://img.shields.io/pypi/v/DirectMultiStep.svg)](https://pypi.org/project/DirectMultiStep/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/directmultistep)](https://pypi.org/project/DirectMultiStep/)

## Overview

This work has been published in [*J. Chem. Inf. Model*](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01982). The preprint for this work was posted on [arXiv](https://arxiv.org/abs/2405.13983).

You can use DMS models without installation through our web interface at [models.batistalab.com](https://models.batistalab.com). Or, if you want, you can install the package from pypi `pip install directmultistep`. Check out [dms.batistalab.com](https://dms.batistalab.com) for full documentation.

## How to use

Here's a quick example to generate a retrosynthesis route (you can get relevant checkpoints by running `bash download_files.sh`).

```python
from directmultistep.generate import generate_routes
from pathlib import Path

data_path = Path(__file__).resolve().parents[1] / "data"
ckpt_path = data_path / "checkpoints"
fig_path = data_path / "figures"
config_path = data_path / "configs" / "dms_dictionary.yaml"

# Generate a route for a target molecule
target = "CNCc1cc(-c2ccccc2F)n(S(=O)(=O)c2cccnc2)c1"
starting_material = "CN"

# Find routes with different models:
# Using flash model with starting material
paths = generate_routes(
    target, 
    n_steps=2, 
    starting_material=starting_material, 
    model="flash", beam_size=5,
    config_path=config_path, ckpt_dir=ckpt_path
)

# Or use explorer model to automatically determine steps
paths = generate_routes(
    target,
    starting_material=starting_material,
    model="explorer",
    beam_size=5,
    config_path=config_path, ckpt_dir=ckpt_path
)
```

See `use-examples/generate-route.py` to see more examples with other models. Other example scripts include:

- `train-model.py`: Train a new model with customizable configuration for local or cluster environments
- `eval-subset.py`: Evaluate a trained model on a subset of data
- `paper-figures.py`: Reproduce figures from the paper
- `visualize-train-curves.py`: Plot training curves and metrics

## Citing

If you use DirectMultiStep in an academic project, please consider citing our publication in [*J. Chem. Inf. Model*](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01982):

```tex
@article{directmultistep,
    author = {Shee, Yu and Morgunov, Anton and Li, Haote and Batista, Victor S.},
    title = {DirectMultiStep: Direct Route Generation for Multistep Retrosynthesis},
    journal = {Journal of Chemical Information and Modeling},
    volume = {65},
    number = {8},
    pages = {3903-3914},
    year = {2025},
    doi = {10.1021/acs.jcim.4c01982},
    note ={PMID: 40197023},
    URL = {https://doi.org/10.1021/acs.jcim.4c01982},
    eprint = {https://doi.org/10.1021/acs.jcim.4c01982}
}
```

## Extra Materials

Through [download_files.sh](./download_files.sh) you can download canonicalized versions of eMols (23M SMILES), Buyables (329k SMILES), ChEMBL-5000 (5k SMILES), and USPTO-190 (190 SMILES). Using pre-canonicalized version saves you roughly a day of cpu time. If you happen to use these canonicalized versions, consider citing the repo from figshare:

```tex
@misc{shee2025figshare,
    author = {Yu Shee and Anton Morgunov},
    title = {Data for ``DirectMultiStep: Direct Route Generation for Multistep Retrosynthesis''},
    year = {2025},
    month = {3},
    howpublished = {\url{https://figshare.com/articles/dataset/Data_for_DirectMultiStep_Direct_Route_Generation_for_Multistep_Retrosynthesis_/28629470}},
    doi = {"10.6084/m9.figshare.28629470.v1"},
    note = {Accessed: 20xx-xx-xx}
}
```

Also check out the [HigherLev Retro](https://github.com/jihye-roh/higherlev_retro) repo which is the source of the Buyables stock set. [route-distances](https://github.com/MolecularAI/route-distances?tab=readme-ov-file) is the source of ChEMBL-5000. [Retro*](https://github.com/binghong-ml/retro_star) is the source of the eMols stock set and USPTO-190.

## Licenses

All code is licensed under MIT License. The content of the [pre-print on arXiv](https://arxiv.org/abs/2405.13983) is licensed under CC-BY 4.0.
