# DirectMultiStep: Direct Route Generation for Multi-Step Retrosynthesis

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/batistagroup/DirectMultiStep/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2405.13983-b31b1b.svg)](https://arxiv.org/abs/2405.13983)
[![image](https://img.shields.io/pypi/v/DirectMultiStep.svg)](https://pypi.org/project/DirectMultiStep/)

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

```
@article{directmultistep,
author = {Shee, Yu and Morgunov, Anton and Li, Haote and Batista, Victor S.},
title = {DirectMultiStep: Direct Route Generation for Multistep Retrosynthesis},
journal = {Journal of Chemical Information and Modeling},
volume = {0},
number = {0},
pages = {null},
year = {0},
doi = {10.1021/acs.jcim.4c01982},
URL = {https://doi.org/10.1021/acs.jcim.4c01982},
eprint = {https://doi.org/10.1021/acs.jcim.4c01982}
}
```

## Licenses

All code is licensed under MIT License. The content of the [pre-print on arXiv](https://arxiv.org/abs/2405.13983) is licensed under CC-BY 4.0.
