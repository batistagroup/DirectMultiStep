# DirectMultiStep: Direct Route Generation for Multi-Step Retrosynthesis

[![arXiv](https://img.shields.io/badge/arXiv-2405.13983-b31b1b.svg)](https://arxiv.org/abs/2405.13983)

DirectMultiStep is a novel multi-step first approach for generating retrosynthesis routes in chemistry. The project provides multiple models for different retrosynthesis generation approaches.

## Quick Start

### Installation

You can install the package directly from PyPI:

```bash
pip install directmultistep
```

### Development

We welcome any contributions, feel free to clone the repo and create a PR. We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Online Demo

Try out our deployed models without any installation at [models.batistalab.com](https://models.batistalab.com).

### Usage Example

Here's a quick example to generate a retrosynthesis route:

```python
from directmultistep.generate import generate_routes
from pathlib import Path

# Generate a route for a target molecule
target = "CNCc1cc(-c2ccccc2F)n(S(=O)(=O)c2cccnc2)c1"
starting_material = "CN"

# Using flash model with starting material
paths = generate_routes(
    target, 
    n_steps=2,
    starting_material=starting_material, 
    model="flash", 
    beam_size=5,
    config_path="path/to/config.yaml", 
    ckpt_dir="path/to/checkpoints"
)

# Or use explorer model to automatically determine steps
paths = generate_routes(
    target,
    starting_material=starting_material,
    model="explorer",
    beam_size=5,
    config_path="path/to/config.yaml", 
    ckpt_dir="path/to/checkpoints"
)
```

## License

- Code: MIT License
- Paper content ([arXiv preprint](https://arxiv.org/abs/2405.13983)): CC-BY 4.0
