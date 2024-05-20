# DirectMultiStep: Direct Route Generation for Multi-Step Retrosynthesis

[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Code coverage with tests:

```bash
Name                  Stmts   Miss  Cover
-----------------------------------------
Data/Dataset.py         157    114    27%
Utils/PreProcess.py      60      8    87%
-----------------------------------------
TOTAL                   217    122    44%
```

## Overview

- [Data/Dataset.py](/Data/Dataset.py) definition of custom torch Datasets used for training and evaluation.
- [Data/process.py](/Data/process.py) scripts to preprocess PaRoutes dataset and create training and evaluation partitions.
- [Models/Architecture.py](/Models/Architecture.py) contains definitions of Encoder, Decoder, and combining Seq2Seq module.
- [Models/Training.py](/Models/Training.py) definition of Lightning Training class
- [Models/Configure.py](/Models/Configure.py) definiton of model config
- [Models/Generation.py](/Models/Generation.py) implementation of beam search using python lists
- [Models/TensorGen.py](/Models/TensorGen.py) implementation of beam search using torch.Tensors to maximize GPU efficiency. Warning: the current algorithm works properly only with batch_size=1 inputs (PRs welcome).

For training see:

- [train_nosm.py](/train_nosm.py) - w/o SM provided to encoder
- [train_wsm.py](/train_wsm.py) - w/ SM provided to encoder

Once everything is set up, it's suffice to simply run `python train_wsm.py`.

## Licenses

All code is licensed under MIT License.
