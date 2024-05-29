# DirectMultiStep: Direct Route Generation for Multi-Step Retrosynthesis

[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/batistagroup/DirectMultiStep/graph/badge.svg?token=2G1x86tsjc)](https://codecov.io/gh/batistagroup/DirectMultiStep)

## Overview

The preprint for this work is posted on [arXiv](https://arxiv.org/abs/2405.13983).

- [Data/process.py](/Data/process.py) scripts to preprocess PaRoutes dataset and create training and evaluation partitions.
- [Models/Architecture.py](/DirectMultiStep/Models/Architecture.py) contains definitions of Encoder, Decoder, and combining Seq2Seq module.
- [Models/Training.py](/DirectMultiStep/Models/Training.py) definition of Lightning Training class
- [Models/Configure.py](/DirectMultiStep/Models/Configure.py) definiton of model config
- [Models/Generation.py](/DirectMultiStep/Models/Generation.py) implementation of beam search using python lists
- [Models/TensorGen.py](/DirectMultiStep/Models/TensorGen.py) implementation of beam search using torch.Tensors to maximize GPU efficiency. Warning: the current algorithm works properly only with batch_size=1 inputs (PRs welcome).
- [Utils/Dataset.py](/DirectMultiStep/Utils/Dataset.py) definition of custom torch Datasets used for training and evaluation.
- [Utils/PreProcess.py](/DirectMultiStep/Utils/PreProcess.py) all functions related to preprocessing of the PaRoutes dataset (used by [Data/process.py](/Data/process.py))
- [Utils/PostProcess.py](/DirectMultiStep/Utils/PostProcess.py) all functions needed to postprocess results of beam search and run evaluations
- [Utils/Visualize.py](/DirectMultiStep/Utils/Visualize.py) function that draws the synthesis tree as a pdf

For training see:

- [train_nosm.py](/DirectMultiStep/train_nosm.py) - w/o SM provided to encoder
- [train_wsm.py](/DirectMultiStep/train_wsm.py) - w/ SM provided to encoder

Once everything is set up, it's suffice to simply run `python train_wsm.py`.

Run `bash download_ckpts.sh` to download our checkpoints from the file storage.

Finally, we provide [assess_single.py](/assess_single.py) which allows to run our model on a single target compound.

## Tutorials

To use the tutorials, simply move/copy them to the root directory. This is necessary because the notebooks use relative imports.

- [Tutorials/Basic_Usage.ipynb](/Tutorials/Basic_Usage.ipynb) walks you through how to input your compounds, steps, and starting materials. Visualization of routes in PDF is shown.
- [Tutorials/Route_Separation.ipynb](/Tutorials/Route_Separation.ipynb) reproduces the route separation results from the paper.
- [Tutorials/Pharma_Compounds.ipynb](/Tutorials/Pharma_Compounds.ipynb) reproduces the three FDA-approved drug results from the paper.

## Licenses

All code is licensed under MIT License. The content of the [pre-print on arXiv](https://arxiv.org/abs/2405.13983) is licensed under CC-BY 4.0.

## TODO

- Bring codecov to 80+.
- Revise [Models/TensorGen.py](/DirectMultiStep/Models/TensorGen.py) so that it can work with batch size greater than 1.
