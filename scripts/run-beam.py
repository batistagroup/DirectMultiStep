from pathlib import Path

import numpy as np
import pytest
import torch

from directmultistep.generation.tensor_gen import BatchedBeamSearch, BeamSearchOptimized
from directmultistep.utils.dataset import RoutesProcessing
from directmultistep.generate import load_published_model
from directmultistep.generate import prepare_input_tensors

torch.manual_seed(42)
np.random.seed(42)

config_path = Path("data/configs/dms_dictionary.yaml")
ckpt_dir = Path("data/checkpoints")

if not config_path.exists() or not ckpt_dir.exists():
    pytest.skip("Model files not found. Ensure data is downloaded.")



model = load_published_model("flash", ckpt_dir)
rds = RoutesProcessing(metadata_path=config_path)

device = next(model.parameters()).device
beam_obj = BatchedBeamSearch(
    model=model,
    beam_size=5,
    start_idx=0,
    pad_idx=52,
    end_idx=22,
    max_length=1074,
    idx_to_token=rds.idx_to_token,
    device=device,
)

target = "CNCc1cc(-c2ccccc2F)n(S(=O)(=O)c2cccnc2)c1"
starting_material = "CN"
n_steps = 2
