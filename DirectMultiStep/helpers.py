# MIT License

# Copyright (c) 2024 Batista Lab (Yale University)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
from DirectMultiStep.Utils.Dataset import RoutesStepsDataset, RoutesStepsSMDataset
from typing import Tuple, Optional
from pathlib import Path
import re


def prepare_datasets(
    train_data_path: str | Path,
    val_data_path: str | Path,
    metadata_path: str | Path,
) -> Tuple[RoutesStepsSMDataset, ...]:
    with open(train_data_path, "rb") as f:
        (products, starting_materials, path_strings, n_steps_list) = pickle.load(f)
    ds_train = RoutesStepsSMDataset(
        products=products,
        starting_materials=starting_materials,
        path_strings=path_strings,
        n_steps_list=n_steps_list,
        metadata_path=str(metadata_path),
    )
    with open(val_data_path, "rb") as f:
        (n1_products, n1_sms, n1_path_strings, n1_steps_list) = pickle.load(f)
    ds_val = RoutesStepsSMDataset(
        products=n1_products,
        starting_materials=n1_sms,
        path_strings=n1_path_strings,
        n_steps_list=n1_steps_list,
        metadata_path=str(metadata_path),
    )
    return ds_train, ds_val


def prepare_datasets_nosm(
    train_data_path: str | Path,
    val_data_path: str | Path,
    metadata_path: str | Path,
) -> Tuple[RoutesStepsDataset, ...]:
    with open(train_data_path, "rb") as f:
        (products, path_strings, n_steps_list) = pickle.load(f)
    ds_train = RoutesStepsDataset(
        products=products,
        path_strings=path_strings,
        n_steps_list=n_steps_list,
        metadata_path=str(metadata_path),
    )
    with open(val_data_path, "rb") as f:
        (n1_products, n1_path_strings, n1_steps_list) = pickle.load(f)
    ds_val = RoutesStepsDataset(
        products=n1_products,
        path_strings=n1_path_strings,
        n_steps_list=n1_steps_list,
        metadata_path=str(metadata_path),
    )
    return ds_train, ds_val


def find_checkpoint(train_path: str | Path, run_name: str) -> Optional[Path]:
    ckpt_path = Path(train_path) / run_name
    checkpoints = list(ckpt_path.glob("*.ckpt"))

    # First, check if there's a file with "last" in its name
    last_checkpoints = [ckpt for ckpt in checkpoints if "last" in ckpt.stem]
    if last_checkpoints:
        # Extract version number if present, else default to 0 (e.g., last.ckpt is treated as v0)
        def parse_version(ckpt: Path) -> int:
            match = re.search(r"last-v(\d+)", ckpt.stem)
            return int(match.group(1)) if match else 0

        # Sort by version number in descending order and return the latest
        return sorted(last_checkpoints, key=parse_version, reverse=True)[0]

    # If no "last" file, find the checkpoint with the largest epoch and step
    def parse_epoch_step(filename: str):
        # This pattern will match 'epoch=X-step=Y.ckpt' and extract X and Y
        match = re.search(r"epoch=(\d+)-step=(\d+)\.ckpt", filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return -1, -1  # Default to -1 if no match found

    checkpoints.sort(key=lambda ckpt: parse_epoch_step(ckpt.name), reverse=True)
    return checkpoints[0] if checkpoints else None


if __name__ == "__main__":
    train_path = Path(__file__).resolve().parent / "Data" / "Training"
    run_name = "moe_3x2_3x3_002_local"
    o = find_checkpoint(train_path, run_name)
    print(f"{o=}")
