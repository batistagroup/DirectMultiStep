import pickle
import re
from pathlib import Path

from directmultistep.utils.dataset import RoutesDataset


def prepare_datasets(
    train_data_path: Path,
    val_data_path: Path,
    metadata_path: Path,
    load_sm: bool = True,
    mode: str = "training",
) -> tuple[RoutesDataset, ...]:
    with open(train_data_path, "rb") as f:
        (products, starting_materials, path_strings, n_steps_list) = pickle.load(f)
    if not load_sm:
        starting_materials = None
    ds_train = RoutesDataset(
        metadata_path=metadata_path,
        products=products,
        starting_materials=starting_materials,
        path_strings=path_strings,
        n_steps_list=n_steps_list,
        mode=mode,
    )
    with open(val_data_path, "rb") as f:
        (val_products, val_sms, val_path_strings, val_steps_list) = pickle.load(f)
    if not load_sm:
        val_sms = None
    ds_val = RoutesDataset(
        metadata_path=metadata_path,
        products=val_products,
        starting_materials=val_sms,
        path_strings=val_path_strings,
        n_steps_list=val_steps_list,
        mode=mode,
    )
    return ds_train, ds_val


def find_checkpoint(train_path: Path, run_name: str) -> Path | None:
    ckpt_path = train_path / run_name
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
    def parse_epoch_step(filename: str) -> tuple[int, int]:
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
