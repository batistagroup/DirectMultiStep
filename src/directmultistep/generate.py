from pathlib import Path
from typing import Literal

import torch
import yaml

from directmultistep.generation.tensor_gen import BeamSearchOptimized as BeamSearch
from directmultistep.model import ModelFactory
from directmultistep.utils.dataset import RoutesProcessing
from directmultistep.utils.post_process import find_valid_paths, process_path_single

ModelName = Literal["flash", "flash-20M", "flex-20M", "deep", "wide", "explorer", "explorer XL"]

MODEL_CHECKPOINTS = {
    "flash": ("flash_10M", "flash.ckpt"),
    "flash-20M": ("flash_20M", "flash_20.ckpt"),
    "flex-20M": ("flex_20M", "flex.ckpt"),
    "deep": ("deep_40M", "deep.ckpt"),
    "wide": ("wide_40M", "wide.ckpt"),
    "explorer": ("explorer_19M", "explorer.ckpt"),
    "explorer XL": ("explorer_xl_50M", "explorer_xl.ckpt"),
}


def validate_model_constraints(model_name: ModelName, n_steps: int | None, starting_material: str | None) -> None:
    """Validate model-specific constraints for route generation."""
    if model_name in ["deep", "wide"] and starting_material is not None:
        raise ValueError(f"{model_name} model does not support starting material specification")
    if model_name == "explorer" and n_steps is not None:
        raise ValueError("explorer model does not support step count specification")
    if model_name == "explorer XL" and (n_steps is not None or starting_material is not None):
        raise ValueError("explorer XL model does not support step count or starting material specification")


def load_model(model_name: ModelName, ckpt_dir: Path) -> torch.nn.Module:
    """Load a model by name from the available checkpoints."""
    if model_name not in MODEL_CHECKPOINTS:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(MODEL_CHECKPOINTS.keys())}")

    preset_name, ckpt_file = MODEL_CHECKPOINTS[model_name]
    device = torch.device("cpu")  # For now, we'll use CPU for all models
    model = ModelFactory.from_preset(preset_name, compile_model=False).create_model()
    return ModelFactory.load_checkpoint(model, ckpt_dir / ckpt_file, device)


def create_beam_search(model: torch.nn.Module, beam_size: int, config_path: Path) -> tuple[int, int, BeamSearch]:
    """Create a beam search object and return product/sm max lengths and the beam search object."""
    device = next(model.parameters()).device
    with open(config_path, "rb") as file:
        data = yaml.safe_load(file)
        idx_to_token = data["invdict"]
        product_max_length = data["product_max_length"]
        sm_max_length = data["sm_max_length"]

    beam = BeamSearch(
        model=model,
        beam_size=beam_size,
        start_idx=0,
        pad_idx=52,
        end_idx=22,
        max_length=1074,
        idx_to_token=idx_to_token,
        device=device,
    )
    return product_max_length, sm_max_length, beam


def prepare_input_tensors(
    target: str,
    n_steps: int | None,
    starting_material: str | None,
    rds: RoutesProcessing,
    product_max_length: int,
    sm_max_length: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Prepare input tensors for the model."""
    prod_tens = rds.smile_to_tokens(target, product_max_length)
    if starting_material:
        sm_tens = rds.smile_to_tokens(starting_material, sm_max_length)
        encoder_inp = torch.cat([prod_tens, sm_tens], dim=0).unsqueeze(0)
    else:
        encoder_inp = torch.cat([prod_tens], dim=0).unsqueeze(0)

    steps_tens = torch.tensor([n_steps]).unsqueeze(0) if n_steps is not None else None
    path_start = "{'smiles':'" + target + "','children':[{'smiles':'"
    path_tens = rds.path_string_to_tokens(path_start, max_length=None, add_eos=False).unsqueeze(0)

    return encoder_inp, steps_tens, path_tens


def generate_routes(
    target: str,
    n_steps: int | None,
    starting_material: str | None,
    beam_size: int,
    model: ModelName | torch.nn.Module,
    config_path: Path,
    ckpt_dir: Path | None = None,
) -> list[str]:
    """Generate synthesis routes using the model.

    Args:
        target: SMILES string of the target molecule
        n_steps: Number of synthesis steps. If None, will try multiple steps
        starting_material: Optional SMILES string of the starting material
        beam_size: Beam size for the beam search
        model: Either a model name or a torch.nn.Module
        config_path: Path to the model configuration file
        ckpt_dir: Directory containing model checkpoints (required if model is a string)
    """
    # Handle model loading and validation
    if isinstance(model, str):
        if ckpt_dir is None:
            raise ValueError("ckpt_dir must be provided when model is specified by name")
        validate_model_constraints(model, n_steps, starting_material)
        model = load_model(model, ckpt_dir)

    rds = RoutesProcessing(metadata_path=config_path)
    product_max_length, sm_max_length, beam_obj = create_beam_search(model, beam_size, config_path)

    # Prepare input tensors
    encoder_inp, steps_tens, path_tens = prepare_input_tensors(
        target, n_steps, starting_material, rds, product_max_length, sm_max_length
    )

    # Run beam search
    device = ModelFactory.determine_device()
    all_beam_results_NS2: list[list[tuple[str, float]]] = []
    beam_result_BS2 = beam_obj.decode(
        src_BC=encoder_inp.to(device),
        steps_B1=steps_tens.to(device) if steps_tens is not None else None,
        path_start_BL=path_tens.to(device),
    )
    for beam_result_S2 in beam_result_BS2:
        all_beam_results_NS2.append(beam_result_S2)

    # Process results
    valid_paths_NS2n = find_valid_paths(all_beam_results_NS2)
    correct_paths_NS2n = process_path_single(
        paths_NS2n=valid_paths_NS2n,
        true_products=[target],
        true_reacs=[starting_material] if starting_material else None,
        commercial_stock=None,
    )
    return [beam_result[0] for beam_result in correct_paths_NS2n[0]]
