from pathlib import Path
from typing import Literal, cast

import torch
import torch.nn as nn

from directmultistep.generation.tensor_gen import BatchedBeamSearch
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


def load_published_model(
    model_name: ModelName, ckpt_dir: Path, use_fp16: bool = False, force_device: str | None = None
) -> torch.nn.Module:
    """Load a model by name from the available checkpoints.

    Args:
        model_name: Name of the model to load
        ckpt_dir: Directory containing model checkpoints
        use_fp16: Whether to use half precision (FP16) for model weights
    """
    if model_name not in MODEL_CHECKPOINTS:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(MODEL_CHECKPOINTS.keys())}")

    preset_name, ckpt_file = MODEL_CHECKPOINTS[model_name]
    device = ModelFactory.determine_device(force_device)
    model = ModelFactory.from_preset(preset_name, compile_model=False).create_model()
    model = ModelFactory.load_checkpoint(model, ckpt_dir / ckpt_file, device)

    if use_fp16:
        model = model.half()  # Convert to FP16

    return cast(nn.Module, model)


def create_beam_search(model: torch.nn.Module, beam_size: int, rds: RoutesProcessing) -> BeamSearch:
    """Create a beam search object and return product/sm max lengths and the beam search object."""
    device = next(model.parameters()).device

    beam = BeamSearch(
        model=model,
        beam_size=beam_size,
        start_idx=0,
        pad_idx=52,
        end_idx=22,
        max_length=1074,
        idx_to_token=rds.idx_to_token,
        device=device,
    )
    return beam


def create_batched_beam_search(model: torch.nn.Module, beam_size: int, rds: RoutesProcessing) -> BatchedBeamSearch:
    """Create a batched beam search object that supports variable batch sizes and lengths."""
    device = next(model.parameters()).device

    beam = BatchedBeamSearch(
        model=model,
        beam_size=beam_size,
        start_idx=0,
        pad_idx=52,
        end_idx=22,
        max_length=1074,
        idx_to_token=rds.idx_to_token,
        device=device,
    )
    return beam


def prepare_input_tensors(
    target: str,
    n_steps: int | None,
    starting_material: str | None,
    rds: RoutesProcessing,
    product_max_length: int,
    sm_max_length: int,
    use_fp16: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Prepare input tensors for the model.
    Args:
        target: SMILES string of the target molecule.
        n_steps: Number of synthesis steps.
        starting_material: SMILES string of the starting material, if any.
        rds: RoutesProcessing object for tokenization.
        product_max_length: Maximum length of the product SMILES sequence.
        sm_max_length: Maximum length of the starting material SMILES sequence.
        use_fp16: Whether to use half precision (FP16) for tensors.
        path_start: Initial path string to start generation from.
    Returns:
        A tuple containing:
            - encoder_inp: Input tensor for the encoder.
            - steps_tens: Tensor of the number of steps, or None if not provided.
            - path_tens: Initial path tensor for the decoder.
    """
    if starting_material:
        prod_tens = rds.smile_to_tokens(target, product_max_length)
        sm_tens = rds.smile_to_tokens(starting_material, sm_max_length)
        encoder_inp = torch.cat([prod_tens, sm_tens], dim=0).unsqueeze(0)
    else:
        prod_tens = rds.smile_to_tokens(target, product_max_length + sm_max_length)
        encoder_inp = torch.cat([prod_tens], dim=0).unsqueeze(0)

    steps_tens = torch.tensor([n_steps]).unsqueeze(0) if n_steps is not None else None
    path_start = "{'smiles':'" + target + "','children':[{'smiles':'"
    path_tens = rds.path_string_to_tokens(path_start, max_length=None, add_eos=False).unsqueeze(0)

    if use_fp16:
        encoder_inp = encoder_inp.half()
        if steps_tens is not None:
            steps_tens = steps_tens.half()
        path_tens = path_tens.half()

    return encoder_inp, steps_tens, path_tens


def prepare_batched_input_tensors(
    targets: list[str],
    n_steps_list: list[int | None],
    starting_materials: list[str | None],
    rds: RoutesProcessing,
    product_max_length: int,
    sm_max_length: int,
    use_fp16: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor], list[int]]:
    """Prepare batched input tensors for the model.

    Args:
        targets: List of SMILES strings of target molecules
        n_steps_list: List of number of synthesis steps for each target (can contain None)
        starting_materials: List of SMILES strings of starting materials (can contain None)
        rds: RoutesProcessing object for tokenization
        product_max_length: Maximum length of the product SMILES sequence
        sm_max_length: Maximum length of the starting material SMILES sequence
        use_fp16: Whether to use half precision (FP16) for tensors

    Returns:
        A tuple containing:
            - encoder_batch: Batched input tensor for the encoder [B, C]
            - steps_batch: Batched tensor of steps [B, 1], or None if all n_steps are None
            - path_starts: List of initial path tensors for decoder (variable lengths)
            - target_lengths: List of target max lengths per batch item
    """
    if len(targets) != len(n_steps_list) or len(targets) != len(starting_materials):
        raise ValueError(
            f"Length mismatch: targets={len(targets)}, "
            f"n_steps_list={len(n_steps_list)}, starting_materials={len(starting_materials)}"
        )

    encoder_inputs = []
    steps_tensors = []
    path_starts = []
    target_lengths = []

    for target, n_steps, sm in zip(targets, n_steps_list, starting_materials, strict=False):
        encoder_inp, steps_tens, path_tens = prepare_input_tensors(
            target=target,
            n_steps=n_steps,
            starting_material=sm,
            rds=rds,
            product_max_length=product_max_length,
            sm_max_length=sm_max_length,
            use_fp16=use_fp16,
        )

        encoder_inputs.append(encoder_inp.squeeze(0))
        steps_tensors.append(steps_tens.squeeze(0) if steps_tens is not None else None)
        path_starts.append(path_tens.squeeze(0))
        target_lengths.append(1074)

    encoder_batch = torch.stack(encoder_inputs)
    steps_batch = (
        torch.stack([s for s in steps_tensors if s is not None]) if all(s is not None for s in steps_tensors) else None
    )

    return encoder_batch, steps_batch, path_starts, target_lengths


def generate_routes(
    target: str,
    n_steps: int | None,
    starting_material: str | None,
    beam_size: int,
    model: ModelName | torch.nn.Module,
    config_path: Path,
    ckpt_dir: Path | None = None,
    commercial_stock: set[str] | None = None,
    use_fp16: bool = False,
    show_progress: bool = True,
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
        stock_set: Set of commercially available starting materials (SMILES).
        use_fp16: Whether to use half precision (FP16) for model weights and computations
    """
    # Handle model loading and validation
    if isinstance(model, str):
        if ckpt_dir is None:
            raise ValueError("ckpt_dir must be provided when model is specified by name")
        validate_model_constraints(model, n_steps, starting_material)
        model = load_published_model(model, ckpt_dir, use_fp16)

    rds = RoutesProcessing(metadata_path=config_path)
    beam_obj = create_beam_search(model, beam_size, rds)

    # Prepare input tensors
    encoder_inp, steps_tens, path_tens = prepare_input_tensors(
        target, n_steps, starting_material, rds, rds.product_max_length, rds.sm_max_length, use_fp16
    )

    # Run beam search
    device = ModelFactory.determine_device()
    all_beam_results_NS2: list[list[tuple[str, float]]] = []
    beam_result_BS2 = beam_obj.decode(
        src_BC=encoder_inp.to(device),
        steps_B1=steps_tens.to(device) if steps_tens is not None else None,
        path_start_BL=path_tens.to(device),
        progress_bar=show_progress,
    )
    for beam_result_S2 in beam_result_BS2:
        all_beam_results_NS2.append(beam_result_S2)

    # Process results
    valid_paths_NS2n = find_valid_paths(all_beam_results_NS2)
    correct_paths_NS2n = process_path_single(
        paths_NS2n=valid_paths_NS2n,
        true_products=[target],
        true_reacs=[starting_material] if starting_material else None,
        commercial_stock=commercial_stock,
    )
    return [beam_result[0] for beam_result in correct_paths_NS2n[0]]


def generate_routes_batched(
    targets: list[str],
    n_steps_list: list[int | None],
    starting_materials: list[str | None],
    beam_size: int,
    model: ModelName | torch.nn.Module,
    config_path: Path,
    ckpt_dir: Path | None = None,
    commercial_stock: set[str] | None = None,
    use_fp16: bool = False,
) -> list[list[str]]:
    """Generate synthesis routes for multiple targets using batched beam search.

    Args:
        targets: List of SMILES strings of target molecules
        n_steps_list: List of number of synthesis steps for each target (can contain None)
        starting_materials: List of starting materials for each target (can contain None)
        beam_size: Beam size for the beam search
        model: Either a model name or a torch.nn.Module
        config_path: Path to the model configuration file
        ckpt_dir: Directory containing model checkpoints (required if model is a string)
        commercial_stock: Set of commercially available starting materials (SMILES)
        use_fp16: Whether to use half precision (FP16) for model weights and computations

    Returns:
        List of lists, where each inner list contains valid routes for the corresponding target
    """
    if isinstance(model, str):
        if ckpt_dir is None:
            raise ValueError("ckpt_dir must be provided when model is specified by name")
        for _target, n_steps, sm in zip(targets, n_steps_list, starting_materials, strict=False):
            validate_model_constraints(model, n_steps, sm)
        model = load_published_model(model, ckpt_dir, use_fp16)

    rds = RoutesProcessing(metadata_path=config_path)
    beam_obj = create_batched_beam_search(model, beam_size, rds)

    encoder_batch, steps_batch, path_starts, target_lengths = prepare_batched_input_tensors(
        targets=targets,
        n_steps_list=n_steps_list,
        starting_materials=starting_materials,
        rds=rds,
        product_max_length=rds.product_max_length,
        sm_max_length=rds.sm_max_length,
        use_fp16=use_fp16,
    )

    device = next(model.parameters()).device
    beam_results = beam_obj.decode(
        src_BC=encoder_batch.to(device),
        steps_B1=steps_batch.to(device) if steps_batch is not None else None,
        path_starts=[ps.to(device) for ps in path_starts],
        target_lengths=target_lengths,
        progress_bar=True,
    )

    all_results = []
    for idx, (target, sm) in enumerate(zip(targets, starting_materials, strict=False)):
        valid_paths = find_valid_paths([beam_results[idx]])
        correct_paths = process_path_single(
            paths_NS2n=valid_paths,
            true_products=[target],
            true_reacs=[sm] if sm else None,
            commercial_stock=commercial_stock,
        )
        all_results.append([beam_result[0] for beam_result in correct_paths[0]])

    return all_results
