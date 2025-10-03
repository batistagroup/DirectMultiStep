# Batched Route Generation

The `BatchedBeamSearch` class provides efficient batched route generation for multiple target molecules simultaneously, with support for variable batch sizes and lengths.

## Features

- **Variable Batch Sizes**: Process any number of targets in a single batch
- **Variable Path Start Lengths**: Each target can have different starting material lengths
- **Variable Target Lengths**: Different maximum output lengths per target
- **Early Termination**: Each batch item can finish independently
- **GPU Efficient**: Optimized batching for maximum GPU utilization

## Basic Usage

### Single Target (Compatible with BeamSearchOptimized)

```python
from pathlib import Path
from directmultistep import generate_routes

target = "CNCc1ccccc1"
starting_material = "CN"
n_steps = 1

routes = generate_routes(
    target=target,
    n_steps=n_steps,
    starting_material=starting_material,
    beam_size=5,
    model="flash",
    config_path=Path("data/configs/dms_dictionary.yaml"),
    ckpt_dir=Path("data/checkpoints"),
)

for route in routes:
    print(route)
```

### Multiple Targets (Batched)

```python
from pathlib import Path
from directmultistep import generate_routes_batched

targets = [
    "CNCc1ccccc1",
    "CCOc1ccccc1",
    "c1ccccc1",
]

n_steps_list = [1, 2, 1]

starting_materials = [
    "CN",
    None,
    None,
]

routes = generate_routes_batched(
    targets=targets,
    n_steps_list=n_steps_list,
    starting_materials=starting_materials,
    beam_size=5,
    model="flash",
    config_path=Path("data/configs/dms_dictionary.yaml"),
    ckpt_dir=Path("data/checkpoints"),
)

for i, (target, routes_for_target) in enumerate(zip(targets, routes)):
    print(f"Target {i+1}: {target}")
    print(f"Routes: {len(routes_for_target)}")
    for route in routes_for_target[:3]:
        print(f"  {route}")
```

## Advanced Usage

### Using the Low-Level API

For more control, you can use the lower-level APIs:

```python
from pathlib import Path
import torch
from directmultistep import (
    load_published_model,
    create_batched_beam_search,
    prepare_batched_input_tensors,
)
from directmultistep.utils.dataset import RoutesProcessing

# Load model
model = load_published_model("flash", Path("data/checkpoints"))
rds = RoutesProcessing(metadata_path=Path("data/configs/dms_dictionary.yaml"))

# Create batched beam search
beam_search = create_batched_beam_search(model, beam_size=5, rds=rds)

# Prepare batched inputs
targets = ["CNCc1ccccc1", "CCOc1ccccc1"]
n_steps_list = [1, 2]
starting_materials = ["CN", None]

encoder_batch, steps_batch, path_starts, target_lengths = prepare_batched_input_tensors(
    targets=targets,
    n_steps_list=n_steps_list,
    starting_materials=starting_materials,
    rds=rds,
    product_max_length=rds.product_max_length,
    sm_max_length=rds.sm_max_length,
)

# Run batched beam search
device = next(model.parameters()).device
results = beam_search.decode(
    src_BC=encoder_batch.to(device),
    steps_B1=steps_batch.to(device) if steps_batch is not None else None,
    path_starts=[ps.to(device) for ps in path_starts],
    target_lengths=target_lengths,
    progress_bar=True,
)

# Results is a list of lists: results[batch_idx][beam_idx] = (sequence, log_prob)
for batch_idx, beam_results in enumerate(results):
    print(f"\nTarget {batch_idx}: {targets[batch_idx]}")
    for beam_idx, (sequence, log_prob) in enumerate(beam_results):
        print(f"  Beam {beam_idx}: score={log_prob:.2f}, seq={sequence[:50]}...")
```

### Custom Batch Processing

You can also directly use `BatchedBeamSearch` for custom processing:

```python
from directmultistep.generation.tensor_gen import BatchedBeamSearch

# Create custom beam search with specific parameters
beam_search = BatchedBeamSearch(
    model=model,
    beam_size=10,
    start_idx=0,
    pad_idx=52,
    end_idx=22,
    max_length=1074,
    idx_to_token=rds.idx_to_token,
    device=device,
)

# Use with custom target lengths per batch item
results = beam_search.decode(
    src_BC=encoder_batch,
    steps_B1=steps_batch,
    path_starts=path_starts,
    target_lengths=[500, 1000, 1500],  # Different max length per target
    progress_bar=True,
)
```

## API Reference

### High-Level Functions

#### `generate_routes_batched`

```python
def generate_routes_batched(
    targets: Sequence[str],
    n_steps_list: Sequence[int] | None,
    starting_materials: Sequence[str | None],
    beam_size: int,
    model: ModelName | torch.nn.Module,
    config_path: Path,
    ckpt_dir: Path | None = None,
    commercial_stock: set[str] | None = None,
    use_fp16: bool = False,
) -> list[list[str]]:
```

Generate synthesis routes for multiple targets using batched beam search.

**Arguments:**
- `targets`: List of SMILES strings of target molecules
- `n_steps_list`: List of number of synthesis steps for each target or None (for explorer)
- `starting_materials`: List of starting materials for each target (can contain None)
- `beam_size`: Beam size for the beam search
- `model`: Either a model name or a torch.nn.Module
- `config_path`: Path to the model configuration file
- `ckpt_dir`: Directory containing model checkpoints (required if model is a string)
- `commercial_stock`: Set of commercially available starting materials (SMILES)
- `use_fp16`: Whether to use half precision (FP16)

**Returns:**
- List of lists, where each inner list contains valid routes for the corresponding target

### Utility Functions

#### `prepare_batched_input_tensors`

```python
def prepare_batched_input_tensors(
    targets: Sequence[str],
    n_steps_list: Sequence[int] | None,
    starting_materials: Sequence[str | None],
    rds: RoutesProcessing,
    product_max_length: int,
    sm_max_length: int,
    use_fp16: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor], list[int]]:
```

Prepare batched input tensors for the model.

**Returns:**
- `encoder_batch`: Batched input tensor for the encoder [B, C]
- `steps_batch`: Batched tensor of steps [B, 1], or None if all n_steps are None
- `path_starts`: List of initial path tensors for decoder (variable lengths)
- `target_lengths`: List of target max lengths per batch item

#### `create_batched_beam_search`

```python
def create_batched_beam_search(
    model: torch.nn.Module,
    beam_size: int,
    rds: RoutesProcessing
) -> BatchedBeamSearch:
```

Create a batched beam search object that supports variable batch sizes and lengths.

### BatchedBeamSearch Class

```python
class BatchedBeamSearch:
    def __init__(
        self,
        model: nn.Module,
        beam_size: int,
        start_idx: int,
        pad_idx: int,
        end_idx: int,
        max_length: int,
        idx_to_token: dict[int, str],
        device: torch.device,
    ):
        ...

    def decode(
        self,
        src_BC: Tensor,
        steps_B1: Tensor | None,
        path_starts: list[Tensor | None] | None = None,
        target_lengths: list[int] | None = None,
        progress_bar: bool = True,
        token_processor: Callable[[list[str]], str] | None = None,
    ) -> list[list[tuple[str, float]]]:
        ...
```

## Performance Considerations

1. **Batch Size**: Larger batches improve GPU utilization but increase memory usage
2. **Variable Lengths**: The implementation handles variable lengths efficiently by grouping active beams
3. **Early Termination**: Batch items that finish early are removed from computation
4. **Memory Usage**: Peak memory scales with `batch_size * beam_size * max_sequence_length`

## Comparison with BeamSearchOptimized

| Feature | BeamSearchOptimized | BatchedBeamSearch |
|---------|-------------------|------------------|
| Batch Size | Only 1 | Any positive integer |
| Variable Path Starts | No | Yes |
| Variable Target Lengths | No | Yes |
| Early Termination | All beams together | Per batch item |
| API Compatibility | Single target | Multiple targets |

For single-target generation, both implementations produce identical results. For multiple targets, use `BatchedBeamSearch` for better efficiency.
