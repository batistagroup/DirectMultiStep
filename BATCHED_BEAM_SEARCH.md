# Batched Beam Search Implementation

## Summary

This document summarizes the implementation of full batched beam search support for DirectMultiStep, enabling efficient route generation for multiple targets with variable batch sizes and lengths.

## Problem Statement

The original `BeamSearchOptimized` class expected batched inputs but only worked correctly for batch size 1. Processing multiple targets required sequential calls, which was inefficient for GPU utilization.

## Solution

Implemented a new `BatchedBeamSearch` class that provides true batched processing with support for:
- Variable batch sizes (B can be any positive integer)
- Different path start lengths per batch item
- Different target max lengths per batch item
- Early termination per batch item when beams complete
- Efficient GPU utilization through dynamic batching

## Implementation Details

### Core Algorithm

The implementation follows this approach:

1. **Independent Tracking**: Each batch item maintains its own beam states, positions, and finished list
2. **Dynamic Batching**: Active beams from all batch items are grouped for efficient GPU forward passes
3. **Beam Management**: Each batch item independently selects its top beams based on normalized scores
4. **Early Termination**: Batch items finish independently when all beams complete or reach max length

### Key Files Modified/Created

1. **src/directmultistep/generation/tensor_gen.py**
   - Added `BatchedBeamSearch` class (lines 30-237)
   - Kept `BeamSearchOptimized` for backward compatibility

2. **src/directmultistep/generate.py**
   - Added `create_batched_beam_search()` function
   - Added `prepare_batched_input_tensors()` utility
   - Added `generate_routes_batched()` high-level API

3. **src/directmultistep/__init__.py**
   - Exported new batched functions and classes

4. **tests/generation/test_batched_beam_search.py**
   - Comprehensive test suite with 13+ tests
   - Tests for variable batch sizes, lengths, and edge cases
   - Comparison tests ensuring equivalence with `BeamSearchOptimized` for single batch

5. **examples/batched_generation_example.py**
   - Simple usage example

6. **docs/batched-generation.md**
   - Complete API documentation and usage guide

## API Overview

### High-Level API (Recommended)

```python
from directmultistep import generate_routes_batched

routes = generate_routes_batched(
    targets=["CNCc1ccccc1", "CCOc1ccccc1"],
    n_steps_list=[1, 2],
    starting_materials=["CN", None],
    beam_size=5,
    model="flash",
    config_path=Path("data/configs/dms_dictionary.yaml"),
    ckpt_dir=Path("data/checkpoints"),
)
```

### Mid-Level API

```python
from directmultistep import (
    create_batched_beam_search,
    prepare_batched_input_tensors,
)

beam_search = create_batched_beam_search(model, beam_size=5, rds=rds)
encoder_batch, steps_batch, path_starts, target_lengths = prepare_batched_input_tensors(...)

results = beam_search.decode(
    src_BC=encoder_batch.to(device),
    steps_B1=steps_batch.to(device),
    path_starts=[ps.to(device) for ps in path_starts],
    target_lengths=target_lengths,
)
```

### Low-Level API

```python
from directmultistep.generation.tensor_gen import BatchedBeamSearch

beam_search = BatchedBeamSearch(
    model=model,
    beam_size=beam_size,
    start_idx=0,
    pad_idx=52,
    end_idx=22,
    max_length=1074,
    idx_to_token=idx_to_token,
    device=device,
)
```

## Testing

### Test Coverage

1. **Basic Functionality**
   - Initialization
   - Single and multiple batch decoding
   - Variable path start lengths
   - Variable target lengths
   - None path starts handling

2. **Correctness Verification**
   - Comparison with `BeamSearchOptimized` for single batch
   - Beam ordering verification
   - Consistency across different batch sizes

3. **Edge Cases**
   - Mixed None/tensor path starts
   - Large batch sizes
   - Different target lengths per batch

### Running Tests

```bash
pytest tests/generation/test_batched_beam_search.py -v
```

## Performance Characteristics

- **Memory**: O(B × S × L) where B=batch size, S=beam size, L=max length
- **Computation**: Active beams are batched for efficient GPU utilization
- **Early Termination**: Batch items that finish early reduce computation load

## Backward Compatibility

- `BeamSearchOptimized` remains unchanged
- Existing code using `generate_routes()` works as before
- New `BatchedBeamSearch` is opt-in via new functions

## Future Enhancements

Potential improvements:
1. Flash Attention support for longer sequences
2. Adaptive beam size per batch item
3. Beam pruning based on score thresholds
4. Multi-GPU support for very large batches

## References

- Original implementation: `BeamSearchOptimized` in `tensor_gen.py`
- Test suite: `tests/generation/test_batched_beam_search.py`
- Documentation: `docs/batched-generation.md`
- Example: `examples/batched_generation_example.py`
