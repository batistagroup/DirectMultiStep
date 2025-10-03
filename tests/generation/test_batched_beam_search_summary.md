# Batched Beam Search Test Suite

## Test Organization

### TestBatchedBeamSearch
Basic initialization test only - verifies the object is created correctly.

### TestBatchedVsOptimizedComparison
**Core correctness tests** - These verify that `BatchedBeamSearch` produces identical results to `BeamSearchOptimized`:

1. **test_single_batch_equivalence**: Single batch without starting material
2. **test_single_batch_equivalence_with_sm**: Single batch with starting material  
3. **test_multiple_batches_independently_correct**: Multiple batches processed together match individual processing

## Key Points

- All tests verify **actual correctness** by comparing sequences and probabilities
- No tests that only check types/shapes without validating output
- Fast execution with simple molecules (C, CC, etc.)
- Comprehensive coverage of batching scenarios

## Running Tests

```bash
# Run all batched beam search tests
pytest tests/generation/test_batched_beam_search.py -v

# Run only correctness comparison tests
pytest tests/generation/test_batched_beam_search.py::TestBatchedVsOptimizedComparison -v
```
