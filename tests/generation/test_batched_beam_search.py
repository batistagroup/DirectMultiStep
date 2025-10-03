"""
Tests for BatchedBeamSearch with variable batch sizes and lengths.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from directmultistep.generation.tensor_gen import BatchedBeamSearch, BeamSearchOptimized
from directmultistep.utils.dataset import RoutesProcessing

torch.manual_seed(42)
np.random.seed(42)


class TestBatchedBeamSearch:
    """Test suite for BatchedBeamSearch functionality."""

    @pytest.fixture(scope="class")
    def model_components(self):
        """Load model and create batched beam search components."""
        config_path = Path("data/configs/dms_dictionary.yaml")
        ckpt_dir = Path("data/checkpoints")

        if not config_path.exists() or not ckpt_dir.exists():
            pytest.skip("Model files not found. Ensure data is downloaded.")

        from directmultistep.generate import load_published_model

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

        return model, rds, beam_obj

    def test_batched_beam_search_initialization(self, model_components):
        """Test that batched beam search object can be initialized properly."""
        model, rds, beam_obj = model_components

        assert beam_obj.model == model
        assert beam_obj.beam_size == 5
        assert beam_obj.start_idx == 0
        assert beam_obj.pad_idx == 52
        assert beam_obj.end_idx == 22
        assert beam_obj.max_length == 1074
        assert isinstance(beam_obj.idx_to_token, dict)
        assert isinstance(beam_obj.device, torch.device)

    def test_batched_decode_single_batch(self, model_components):
        """Test batched beam search with single batch item."""
        model, rds, beam_obj = model_components

        target = "CNCc1cc(-c2ccccc2F)n(S(=O)(=O)c2cccnc2)c1"
        starting_material = "CN"
        n_steps = 2

        from directmultistep.generate import prepare_input_tensors

        encoder_inp, steps_tens, path_tens = prepare_input_tensors(
            target, n_steps, starting_material, rds, rds.product_max_length, rds.sm_max_length
        )

        results = beam_obj.decode(
            src_BC=encoder_inp.to(beam_obj.device),
            steps_B1=steps_tens.to(beam_obj.device) if steps_tens is not None else None,
            path_starts=[path_tens[0].to(beam_obj.device)],
            progress_bar=False,
        )

        assert isinstance(results, list)
        assert len(results) == 1
        assert len(results[0]) == beam_obj.beam_size

        for path, log_prob in results[0]:
            assert isinstance(path, str)
            assert isinstance(log_prob, float)

    def test_batched_decode_multiple_batches(self, model_components):
        """Test batched beam search with multiple batch items."""
        model, rds, beam_obj = model_components

        targets = [
            "CNCc1cc(-c2ccccc2F)n(S(=O)(=O)c2cccnc2)c1",
            "CCOc1ccc(C(=O)N2CCN(c3ccccc3)CC2)cc1",
        ]
        starting_materials = ["CN", "CC"]
        n_steps_list = [2, 3]

        from directmultistep.generate import prepare_input_tensors

        encoder_inputs = []
        steps_tensors = []
        path_starts_list = []

        for target, sm, n_steps in zip(targets, starting_materials, n_steps_list, strict=False):
            encoder_inp, steps_tens, path_tens = prepare_input_tensors(
                target, n_steps, sm, rds, rds.product_max_length, rds.sm_max_length
            )
            encoder_inputs.append(encoder_inp[0])
            steps_tensors.append(steps_tens[0] if steps_tens is not None else None)
            path_starts_list.append(path_tens[0])

        encoder_batch = torch.stack(encoder_inputs).to(beam_obj.device)
        steps_batch = torch.stack(steps_tensors).to(beam_obj.device) if steps_tensors[0] is not None else None
        path_starts_batch = [ps.to(beam_obj.device) for ps in path_starts_list]

        results = beam_obj.decode(
            src_BC=encoder_batch,
            steps_B1=steps_batch,
            path_starts=path_starts_batch,
            progress_bar=False,
        )

        assert isinstance(results, list)
        assert len(results) == 2

        for batch_result in results:
            assert len(batch_result) == beam_obj.beam_size
            for path, log_prob in batch_result:
                assert isinstance(path, str)
                assert isinstance(log_prob, float)

    def test_variable_path_start_lengths(self, model_components):
        """Test batched beam search with different path start lengths."""
        model, rds, beam_obj = model_components

        targets = ["CNCc1ccccc1", "CCOc1ccccc1"]
        starting_materials = ["CN", "CCO"]
        n_steps_list = [1, 1]

        from directmultistep.generate import prepare_input_tensors

        encoder_inputs = []
        steps_tensors = []
        path_starts_list = []

        for target, sm, n_steps in zip(targets, starting_materials, n_steps_list, strict=False):
            encoder_inp, steps_tens, path_tens = prepare_input_tensors(
                target, n_steps, sm, rds, rds.product_max_length, rds.sm_max_length
            )
            encoder_inputs.append(encoder_inp[0])
            steps_tensors.append(steps_tens[0] if steps_tens is not None else None)
            path_starts_list.append(path_tens[0])

        assert path_starts_list[0].size(0) != path_starts_list[1].size(0), "Path starts should have different lengths"

        encoder_batch = torch.stack(encoder_inputs).to(beam_obj.device)
        steps_batch = torch.stack(steps_tensors).to(beam_obj.device) if steps_tensors[0] is not None else None
        path_starts_batch = [ps.to(beam_obj.device) for ps in path_starts_list]

        results = beam_obj.decode(
            src_BC=encoder_batch,
            steps_B1=steps_batch,
            path_starts=path_starts_batch,
            progress_bar=False,
        )

        assert len(results) == 2
        for batch_result in results:
            assert len(batch_result) == beam_obj.beam_size

    def test_variable_target_lengths(self, model_components):
        """Test batched beam search with different target max lengths per batch."""
        model, rds, beam_obj = model_components

        targets = ["C", "CCO"]
        n_steps_list = [1, 1]

        from directmultistep.generate import prepare_input_tensors

        encoder_inputs = []
        steps_tensors = []
        path_starts_list = []
        target_lengths = [50, 100]

        for target, n_steps in zip(targets, n_steps_list, strict=False):
            encoder_inp, steps_tens, path_tens = prepare_input_tensors(
                target, n_steps, None, rds, rds.product_max_length, rds.sm_max_length
            )
            encoder_inputs.append(encoder_inp[0])
            steps_tensors.append(steps_tens[0] if steps_tens is not None else None)
            path_starts_list.append(path_tens[0])

        encoder_batch = torch.stack(encoder_inputs).to(beam_obj.device)
        steps_batch = torch.stack(steps_tensors).to(beam_obj.device) if steps_tensors[0] is not None else None
        path_starts_batch = [ps.to(beam_obj.device) for ps in path_starts_list]

        results = beam_obj.decode(
            src_BC=encoder_batch,
            steps_B1=steps_batch,
            path_starts=path_starts_batch,
            target_lengths=target_lengths,
            progress_bar=False,
        )

        assert len(results) == 2
        for batch_result in results:
            assert len(batch_result) == beam_obj.beam_size

    def test_beam_ordering_in_batched(self, model_components):
        """Test that beams are ordered by log probability within each batch."""
        model, rds, beam_obj = model_components

        targets = ["CNCc1ccccc1", "CCOc1ccccc1"]
        n_steps_list = [1, 1]

        from directmultistep.generate import prepare_input_tensors

        encoder_inputs = []
        steps_tensors = []
        path_starts_list = []

        for target, n_steps in zip(targets, n_steps_list, strict=False):
            encoder_inp, steps_tens, path_tens = prepare_input_tensors(
                target, n_steps, None, rds, rds.product_max_length, rds.sm_max_length
            )
            encoder_inputs.append(encoder_inp[0])
            steps_tensors.append(steps_tens[0] if steps_tens is not None else None)
            path_starts_list.append(path_tens[0])

        encoder_batch = torch.stack(encoder_inputs).to(beam_obj.device)
        steps_batch = torch.stack(steps_tensors).to(beam_obj.device) if steps_tensors[0] is not None else None
        path_starts_batch = [ps.to(beam_obj.device) for ps in path_starts_list]

        torch.manual_seed(42)
        results = beam_obj.decode(
            src_BC=encoder_batch,
            steps_B1=steps_batch,
            path_starts=path_starts_batch,
            progress_bar=False,
        )

        for batch_idx, batch_result in enumerate(results):
            log_probs = [log_prob for _, log_prob in batch_result]
            for i in range(len(log_probs) - 1):
                assert log_probs[i] >= log_probs[i + 1], (
                    f"Batch {batch_idx}: Beams should be ordered by log probability: "
                    f"beam {i} ({log_probs[i]:.6f}) should be >= beam {i + 1} ({log_probs[i + 1]:.6f})"
                )

    def test_none_path_starts(self, model_components):
        """Test batched beam search with None path starts."""
        model, rds, beam_obj = model_components

        targets = ["C", "CC"]
        n_steps_list = [1, 1]

        from directmultistep.generate import prepare_input_tensors

        encoder_inputs = []
        steps_tensors = []

        for target, n_steps in zip(targets, n_steps_list, strict=False):
            encoder_inp, steps_tens, _ = prepare_input_tensors(
                target, n_steps, None, rds, rds.product_max_length, rds.sm_max_length
            )
            encoder_inputs.append(encoder_inp[0])
            steps_tensors.append(steps_tens[0] if steps_tens is not None else None)

        encoder_batch = torch.stack(encoder_inputs).to(beam_obj.device)
        steps_batch = torch.stack(steps_tensors).to(beam_obj.device) if steps_tensors[0] is not None else None

        results = beam_obj.decode(
            src_BC=encoder_batch,
            steps_B1=steps_batch,
            path_starts=None,
            progress_bar=False,
        )

        assert len(results) == 2
        for batch_result in results:
            assert len(batch_result) == beam_obj.beam_size

    def test_mixed_none_and_tensor_path_starts(self, model_components):
        """Test batched beam search with mixed None and tensor path starts."""
        model, rds, beam_obj = model_components

        targets = ["CNCc1ccccc1", "CCOc1ccccc1"]
        starting_materials = ["CN", None]
        n_steps_list = [1, 1]

        from directmultistep.generate import prepare_input_tensors

        encoder_inputs = []
        steps_tensors = []
        path_starts_list = []

        for target, sm, n_steps in zip(targets, starting_materials, n_steps_list, strict=False):
            encoder_inp, steps_tens, path_tens = prepare_input_tensors(
                target, n_steps, sm, rds, rds.product_max_length, rds.sm_max_length
            )
            encoder_inputs.append(encoder_inp[0])
            steps_tensors.append(steps_tens[0] if steps_tens is not None else None)
            if sm is None:
                path_starts_list.append(None)
            else:
                path_starts_list.append(path_tens[0])

        encoder_batch = torch.stack(encoder_inputs).to(beam_obj.device)
        steps_batch = torch.stack(steps_tensors).to(beam_obj.device) if steps_tensors[0] is not None else None
        path_starts_batch = [ps.to(beam_obj.device) if ps is not None else None for ps in path_starts_list]

        results = beam_obj.decode(
            src_BC=encoder_batch,
            steps_B1=steps_batch,
            path_starts=path_starts_batch,
            progress_bar=False,
        )

        assert len(results) == 2
        for batch_result in results:
            assert len(batch_result) == beam_obj.beam_size

    def test_large_batch_size(self, model_components):
        """Test batched beam search with larger batch size."""
        model, rds, beam_obj = model_components

        batch_size = 4
        targets = ["C", "CC", "CCC", "CCCC"]
        n_steps_list = [1] * batch_size

        from directmultistep.generate import prepare_input_tensors

        encoder_inputs = []
        steps_tensors = []

        for target, n_steps in zip(targets, n_steps_list, strict=False):
            encoder_inp, steps_tens, _ = prepare_input_tensors(
                target, n_steps, None, rds, rds.product_max_length, rds.sm_max_length
            )
            encoder_inputs.append(encoder_inp[0])
            steps_tensors.append(steps_tens[0] if steps_tens is not None else None)

        encoder_batch = torch.stack(encoder_inputs).to(beam_obj.device)
        steps_batch = torch.stack(steps_tensors).to(beam_obj.device) if steps_tensors[0] is not None else None

        results = beam_obj.decode(
            src_BC=encoder_batch,
            steps_B1=steps_batch,
            path_starts=None,
            progress_bar=False,
        )

        assert len(results) == batch_size
        for batch_result in results:
            assert len(batch_result) == beam_obj.beam_size
            for path, log_prob in batch_result:
                assert isinstance(path, str)
                assert isinstance(log_prob, float)
                assert not np.isnan(log_prob)


class TestBatchedVsOptimizedComparison:
    """Test that BatchedBeamSearch produces same results as BeamSearchOptimized for single batch."""

    @pytest.fixture(scope="class")
    def model_components(self):
        """Load model and create both beam search implementations."""
        config_path = Path("data/configs/dms_dictionary.yaml")
        ckpt_dir = Path("data/checkpoints")

        if not config_path.exists() or not ckpt_dir.exists():
            pytest.skip("Model files not found. Ensure data is downloaded.")

        from directmultistep.generate import load_published_model

        model = load_published_model("flash", ckpt_dir)
        rds = RoutesProcessing(metadata_path=config_path)
        
        device = next(model.parameters()).device
        
        batched_beam = BatchedBeamSearch(
            model=model,
            beam_size=5,
            start_idx=0,
            pad_idx=52,
            end_idx=22,
            max_length=1074,
            idx_to_token=rds.idx_to_token,
            device=device,
        )
        
        optimized_beam = BeamSearchOptimized(
            model=model,
            beam_size=5,
            start_idx=0,
            pad_idx=52,
            end_idx=22,
            max_length=1074,
            idx_to_token=rds.idx_to_token,
            device=device,
        )

        return model, rds, batched_beam, optimized_beam

    def test_single_batch_equivalence(self, model_components):
        """Test that BatchedBeamSearch gives same results as BeamSearchOptimized for single batch."""
        model, rds, batched_beam, optimized_beam = model_components

        target = "CNCc1cc(-c2ccccc2F)n(S(=O)(=O)c2cccnc2)c1"
        starting_material = "CN"
        n_steps = 2

        from directmultistep.generate import prepare_input_tensors

        encoder_inp, steps_tens, path_tens = prepare_input_tensors(
            target, n_steps, starting_material, rds, rds.product_max_length, rds.sm_max_length
        )

        encoder_inp = encoder_inp.to(batched_beam.device)
        steps_tens = steps_tens.to(batched_beam.device) if steps_tens is not None else None
        path_tens = path_tens.to(batched_beam.device)

        torch.manual_seed(42)
        batched_results = batched_beam.decode(
            src_BC=encoder_inp,
            steps_B1=steps_tens,
            path_starts=[path_tens[0]],
            progress_bar=False,
        )

        torch.manual_seed(42)
        optimized_results = optimized_beam.decode(
            src_BC=encoder_inp,
            steps_B1=steps_tens,
            path_start_BL=path_tens,
            progress_bar=False,
        )

        assert len(batched_results) == 1
        assert len(optimized_results) == 1
        
        batched_seqs = [seq for seq, _ in batched_results[0]]
        optimized_seqs = [seq for seq, _ in optimized_results[0]]
        
        batched_probs = [prob for _, prob in batched_results[0]]
        optimized_probs = [prob for _, prob in optimized_results[0]]

        for i, (b_seq, o_seq) in enumerate(zip(batched_seqs, optimized_seqs, strict=False)):
            assert b_seq == o_seq, f"Beam {i}: Sequence mismatch.\nBatched: {b_seq}\nOptimized: {o_seq}"

        for i, (b_prob, o_prob) in enumerate(zip(batched_probs, optimized_probs, strict=False)):
            assert abs(b_prob - o_prob) < 1e-5, (
                f"Beam {i}: Log prob mismatch. Batched: {b_prob:.6f}, Optimized: {o_prob:.6f}"
            )

    def test_single_batch_equivalence_no_sm(self, model_components):
        """Test equivalence when starting material is None."""
        model, rds, batched_beam, optimized_beam = model_components

        target = "CCOc1ccccc1"
        n_steps = 1

        from directmultistep.generate import prepare_input_tensors

        encoder_inp, steps_tens, path_tens = prepare_input_tensors(
            target, n_steps, None, rds, rds.product_max_length, rds.sm_max_length
        )

        encoder_inp = encoder_inp.to(batched_beam.device)
        steps_tens = steps_tens.to(batched_beam.device) if steps_tens is not None else None
        path_tens = path_tens.to(batched_beam.device)

        torch.manual_seed(42)
        batched_results = batched_beam.decode(
            src_BC=encoder_inp,
            steps_B1=steps_tens,
            path_starts=[path_tens[0]],
            progress_bar=False,
        )

        torch.manual_seed(42)
        optimized_results = optimized_beam.decode(
            src_BC=encoder_inp,
            steps_B1=steps_tens,
            path_start_BL=path_tens,
            progress_bar=False,
        )

        batched_seqs = [seq for seq, _ in batched_results[0]]
        optimized_seqs = [seq for seq, _ in optimized_results[0]]

        for i, (b_seq, o_seq) in enumerate(zip(batched_seqs, optimized_seqs, strict=False)):
            assert b_seq == o_seq, f"Beam {i}: Sequence mismatch.\nBatched: {b_seq}\nOptimized: {o_seq}"

    def test_multiple_targets_consistency(self, model_components):
        """Test that multiple single-batch calls to BatchedBeamSearch match individual calls."""
        model, rds, batched_beam, optimized_beam = model_components

        targets = [
            "CNCc1ccccc1",
            "CCOc1ccccc1",
        ]
        n_steps_list = [1, 1]

        from directmultistep.generate import prepare_input_tensors

        encoder_inputs = []
        steps_tensors = []
        path_starts_list = []

        for target, n_steps in zip(targets, n_steps_list, strict=False):
            encoder_inp, steps_tens, path_tens = prepare_input_tensors(
                target, n_steps, None, rds, rds.product_max_length, rds.sm_max_length
            )
            encoder_inputs.append(encoder_inp[0])
            steps_tensors.append(steps_tens[0] if steps_tens is not None else None)
            path_starts_list.append(path_tens[0])

        encoder_batch = torch.stack(encoder_inputs).to(batched_beam.device)
        steps_batch = torch.stack(steps_tensors).to(batched_beam.device) if steps_tensors[0] is not None else None
        path_starts_batch = [ps.to(batched_beam.device) for ps in path_starts_list]

        torch.manual_seed(42)
        batched_results = batched_beam.decode(
            src_BC=encoder_batch,
            steps_B1=steps_batch,
            path_starts=path_starts_batch,
            progress_bar=False,
        )

        for idx, (target, n_steps) in enumerate(zip(targets, n_steps_list, strict=False)):
            encoder_inp, steps_tens, path_tens = prepare_input_tensors(
                target, n_steps, None, rds, rds.product_max_length, rds.sm_max_length
            )
            
            encoder_inp = encoder_inp.to(optimized_beam.device)
            steps_tens = steps_tens.to(optimized_beam.device) if steps_tens is not None else None
            path_tens = path_tens.to(optimized_beam.device)

            torch.manual_seed(42)
            optimized_results = optimized_beam.decode(
                src_BC=encoder_inp,
                steps_B1=steps_tens,
                path_start_BL=path_tens,
                progress_bar=False,
            )

            batched_seqs = [seq for seq, _ in batched_results[idx]]
            optimized_seqs = [seq for seq, _ in optimized_results[0]]

            for beam_idx, (b_seq, o_seq) in enumerate(zip(batched_seqs, optimized_seqs, strict=False)):
                assert b_seq == o_seq, (
                    f"Target {idx}, Beam {beam_idx}: Sequence mismatch.\n"
                    f"Batched: {b_seq}\nOptimized: {o_seq}"
                )
