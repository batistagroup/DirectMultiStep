"""
Tests for BatchedBeamSearch with variable batch sizes and lengths.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from directmultistep.generation.tensor_gen import BatchedBeamSearch, BeamSearchOptimized, VectorizedBatchedBeamSearch
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
        vec_beam = VectorizedBatchedBeamSearch(
            model=model,
            beam_size=5,
            start_idx=0,
            pad_idx=52,
            end_idx=22,
            max_length=1074,
            idx_to_token=rds.idx_to_token,
            device=device,
        )

        return model, rds, batched_beam, optimized_beam, vec_beam

    def test_single_batch_equivalence(self, model_components):
        """Test that BatchedBeamSearch gives same results as BeamSearchOptimized for single batch."""
        model, rds, batched_beam, optimized_beam, vec_beam = model_components

        target = "CNCc1ccccc1"
        starting_material = None
        n_steps = 1

        from directmultistep.generate import prepare_input_tensors

        encoder_inp, steps_tens, path_tens = prepare_input_tensors(
            target, n_steps, starting_material, rds, rds.product_max_length, rds.sm_max_length
        )

        encoder_inp = encoder_inp.to(batched_beam.device)
        steps_tens = steps_tens.to(batched_beam.device) if steps_tens is not None else None
        path_tens = path_tens.to(batched_beam.device)

        torch.manual_seed(42)
        optimized_results = optimized_beam.decode(
            src_BC=encoder_inp,
            steps_B1=steps_tens,
            path_start_BL=path_tens,
            progress_bar=True,
        )

        torch.manual_seed(42)
        batched_results = batched_beam.decode(
            src_BC=encoder_inp,
            steps_B1=steps_tens,
            path_starts=[path_tens[0]],
            progress_bar=True,
        )
        vec_results = vec_beam.decode(
            src_BC=encoder_inp,
            steps_B1=steps_tens,
            path_starts=[path_tens[0]],
            progress_bar=True,
        )

        assert len(batched_results) == 1
        assert len(optimized_results) == 1
        assert len(vec_results) == 1

        batched_seqs = [seq for seq, _ in batched_results[0]]
        optimized_seqs = [seq for seq, _ in optimized_results[0]]
        vec_seqs = [seq for seq, _ in vec_results[0]]

        batched_probs = [prob for _, prob in batched_results[0]]
        optimized_probs = [prob for _, prob in optimized_results[0]]
        vec_probs = [prob for _, prob in vec_results[0]]

        for i, (b_seq, o_seq, v_seq) in enumerate(zip(batched_seqs, optimized_seqs, vec_seqs, strict=False)):
            assert b_seq == o_seq == v_seq, (
                f"Beam {i}: Sequence mismatch.\nBatched: {b_seq}\nOptimized: {o_seq}\nVectorized: {v_seq}"
            )

        for i, (b_prob, o_prob, v_prob) in enumerate(zip(batched_probs, optimized_probs, vec_probs, strict=False)):
            assert abs(b_prob - o_prob) < 1e-5 and abs(b_prob - v_prob) < 1e-5, (
                f"Beam {i}: Log prob mismatch. Batched: {b_prob:.6f}, Optimized: {o_prob:.6f}, Vectorized: {v_prob:.6f}"
            )

    def test_single_batch_equivalence_with_sm(self, model_components):
        """Test equivalence when starting material is provided."""
        model, rds, batched_beam, optimized_beam, vec_beam = model_components

        target = "CNCc1ccccc1"
        starting_material = "CN"
        n_steps = 1

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

        torch.manual_seed(42)
        vec_results = vec_beam.decode(
            src_BC=encoder_inp,
            steps_B1=steps_tens,
            path_starts=[path_tens[0]],
            progress_bar=False,
        )

        batched_seqs = [seq for seq, _ in batched_results[0]]
        optimized_seqs = [seq for seq, _ in optimized_results[0]]
        vec_seqs = [seq for seq, _ in vec_results[0]]

        batched_probs = [prob for _, prob in batched_results[0]]
        optimized_probs = [prob for _, prob in optimized_results[0]]
        vec_probs = [prob for _, prob in vec_results[0]]

        for i, (b_seq, o_seq, v_seq) in enumerate(zip(batched_seqs, optimized_seqs, vec_seqs, strict=False)):
            assert b_seq == o_seq == v_seq, (
                f"Beam {i}: Sequence mismatch.\nBatched: {b_seq}\nOptimized: {o_seq}\nVectorized: {v_seq}"
            )

        for i, (b_prob, o_prob, v_prob) in enumerate(zip(batched_probs, optimized_probs, vec_probs, strict=False)):
            assert abs(b_prob - o_prob) < 1e-5 and abs(b_prob - v_prob) < 1e-5, (
                f"Beam {i}: Log prob mismatch. Batched: {b_prob:.6f}, Optimized: {o_prob:.6f}, Vectorized: {v_prob:.6f}"
            )

    def test_multiple_batches_independently_correct(self, model_components):
        """Test that batched decoding produces correct results for each batch item independently."""
        model, rds, batched_beam, optimized_beam, vec_beam = model_components

        targets = ["CNCc1ccccc1", "CC(=O)OC1=CC=CC=C1C(=O)O"]
        n_steps_list = [1, 1]

        from directmultistep.generate import prepare_batched_input_tensors

        encoder_batch, steps_batch, path_starts, target_lengths = prepare_batched_input_tensors(
            targets=targets,
            n_steps_list=n_steps_list,
            starting_materials=[None, None],
            rds=rds,
            product_max_length=rds.product_max_length,
            sm_max_length=rds.sm_max_length,
        )

        torch.manual_seed(42)
        batched_results = batched_beam.decode(
            src_BC=encoder_batch.to(batched_beam.device),
            steps_B1=steps_batch.to(batched_beam.device) if steps_batch is not None else None,
            path_starts=[ps.to(batched_beam.device) for ps in path_starts],
            progress_bar=True,
        )

        torch.manual_seed(42)
        vec_results = vec_beam.decode(
            src_BC=encoder_batch.to(vec_beam.device),
            steps_B1=steps_batch.to(vec_beam.device) if steps_batch is not None else None,
            path_starts=[ps.to(vec_beam.device) for ps in path_starts],
            progress_bar=True,
        )

        from directmultistep.generate import prepare_input_tensors

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
                progress_bar=True,
            )

            batched_seqs = [seq for seq, _ in batched_results[idx]]
            vec_seqs = [seq for seq, _ in vec_results[idx]]
            optimized_seqs = [seq for seq, _ in optimized_results[0]]

            for beam_idx, (b_seq, v_seq, o_seq) in enumerate(zip(batched_seqs, vec_seqs, optimized_seqs, strict=False)):
                assert b_seq == v_seq == o_seq, (
                    f"Target {idx} ('{target}'), Beam {beam_idx}: Sequence mismatch.\n"
                    f"Batched: {b_seq}\nVectorized: {v_seq}\nOptimized: {o_seq}"
                )

    def test_multiple_batches_independently_correct_hard(self, model_components):
        """Test that batched decoding produces correct results for each batch item independently."""
        model, rds, batched_beam, optimized_beam, vec_beam = model_components

        targets_list = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            "CNCc1cc(-c2ccccc2F)n(S(=O)(=O)c2cccnc2)c1",
            "O=C(c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1)N1CCN(CC2CC2)CC1",
            "COc1ccc(-n2nccn2)c(C(=O)N2CCC[C@@]2(C)c2nc3c(C)c(Cl)ccc3[nH]2)c1",
        ]
        sms_list = [None, "O=S(=O)(Cl)c1cccnc1", "CCOC(=O)c1ccc(N)cc1", "C[C@@]1(C(=O)O)CCCN1"]
        n_steps_list = [1, 2, 5, 4]

        from directmultistep.generate import prepare_batched_input_tensors

        encoder_batch, steps_batch, path_starts, target_lengths = prepare_batched_input_tensors(
            targets=targets_list,
            n_steps_list=n_steps_list,
            starting_materials=sms_list,
            rds=rds,
            product_max_length=rds.product_max_length,
            sm_max_length=rds.sm_max_length,
        )

        torch.manual_seed(42)
        batched_results = batched_beam.decode(
            src_BC=encoder_batch.to(batched_beam.device),
            steps_B1=steps_batch.to(batched_beam.device) if steps_batch is not None else None,
            path_starts=[ps.to(batched_beam.device) for ps in path_starts],
            progress_bar=True,
        )

        torch.manual_seed(42)
        vec_results = vec_beam.decode(
            src_BC=encoder_batch.to(vec_beam.device),
            steps_B1=steps_batch.to(vec_beam.device) if steps_batch is not None else None,
            path_starts=[ps.to(vec_beam.device) for ps in path_starts],
            progress_bar=True,
        )

        from directmultistep.generate import prepare_input_tensors

        for idx, (target, sm, n_steps) in enumerate(zip(targets_list, sms_list, n_steps_list, strict=False)):
            encoder_inp, steps_tens, path_tens = prepare_input_tensors(
                target, n_steps, sm, rds, rds.product_max_length, rds.sm_max_length
            )

            encoder_inp = encoder_inp.to(optimized_beam.device)
            steps_tens = steps_tens.to(optimized_beam.device) if steps_tens is not None else None
            path_tens = path_tens.to(optimized_beam.device)

            torch.manual_seed(42)
            optimized_results = optimized_beam.decode(
                src_BC=encoder_inp,
                steps_B1=steps_tens,
                path_start_BL=path_tens,
                progress_bar=True,
            )

            batched_seqs = [seq for seq, _ in batched_results[idx]]
            vec_seqs = [seq for seq, _ in vec_results[idx]]
            optimized_seqs = [seq for seq, _ in optimized_results[0]]

            for beam_idx, (b_seq, v_seq, o_seq) in enumerate(zip(batched_seqs, vec_seqs, optimized_seqs, strict=False)):
                assert b_seq == v_seq == o_seq, (
                    f"Target {idx} ('{target}'), Beam {beam_idx}: Sequence mismatch.\n"
                    f"Batched: {b_seq}\nVector: {v_seq}\nOptimized: {o_seq}"
                )

    def test_multiple_batches_independently_correct_hard_beam20(self, model_components):
        """Test that batched decoding produces correct results for each batch item independently."""
        model, rds, batched_beam, optimized_beam, vec_beam = model_components

        targets_list = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            "CNCc1cc(-c2ccccc2F)n(S(=O)(=O)c2cccnc2)c1",
            "O=C(c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1)N1CCN(CC2CC2)CC1",
            "COc1ccc(-n2nccn2)c(C(=O)N2CCC[C@@]2(C)c2nc3c(C)c(Cl)ccc3[nH]2)c1",
        ]
        sms_list = [None, "O=S(=O)(Cl)c1cccnc1", "CCOC(=O)c1ccc(N)cc1", "C[C@@]1(C(=O)O)CCCN1"]
        n_steps_list = [1, 2, 5, 4]

        from directmultistep.generate import prepare_batched_input_tensors

        encoder_batch, steps_batch, path_starts, target_lengths = prepare_batched_input_tensors(
            targets=targets_list,
            n_steps_list=n_steps_list,
            starting_materials=sms_list,
            rds=rds,
            product_max_length=rds.product_max_length,
            sm_max_length=rds.sm_max_length,
        )

        torch.manual_seed(42)
        vec_beam.beam_size = 20
        vec_results = vec_beam.decode(
            src_BC=encoder_batch.to(vec_beam.device),
            steps_B1=steps_batch.to(vec_beam.device) if steps_batch is not None else None,
            path_starts=[ps.to(vec_beam.device) for ps in path_starts],
            progress_bar=True,
        )

        from directmultistep.generate import prepare_input_tensors

        for idx, (target, sm, n_steps) in enumerate(zip(targets_list, sms_list, n_steps_list, strict=False)):
            encoder_inp, steps_tens, path_tens = prepare_input_tensors(
                target, n_steps, sm, rds, rds.product_max_length, rds.sm_max_length
            )

            encoder_inp = encoder_inp.to(optimized_beam.device)
            steps_tens = steps_tens.to(optimized_beam.device) if steps_tens is not None else None
            path_tens = path_tens.to(optimized_beam.device)

            torch.manual_seed(42)
            optimized_beam.beam_size = 20
            optimized_results = optimized_beam.decode(
                src_BC=encoder_inp,
                steps_B1=steps_tens,
                path_start_BL=path_tens,
                progress_bar=True,
            )

            vec_seqs = [seq for seq, _ in vec_results[idx]]
            optimized_seqs = [seq for seq, _ in optimized_results[0]]

            for beam_idx, (v_seq, o_seq) in enumerate(zip(vec_seqs, optimized_seqs, strict=False)):
                assert v_seq == o_seq, (
                    f"Target {idx} ('{target}'), Beam {beam_idx}: Sequence mismatch.\n"
                    f"Vector: {v_seq}\nOptimized: {o_seq}"
                )
