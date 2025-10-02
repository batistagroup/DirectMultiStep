"""
Script to generate and save test data for beam search tests.
This script creates comprehensive test data including intermediate tensors,
logits, and beam search states for reproducible testing.
"""

import pickle
from pathlib import Path

import numpy as np
import torch

from directmultistep.generate import create_beam_search, generate_routes, load_published_model, prepare_input_tensors
from directmultistep.model import ModelFactory
from directmultistep.utils.dataset import RoutesProcessing

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the test cases
TEST_CASES = [
    {
        "name": "target1",
        "target": "CNCc1cc(-c2ccccc2F)n(S(=O)(=O)c2cccnc2)c1",
        "starting_material": "CN",
        "n_steps": 2,
    },
    {
        "name": "target2",
        "target": "O=C(c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1)N1CCN(CC2CC2)CC1",
        "starting_material": "CCOC(=O)c1ccc(N)cc1",
        "n_steps": 5,
    },
]


class BeamSearchTestDataGenerator:
    def __init__(self, model_name="flash", beam_size=5, device=None):
        self.model_name = model_name
        self.beam_size = beam_size
        self.device = device or ModelFactory.determine_device()
        self.config_path = Path("data/configs/dms_dictionary.yaml")
        self.ckpt_dir = Path("data/checkpoints")

    def load_model_and_components(self):
        """Load model and create beam search components."""
        model = load_published_model(self.model_name, self.ckpt_dir)
        rds = RoutesProcessing(metadata_path=self.config_path)
        beam_obj = create_beam_search(model, self.beam_size, rds)
        return model, rds, beam_obj

    def generate_intermediate_data(self, target, n_steps, starting_material, model, rds, beam_obj):
        """Generate intermediate data for beam search testing."""
        # Prepare input tensors
        encoder_inp, steps_tens, path_tens = prepare_input_tensors(
            target, n_steps, starting_material, rds, rds.product_max_length, rds.sm_max_length
        )

        # Move tensors to device
        encoder_inp = encoder_inp.to(self.device)
        steps_tens = steps_tens.to(self.device) if steps_tens is not None else None
        path_tens = path_tens.to(self.device)

        # Get encoder outputs
        src_mask_B11C = (encoder_inp != beam_obj.pad_idx).unsqueeze(1).unsqueeze(2)
        with torch.no_grad():
            enc_src_BCD = model.encoder(encoder_inp.long(), src_mask_B11C, steps_tens)

        # Store intermediate data
        intermediate_data = {
            "encoder_input": encoder_inp.cpu(),
            "steps_tensor": steps_tens.cpu() if steps_tens is not None else None,
            "path_start_tensor": path_tens.cpu(),
            "encoder_output": enc_src_BCD.cpu(),
            "src_mask": src_mask_B11C.cpu(),
            "target": target,
            "starting_material": starting_material,
            "n_steps": n_steps,
        }

        return intermediate_data

    def generate_beam_search_steps(self, intermediate_data, model, beam_obj):
        """Generate step-by-step beam search data for testing."""
        B, C = intermediate_data["encoder_input"].shape
        S = self.beam_size
        L = beam_obj.max_length

        # Reconstruct tensors on device
        src_BC = intermediate_data["encoder_input"].to(self.device)
        steps_B1 = (
            intermediate_data["steps_tensor"].to(self.device) if intermediate_data["steps_tensor"] is not None else None
        )
        path_start_BL = intermediate_data["path_start_tensor"].to(self.device)
        enc_src_BCD = intermediate_data["encoder_output"].to(self.device)
        src_mask_B11C = intermediate_data["src_mask"].to(self.device)

        # Initialize beam search state
        beam_enc_WCD = enc_src_BCD.repeat_interleave(S, dim=0)
        beam_src_WC = src_BC.repeat_interleave(S, dim=0)
        beam_src_mask_W11C = (beam_src_WC != beam_obj.pad_idx).unsqueeze(1).unsqueeze(2)

        beam_idxs_WL = torch.full((B * S, L), beam_obj.pad_idx, dtype=torch.long, device=self.device)
        if path_start_BL is None:
            beam_idxs_WL[:, 0] = beam_obj.start_idx
            first_step = 1
            beam_log_probs_W = torch.zeros(B * S, device=self.device)
        else:
            beam_idxs_WL[:, : path_start_BL.size(1)] = path_start_BL
            first_step = path_start_BL.size(1)
            beam_log_probs_W = torch.zeros(B * S, device=self.device)

        finished_sequences_W = torch.zeros(B * S, dtype=torch.bool, device=self.device)

        # Store step-by-step data
        step_data = []

        for step in range(first_step, min(first_step + 10, L - 1)):  # Limit steps for testing
            with torch.no_grad():
                output_WLV = model.decoder(
                    trg_BL=beam_idxs_WL[:, :step],
                    enc_src_BCD=beam_enc_WCD,
                    src_mask_B11C=beam_src_mask_W11C,
                    trg_mask_B1LL=None,
                )

            output_WV = output_WLV[:, -1, :]
            log_probs_WV = torch.log_softmax(output_WV, dim=-1)

            finished_sequences_W = torch.any(beam_idxs_WL == beam_obj.end_idx, dim=-1)
            active_mask_W = ~finished_sequences_W

            step_info = {
                "step": step,
                "decoder_output": output_WV.cpu(),
                "log_probs": log_probs_WV.cpu(),
                "beam_indices": beam_idxs_WL.cpu(),
                "beam_log_probs": beam_log_probs_W.cpu(),
                "finished_sequences": finished_sequences_W.cpu(),
                "active_mask": active_mask_W.cpu(),
            }

            # Simple beam update for testing (first beam only)
            if step == first_step:
                log_probs_BSV = log_probs_WV.view(B, S, -1)
                log_probs_WS, top_k_idxs_WS = torch.topk(log_probs_BSV[:, 0, :], S, dim=-1)
                beam_log_probs_W = log_probs_WS.view(B * S)
                beam_idxs_WL[:, step] = top_k_idxs_WS.view(B * S)
            else:
                # Simplified update - just take top tokens for first beam
                log_probs_BSV = log_probs_WV.view(B, S, -1)
                top_log_probs, top_indices = torch.topk(log_probs_BSV[:, 0, :], S, dim=-1)
                beam_idxs_WL[:, step] = top_indices.view(B * S)
                beam_log_probs_W = top_log_probs.view(B * S)

            step_data.append(step_info)

            if finished_sequences_W.all():
                break

        return step_data


def main():
    # Output files for the test data
    output_dir = Path("tests/test_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = BeamSearchTestDataGenerator(model_name="flash", beam_size=5)

    print("Loading model and components...")
    model, rds, beam_obj = generator.load_model_and_components()

    test_data = {}

    for case in TEST_CASES:
        print(f"Generating test data for {case['name']}...")
        try:
            # Generate intermediate data
            intermediate_data = generator.generate_intermediate_data(
                target=case["target"],
                n_steps=case["n_steps"],
                starting_material=case["starting_material"],
                model=model,
                rds=rds,
                beam_obj=beam_obj,
            )

            # Generate beam search steps
            step_data = generator.generate_beam_search_steps(intermediate_data, model, beam_obj)

            # Generate final routes using the standard function
            paths = generate_routes(
                target=case["target"],
                n_steps=case["n_steps"],
                starting_material=case["starting_material"],
                model=model,
                beam_size=5,
                config_path=generator.config_path,
                ckpt_dir=generator.ckpt_dir,
            )

            test_data[case["name"]] = {
                "intermediate_data": intermediate_data,
                "beam_search_steps": step_data,
                "final_paths": paths,
                "case_info": case,
            }

            print(f"Generated {len(paths)} paths and {len(step_data)} beam search steps for {case['name']}")

        except Exception as e:
            print(f"Error generating test data for {case['name']}: {e}")
            test_data[case["name"]] = None

    # Save comprehensive test data
    comprehensive_file = output_dir / "beam_search_comprehensive_test_data.pkl"
    with open(comprehensive_file, "wb") as f:
        pickle.dump(test_data, f)
    print(f"Comprehensive test data saved to {comprehensive_file}")

    # Also save a simplified version for basic testing
    simple_test_data = {}
    for name, data in test_data.items():
        if data is not None:
            simple_test_data[name] = {"final_paths": data["final_paths"], "case_info": data["case_info"]}

    simple_file = output_dir / "beam_search_simple_test_data.pkl"
    with open(simple_file, "wb") as f:
        pickle.dump(simple_test_data, f)
    print(f"Simple test data saved to {simple_file}")


if __name__ == "__main__":
    main()
