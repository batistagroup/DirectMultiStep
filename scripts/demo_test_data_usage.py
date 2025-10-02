"""
Example script demonstrating how to use the generated test data for beam search testing.
"""

import pickle
from pathlib import Path

from directmultistep.generate import create_beam_search, load_published_model
from directmultistep.utils.dataset import RoutesProcessing


def load_test_data():
    """Load the generated test data."""
    test_data_path = Path("tests/test_data/beam_search_comprehensive_test_data.pkl")

    if not test_data_path.exists():
        print(f"Test data not found at {test_data_path}")
        print("Please run: python scripts/save-data-for-tests.py")
        return None

    with open(test_data_path, "rb") as f:
        return pickle.load(f)


def demonstrate_beam_search_testing():
    """Demonstrate how to use the test data for testing beam search."""
    print("Loading test data...")
    test_data = load_test_data()

    if test_data is None:
        return

    # Load model and components
    print("Loading model and components...")
    config_path = Path("data/configs/dms_dictionary.yaml")
    ckpt_dir = Path("data/checkpoints")

    if not config_path.exists() or not ckpt_dir.exists():
        print("Model files not found. Please ensure data is downloaded.")
        return

    model = load_published_model("flash", ckpt_dir)
    rds = RoutesProcessing(metadata_path=config_path)
    beam_obj = create_beam_search(model, beam_size=5, rds=rds)

    # Test with the first case
    case_name = "target1"
    if case_name in test_data and test_data[case_name] is not None:
        print(f"\nTesting with {case_name}...")

        case_data = test_data[case_name]
        intermediate_data = case_data["intermediate_data"]
        expected_paths = case_data["final_paths"]

        print(f"Expected number of paths: {len(expected_paths)}")

        # Run beam search with the saved intermediate data
        results = beam_obj.decode(
            src_BC=intermediate_data["encoder_input"].to(beam_obj.device),
            steps_B1=intermediate_data["steps_tensor"].to(beam_obj.device)
            if intermediate_data["steps_tensor"] is not None
            else None,
            path_start_BL=intermediate_data["path_start_tensor"].to(beam_obj.device),
            progress_bar=False,
        )

        print(f"Generated {len(results[0])} beam results")

        # Show top results
        print("\nTop 3 generated paths:")
        for i, (path, log_prob) in enumerate(results[0][:3]):
            print(f"  {i + 1}. Log prob: {log_prob:.4f}")
            print(f"     Path: {path[:100]}...")  # Truncate for readability

        # Demonstrate step-by-step data usage
        step_data = case_data["beam_search_steps"]
        if step_data:
            print(f"\nStep-by-step data available for {len(step_data)} steps")

            # Show first step info
            first_step = step_data[0]
            print(f"First step decoder output shape: {first_step['decoder_output'].shape}")
            print(f"First step log probs shape: {first_step['log_probs'].shape}")

            # Verify that log probabilities are valid
            log_probs = first_step["log_probs"]
            print(f"Log prob range: [{log_probs.min():.4f}, {log_probs.max():.4f}]")
            print(f"All log probs <= 0: {(log_probs <= 0).all()}")

    print("\nDemonstration complete!")


def demonstrate_simple_test_data():
    """Demonstrate using the simple test data."""
    print("\n" + "=" * 50)
    print("Simple Test Data Demonstration")
    print("=" * 50)

    simple_data_path = Path("tests/test_data/beam_search_simple_test_data.pkl")
    if not simple_data_path.exists():
        print(f"Simple test data not found at {simple_data_path}")
        return

    with open(simple_data_path, "rb") as f:
        simple_data = pickle.load(f)

    print("Available test cases:")
    for case_name, case_data in simple_data.items():
        if case_data is not None:
            print(f"  - {case_name}: {len(case_data['final_paths'])} paths generated")
            print(f"    Target: {case_data['case_info']['target']}")
            print(f"    Starting material: {case_data['case_info']['starting_material']}")
            print(f"    Steps: {case_data['case_info']['n_steps']}")


if __name__ == "__main__":
    demonstrate_beam_search_testing()
    demonstrate_simple_test_data()
