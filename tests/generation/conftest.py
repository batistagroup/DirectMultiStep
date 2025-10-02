"""
Pytest configuration and fixtures for generation tests.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

# Set seeds for reproducible tests
torch.manual_seed(42)
np.random.seed(42)

# Test cases for beam search testing
TEST_CASES = [
    {
        "name": "target1",
        "target": "CNCc1cc(-c2ccccc2F)n(S(=O)(=O)c2cccnc2)c1",
        "starting_material": "CN",
        "n_steps": 2,
        "description": "Simple target with primary amine starting material",
    },
    {
        "name": "target2",
        "target": "O=C(c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1)N1CCN(CC2CC2)CC1",
        "starting_material": "CCOC(=O)c1ccc(N)cc1",
        "n_steps": 5,
        "description": "Complex target with multiple functional groups",
    },
    {
        "name": "target3",
        "target": "CC(C)(C)[C@@H](CS(C)(=O)=O)Nc1nc(-c2c[nH]c3ncc(F)cc23)ncc1F",
        "starting_material": "CC(C)(C)[C@H](N)CO",
        "n_steps": 3,
        "description": "Complex pharmaceutical-like molecule",
    },
]

SIMPLE_SMILES_CASES = [
    {"name": "simple_alkane", "smiles": "CCCC", "description": "Simple butane molecule"},
    {"name": "simple_arene", "smiles": "c1ccccc1", "description": "Benzene ring"},
    {"name": "simple_functional", "smiles": "CC(=O)O", "description": "Acetic acid"},
]


@pytest.fixture(scope="session")
def test_cases():
    """Return the standard test cases for generation testing."""
    return TEST_CASES


@pytest.fixture(scope="session")
def simple_smiles_cases():
    """Return simple SMILES cases for basic testing."""
    return SIMPLE_SMILES_CASES


@pytest.fixture
def model_files_available():
    """Check if model files are available for testing."""
    config_path = Path("data/configs/dms_dictionary.yaml")
    ckpt_dir = Path("data/checkpoints")

    return config_path.exists() and ckpt_dir.exists() and any(ckpt_dir.iterdir())


@pytest.fixture
def test_data_dir():
    """Return the test data directory path."""
    return Path("tests/test_data")


@pytest.fixture
def require_test_data(test_data_dir):
    """Skip test if test data is not available."""
    simple_data_file = test_data_dir / "beam_search_simple_test_data.pkl"
    comprehensive_data_file = test_data_dir / "beam_search_comprehensive_test_data.pkl"

    if not simple_data_file.exists() or not comprehensive_data_file.exists():
        pytest.skip("Test data not found. Run scripts/save-data-for-tests.py to generate.")

    return {"simple": simple_data_file, "comprehensive": comprehensive_data_file}


@pytest.fixture
def reproducible_seed():
    """Set reproducible seed for tests that need it."""

    def _set_seed(seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        return seed

    return _set_seed


# Pytest configuration
def pytest_configure(config):
    """Configure pytest for generation tests."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "requires_model: marks tests that require model files")
    config.addinivalue_line("markers", "requires_test_data: marks tests that require generated test data")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on fixtures used."""
    for item in items:
        # Add requires_model marker if test uses model_files_available fixture
        if "model_files_available" in item.fixturenames:
            item.add_marker(pytest.mark.requires_model)

        # Add requires_test_data marker if test uses require_test_data fixture
        if "require_test_data" in item.fixturenames:
            item.add_marker(pytest.mark.requires_test_data)
