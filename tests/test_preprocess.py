import pytest

from directmultistep.utils.dataset import tokenize_path_string, tokenize_smile
from directmultistep.utils.pre_process import (
    filter_mol_nodes,
    find_leaves,
    generate_permutations,
    max_tree_depth,
)

from .test_data import (
    test1_leaves,
    test2_depth1,
    test3_depth2,
    test4_n1route0,
    test5_depth0_leaves,
    test6_depth1_leaves,
    test7_depth2_leaves,
    test8_n1route_leaves,
    test9_tknz_smiles,
    test10_tknz_path,
)

test_filtering_and_depth = [
    pytest.param(test1_leaves, 0, id="leaves"),
    pytest.param(test2_depth1, 1, id="depth1"),
    pytest.param(test3_depth2, 2, id="depth2"),
    pytest.param(test4_n1route0, 8, id="n1_routes_idx0"),
]


@pytest.mark.parametrize("data, _", test_filtering_and_depth)
def test_filter_mol_nodes(data, _):
    for item in data:
        assert filter_mol_nodes(item["paRoute"]) == item["filtered"]


def test_filter_mol_nodes_invalid_type():
    node = {
        "smiles": "COC(=O)c1ccc2c(c1)OCCO2",
        "children": [{"smiles": "BrCCBr"}, {"smiles": "COC(=O)c1ccc(O)c(O)c1"}],
        "type": "invalid",
    }
    with pytest.raises(ValueError) as exc_info:
        filter_mol_nodes(node)
    assert str(exc_info.value) == "Expected 'type' to be 'mol', got invalid"


@pytest.mark.parametrize("data, expected_depth", test_filtering_and_depth)
def test_max_tree_depth(data, expected_depth):
    for item in data:
        assert max_tree_depth(item["filtered"]) == expected_depth


test_leaves = [
    pytest.param(test5_depth0_leaves, id="depth0"),
    pytest.param(test6_depth1_leaves, id="depth1"),
    pytest.param(test7_depth2_leaves, id="depth2"),
    pytest.param(test8_n1route_leaves, id="n1route_idx0"),
]


@pytest.mark.parametrize("data", test_leaves)
def test_find_leaves(data):
    for item in data:
        assert find_leaves(item["filtered"]) == item["leaves"]


@pytest.mark.parametrize("data", test9_tknz_smiles)
def test_tokenize_smile(data):
    assert tokenize_smile(data[0]) == data[1]


@pytest.mark.parametrize("data", test10_tknz_path)
def test_tokenize_path(data):
    assert tokenize_path_string(data[0]) == data[1]


def test_generate_permutations_no_children():
    # Test data with no children
    data = {"smiles": "A"}
    assert generate_permutations(data) == [str(data).replace(" ", "")]


def test_generate_permutations_single_child():
    # Test data with one child
    data = {"smiles": "A", "children": [{"smiles": "B"}]}
    expected_output = [str(data).replace(" ", "")]
    assert generate_permutations(data) == expected_output


def test_generate_permutations_multiple_children():
    # Test data with multiple children
    data = {"smiles": "A", "children": [{"smiles": "B"}, {"smiles": "C"}]}
    expected_output = [
        str({"smiles": "A", "children": [{"smiles": "B"}, {"smiles": "C"}]}).replace(" ", ""),
        str({"smiles": "A", "children": [{"smiles": "C"}, {"smiles": "B"}]}).replace(" ", ""),
    ]
    assert sorted(generate_permutations(data)) == sorted(expected_output)


def test_generate_permutations_nested_children():
    # Test data with nested children
    data = {"smiles": "A", "children": [{"smiles": "B", "children": [{"smiles": "C"}]}]}
    expected_output = [
        str(
            {
                "smiles": "A",
                "children": [{"smiles": "B", "children": [{"smiles": "C"}]}],
            }
        ).replace(" ", "")
    ]
    assert generate_permutations(data) == expected_output


def test_generate_permutations_with_limit():
    # Test data with a permutation limit
    data = {
        "smiles": "A",
        "children": [{"smiles": "B"}, {"smiles": "C"}, {"smiles": "D"}],
    }
    # Limit to 2 permutations
    results = generate_permutations(data, max_perm=2)
    assert len(results) == 2


def test_generate_permutations_complex_case():
    # More complex structure with depth and multiple children at different levels
    data = {
        "smiles": "A",
        "children": [
            {"smiles": "B", "children": [{"smiles": "C"}, {"smiles": "D"}]},
            {"smiles": "E"},
        ],
    }
    results = generate_permutations(data)
    # Test that the correct number of permutations is generated (factorial of children count at each level)
    assert len(results) == 4  # Since there are 2 at top level (B with its children, and E)


@pytest.mark.parametrize(
    "data, expected",
    [
        ({"smiles": "X"}, [str({"smiles": "X"}).replace(" ", "")]),
        (
            {"smiles": "Y", "children": [{"smiles": "Z"}]},
            [str({"smiles": "Y", "children": [{"smiles": "Z"}]}).replace(" ", "")],
        ),
    ],
)
def test_generate_permutations_parametrized(data, expected):
    assert generate_permutations(data) == expected


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
