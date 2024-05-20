# MIT License

# Copyright (c) 2024 Batista Lab (Yale University)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest
from rdkit import Chem
from Utils.PreProcess import filter_mol_nodes, max_tree_depth, find_leaves, generate_permutations
from test_data import *
from Data.Dataset import tokenize_smile, tokenize_path_string

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
    with pytest.raises(AssertionError) as exc_info:
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
    data = {'s': 'A'}
    assert generate_permutations(data, child_key="c") == [str(data).replace(" ", "")]

def test_generate_permutations_single_child():
    # Test data with one child
    data = {'s': 'A', 'c': [{'s': 'B'}]}
    expected_output = [str(data).replace(" ", "")]
    assert generate_permutations(data, child_key="c") == expected_output

def test_generate_permutations_multiple_children():
    # Test data with multiple children
    data = {'s': 'A', 'c': [{'s': 'B'}, {'s': 'C'}]}
    expected_output = [
        str({'s': 'A', 'c': [{'s': 'B'}, {'s': 'C'}]}).replace(" ", ""),
        str({'s': 'A', 'c': [{'s': 'C'}, {'s': 'B'}]}).replace(" ", "")
    ]
    assert sorted(generate_permutations(data, child_key="c")) == sorted(expected_output)

def test_generate_permutations_nested_children():
    # Test data with nested children
    data = {'s': 'A', 'c': [{'s': 'B', 'c': [{'s': 'C'}]}]}
    expected_output = [
        str({'s': 'A', 'c': [{'s': 'B', 'c': [{'s': 'C'}]}]}).replace(" ", "")
    ]
    assert generate_permutations(data, child_key="c") == expected_output

def test_generate_permutations_with_limit():
    # Test data with a permutation limit
    data = {'s': 'A', 'c': [{'s': 'B'}, {'s': 'C'}, {'s': 'D'}]}
    # Limit to 2 permutations
    results = generate_permutations(data, max_perm=2, child_key="c")
    assert len(results) == 2

def test_generate_permutations_complex_case():
    # More complex structure with depth and multiple children at different levels
    data = {'s': 'A', 'c': [{'s': 'B', 'c': [{'s': 'C'}, {'s': 'D'}]}, {'s': 'E'}]}
    results = generate_permutations(data, child_key="c")
    # Test that the correct number of permutations is generated (factorial of children count at each level)
    assert len(results) == 4  # Since there are 2 at top level (B with its children, and E)

@pytest.mark.parametrize("data, expected", [
    ({'s': 'X'}, [str({'s': 'X'}).replace(" ", "")]),
    ({'s': 'Y', 'c': [{'s': 'Z'}]}, [str({'s': 'Y', 'c': [{'s': 'Z'}]}).replace(" ", "")])
])
def test_generate_permutations_parametrized(data, expected):
    assert generate_permutations(data, child_key="c") == expected

