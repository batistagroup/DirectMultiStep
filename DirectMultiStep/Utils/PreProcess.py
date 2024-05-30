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

from rdkit import Chem  # type: ignore
from typing import Dict, List, Set, Union, cast, Optional
import itertools
from itertools import permutations, islice

PaRoutesDict = Dict[str, Union[str, bool, List["PaRoutesDict"]]]
FilteredDict = Dict[str, Union[str, List["FilteredDict"]]]


def filter_mol_nodes(node: PaRoutesDict) -> FilteredDict:
    """
    Remove information like 'metadata', 'rsmi', 'reaction_hash', etc.
    keep only 'smiles' and 'children' keys in the PaRoutes Data dictionary/json.
    An example of our data look like:
    {'smiles': 'COC(=O)c1cc2c(cc1[N+](=O)[O-])OCCO2',
     'children': [{'smiles': 'COC(=O)c1ccc2c(c1)OCCO2',
     'children': [{'smiles': 'BrCCBr'}, {'smiles': 'COC(=O)c1ccc(O)c(O)c1'}]}, {'smiles': 'O=[N+]([O-])O'}]}
     This dictionary will be in string format and can get the dictionary again by calling ```eval(string)```
    """
    # canonicalize smiles by passing through RDKit
    canonical_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(node["smiles"]))
    if "children" not in node:
        return {"smiles": canonical_smiles}
    assert (
        node.get("type") == "mol"
    ), f"Expected 'type' to be 'mol', got {node.get('type')}"
    filtered_node = {"smiles": canonical_smiles, "children": []}
    # we skip one level of the PaRoutes dictionary as it contains the reaction meta data
    assert isinstance(node["children"], list), "Expected 'children' to be a list"
    reaction_meta: List[PaRoutesDict] = node["children"]
    first_child = reaction_meta[0]
    for child in cast(List[PaRoutesDict], first_child["children"]):
        filtered_node["children"].append(filter_mol_nodes(child))
    return filtered_node


def max_tree_depth(node: FilteredDict) -> int:
    """
    Get the max step of the tree.
    """
    if "children" not in node:
        return 0  # Leaf node, depth is 0
    else:
        child_depths = [
            max_tree_depth(child)
            for child in node["children"]
            if isinstance(child, dict)
        ]
        return 1 + max(child_depths)


def find_leaves(node: FilteredDict) -> List[str]:
    """
    Get the starting materials SMILES (which are the SMILES of leave nodes).
    """
    leaves = []
    if "children" in node:
        for child in node["children"]:
            leaves.extend(find_leaves(cast(FilteredDict, child)))
    else:
        leaves.append(cast(str, node["smiles"]))
    return leaves


def canonicalize_smiles(smiles: str) -> str:
    """
    Canonicalize the SMILES using RDKit.
    """
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def stringify_dict(data: FilteredDict) -> str:
    return str(data).replace(" ", "")


def generate_permutations(
    data: FilteredDict, max_perm: Optional[int] = None, child_key: str = "children"
) -> List[str]:
    if child_key not in data or not data[child_key]:
        return [stringify_dict(data)]

    child_permutations = []
    for child in data[child_key]:
        child_permutations.append(
            generate_permutations(cast(FilteredDict, child), max_perm, child_key)
        )

    all_combos = []
    # Conditionally apply permutation limit
    permutation_generator = permutations(range(len(child_permutations)))
    if max_perm is not None:
        permutation_generator = islice(permutation_generator, max_perm)  # type:ignore

    for combo in permutation_generator:
        for product in itertools.product(*(child_permutations[i] for i in combo)):
            new_data = data.copy()
            new_data[child_key] = [eval(child_str) for child_str in product]
            all_combos.append(stringify_dict(new_data))
            if max_perm is not None and len(all_combos) >= max_perm:
                return all_combos  # Return early if maximum number of permutations is reached
    return all_combos


def load_commercial_stock(path: str) -> Set[str]:
    with open(path, "r") as file:
        stock = file.readlines()
    canonical_stock = set()
    for molecule in stock:
        canonical_stock.add(canonicalize_smiles(molecule.strip()))
    print(f"Loaded {len(canonical_stock)} molecules from {path}")
    return canonical_stock
