import itertools
from itertools import islice, permutations
from typing import TypedDict, cast

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PaRoutesDict = dict[str, str | bool | list["PaRoutesDict"]]


class FilteredDict(TypedDict, total=False):
    """A dictionary format for multistep routes, used in DirectMultiStep models.

    This dictionary is designed to represent a node in a synthetic route tree.
    It contains the SMILES string of a molecule and a list of its child nodes.
    To get its string format, use `stringify_dict`.

    Attributes:
        smiles: SMILES string of the molecule.
        children: List of child nodes, each a FilteredDict.
    """

    smiles: str
    children: list["FilteredDict"]


def filter_mol_nodes(node: PaRoutesDict) -> FilteredDict:
    """Filters a PaRoutes dictionary to keep only 'smiles' and 'children' keys.

    This function removes extra information like 'metadata', 'rsmi', and
    'reaction_hash', keeping only the 'smiles' and 'children' keys. It also
    canonicalizes the SMILES string using RDKit.

    Args:
        node: A dictionary representing a node in a PaRoutes data structure.

    Returns:
        A FilteredDict containing the canonicalized SMILES and filtered children.

    Raises:
        ValueError: If the 'type' of the node is not 'mol' or if 'children' is not a list.
    """
    # canonicalize smiles by passing through RDKit
    canonical_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(node["smiles"]))
    if "children" not in node:
        return {"smiles": canonical_smiles}
    if node.get("type") != "mol":
        raise ValueError(f"Expected 'type' to be 'mol', got {node.get('type', 'empty')}")

    filtered_node: FilteredDict = {"smiles": canonical_smiles, "children": []}
    # we skip one level of the PaRoutes dictionary as it contains the reaction meta data
    # assert isinstance(node["children"], list), f"Expected 'children' to be a list, got {type(node['children'])}"
    if not isinstance(node["children"], list):
        raise ValueError(f"Expected 'children' to be a list, got {type(node['children'])}")
    reaction_meta: list[PaRoutesDict] = node["children"]
    first_child = reaction_meta[0]
    for child in cast(list[PaRoutesDict], first_child["children"]):
        filtered_node["children"].append(filter_mol_nodes(child))
    return filtered_node


def max_tree_depth(node: FilteredDict) -> int:
    """Calculates the maximum depth of a synthetic route tree.

    Args:
        node: A FilteredDict representing a node in the route tree.

    Returns:
        The maximum depth of the tree. Returns 0 for a leaf node.
    """
    if "children" not in node:
        return 0  # Leaf node, depth is 0
    else:
        child_depths = [
            max_tree_depth(child)
            for child in node["children"]
            # if isinstance(child, dict)
        ]
        return 1 + max(child_depths)


def find_leaves(node: FilteredDict) -> list[str]:
    """Finds the SMILES strings of all leaf nodes (starting materials) in a route tree.

    Args:
        node: A FilteredDict representing a node in the route tree.

    Returns:
        A list of SMILES strings representing the starting materials.
    """
    leaves = []
    if "children" in node:
        for child in node["children"]:
            leaves.extend(find_leaves(child))
    else:
        leaves.append(node["smiles"])
    return leaves


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalizes a SMILES string using RDKit.

    Args:
        smiles: The SMILES string to canonicalize.

    Returns:
        The canonicalized SMILES string.

    Raises:
        ValueError: If the SMILES string cannot be parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    return cast(str, Chem.MolToSmiles(mol))


def stringify_dict(data: FilteredDict) -> str:
    """Converts a FilteredDict to a string, removing spaces.

    Args:
        data: The FilteredDict to convert.

    Returns:
        A string representation of the FilteredDict with no spaces.
    """
    return str(data).replace(" ", "")


def generate_permutations(data: FilteredDict, max_perm: int | None = None) -> list[str]:
    """Generates permutations of a synthetic route by permuting the order of children.

    This function generates all possible permutations of a synthetic route by
    rearranging the order of child nodes at each level of the tree. It can
    optionally limit the number of permutations generated.

    Args:
        data: A FilteredDict representing the synthetic route.
        max_perm: An optional integer to limit the number of permutations generated.

    Returns:
        A list of stringified FilteredDicts representing the permuted routes.
    """
    if "children" not in data or not data["children"]:
        return [stringify_dict(data)]

    child_permutations = []
    for child in data["children"]:
        child_permutations.append(generate_permutations(child, max_perm))

    all_combos = []
    # Conditionally apply permutation limit
    permutation_generator = permutations(range(len(child_permutations)))
    if max_perm is not None:
        permutation_generator = islice(permutation_generator, max_perm)  # type:ignore

    for combo in permutation_generator:
        for product in itertools.product(*(child_permutations[i] for i in combo)):
            new_data = data.copy()
            new_data["children"] = [eval(child_str) for child_str in product]
            all_combos.append(stringify_dict(new_data))
            if max_perm is not None and len(all_combos) >= max_perm:
                return all_combos  # Return early if maximum number of permutations is reached
    return all_combos


def is_convergent(route: FilteredDict) -> bool:
    """Determines if a synthesis route is convergent (non-linear).

    A route is linear if for every transformation, at most one reactant has children
    (i.e., all other reactants are leaf nodes). A route is convergent if there exists
    at least one transformation where two or more reactants have children.

    Args:
        route: The synthesis route to analyze.

    Returns:
        True if the route is convergent (non-linear), False if it's linear.
    """
    if "children" not in route:
        return False

    # Check if current node's transformation has 2 or more children with their own children
    children = route["children"]
    if len(children) >= 2:  # Need at least 2 children for a transformation
        children_with_children = sum(1 for child in children if "children" in child)
        if children_with_children >= 2:
            return True

    # Recursively check children
    return any(is_convergent(child) for child in children)
