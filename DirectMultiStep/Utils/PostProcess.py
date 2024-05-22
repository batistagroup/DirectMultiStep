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

from Models.Generation import BeamSearchOutput
from Utils.PreProcess import (
    canonicalize_smiles,
    find_leaves,
    generate_permutations,
    FilteredDict,
    stringify_dict,
    max_tree_depth,
)
from typing import List, Tuple, Dict, Set, Optional
import json
from tqdm import tqdm

BeamResultType = List[BeamSearchOutput]
PathReacType = Tuple[str, List[str]]
BeamProcessedType = List[PathReacType]
PathsProcessedType = List[BeamProcessedType]
MatchList = List[int | None]


def find_valid_paths(
    beam_results_NS2: BeamResultType, verbose: bool = False
) -> PathsProcessedType:
    if verbose:
        print("Starting to find valid paths:")
    valid_pathreac_NS2n = []
    iterator = tqdm(beam_results_NS2) if verbose else beam_results_NS2
    for beam_result_S2 in iterator:
        valid_pathreac_S2n = []
        for path_string, score in beam_result_S2:
            try:
                node = eval(path_string)
                reactants = find_leaves(node)
                canon_reactants = [
                    canonicalize_smiles(reactant) for reactant in reactants
                ]
                canon_path = canonicalize_path_string(path_string)
            except:
                continue
            valid_pathreac_S2n.append((canon_path, canon_reactants))
        valid_pathreac_NS2n.append(valid_pathreac_S2n)
    # from now on, NS2n means there is a List with N lists of S lists of Tuples in which first element
    # is a path string and second element is a list of canonicalized reactants
    return valid_pathreac_NS2n


def find_matching_paths(
    paths_NS2n: PathsProcessedType, correct_paths: List[str], verbose: bool = False
) -> Tuple[MatchList]:
    if verbose:
        print("Starting to find matching paths:")
    match_accuracy_N: MatchList = []
    perm_match_accuracy_N: MatchList = []
    iterator = (
        tqdm(zip(paths_NS2n, correct_paths), total=len(paths_NS2n))
        if verbose
        else zip(paths_NS2n, correct_paths)
    )
    for pathreac_S2n, correct_path in iterator:
        path_match = None
        path_match_perm = None
        for rank, (path, _) in enumerate(pathreac_S2n):
            if path_match is None and path == correct_path:
                path_match = rank + 1
            if path_match_perm is None:
                all_perms = generate_permutations(data=eval(path), max_perm=None)
                if correct_path in all_perms:
                    path_match_perm = rank + 1
            if path_match and path_match_perm:
                break
        match_accuracy_N.append(path_match)
        perm_match_accuracy_N.append(path_match_perm)
    return match_accuracy_N, perm_match_accuracy_N


def find_top_n_accuracy(
    match_accuracy: List[int], n_vals: List[int], dec_digs: int = 1
) -> Dict[str, float]:
    n_vals = sorted(n_vals)
    top_counts = {f"Top {n}": 0 for n in n_vals}
    for rank in match_accuracy:
        if rank is None:
            continue
        for n in n_vals:
            if rank <= n:
                top_counts[f"Top {n}"] += 1
    top_fractions = {
        k: f"{(v / len(match_accuracy)* 100):.{dec_digs}f}"
        for k, v in top_counts.items()
    }
    return top_fractions


def remove_repetitions_within_beam_result(
    paths_NS2n: PathsProcessedType, verbose: bool = False
) -> PathsProcessedType:
    if verbose:
        print("Starting to remove repetitions within beam results:")
    unique_paths_NS2n = []
    iterator = tqdm(paths_NS2n) if verbose else paths_NS2n
    for path_reac_S2 in iterator:
        unique_paths_S2n = []
        seen = set()
        for path, reacs_n in path_reac_S2:
            for permuted_pathstring in generate_permutations(
                data=eval(path), max_perm=None
            ):
                if permuted_pathstring in seen:
                    break
            else:
                seen.add(path)
                unique_paths_S2n.append((path, reacs_n))
        unique_paths_NS2n.append(unique_paths_S2n)
    return unique_paths_NS2n


def find_paths_with_commercial_sm(
    paths_NS2n: PathsProcessedType, commercial_stock: Set[str], verbose: bool = False
) -> PathsProcessedType:
    if verbose:
        print("Starting to find paths with commercial reactants:")
    available_paths_NS2n = []
    iterator = tqdm(paths_NS2n) if verbose else paths_NS2n
    for path_reac_S2 in iterator:
        available_paths_S2n = []
        for path, reacs_n in path_reac_S2:
            if all(reactant in commercial_stock for reactant in reacs_n):
                available_paths_S2n.append((path, reacs_n))
        available_paths_NS2n.append(available_paths_S2n)
    return available_paths_NS2n


def find_paths_with_correct_product_and_reactants(
    paths_NS2n: PathsProcessedType,
    true_products: List[str],
    true_reacs: Optional[List[str]]=None,
    verbose: bool = False,
) -> PathsProcessedType:
    if verbose:
        print("Starting to find paths with correct product and reactants:")
    f = canonicalize_smiles
    correct_paths_NS2n = []
    iterator = tqdm(enumerate(paths_NS2n)) if verbose else enumerate(paths_NS2n)
    for idx, path_reac_S2 in iterator:
        correct_paths_S2n = []
        for path, reacs_n in path_reac_S2:
            path_tree = eval(path)
            if (
                f(path_tree["smiles"]) == f(true_products[idx])
                and (true_reacs is None or f(true_reacs[idx]) in reacs_n)
            ):
                correct_paths_S2n.append((path, reacs_n))
        correct_paths_NS2n.append(correct_paths_S2n)
    return correct_paths_NS2n


def canonicalize_path_dict(path_dict: FilteredDict) -> FilteredDict:
    canon_dict: FilteredDict = {}
    canon_dict["smiles"] = canonicalize_smiles(path_dict["smiles"])
    if "children" in path_dict:
        canon_dict["children"] = []
        for child in path_dict["children"]:
            canon_dict["children"].append(canonicalize_path_dict(child))
    return canon_dict


def canonicalize_path_string(path_string: str) -> str:
    canon_dict = canonicalize_path_dict(eval(path_string))
    return stringify_dict(canon_dict)


def canonicalize_paths(paths_NS2n: PathsProcessedType, verbose:bool=False) -> PathsProcessedType:
    if verbose:
        print("Starting to canonicalize paths:")
    canon_paths_NS2n = []
    counter = 0
    iterator = tqdm(paths_NS2n) if verbose else paths_NS2n
    for path_reac_S2 in iterator:
        canon_paths_S2n = []
        for path, reacs_n in path_reac_S2:
            try:
                canon_path = canonicalize_path_string(path)
                canon_paths_S2n.append((canon_path, reacs_n))
            except Exception as e:
                # print(f"Raised {e=}")
                counter += 1
        canon_paths_NS2n.append(canon_paths_S2n)
    print(f"Failed to canonicalize {counter=} path strings")
    return canon_paths_NS2n


def process_paths(
    paths_NS2n: PathsProcessedType,
    true_products: List[str],
    true_reacs: Optional[List[str]]=None,
    commercial_stock: Optional[Set[str]] = None,
    verbose: bool = False,
) -> PathsProcessedType:
    canon_paths_NS2n = canonicalize_paths(paths_NS2n, verbose)
    unique_paths_NS2n = remove_repetitions_within_beam_result(canon_paths_NS2n, verbose)
    if commercial_stock is None:
        available_paths_NS2n = unique_paths_NS2n
    else:
        available_paths_NS2n = find_paths_with_commercial_sm(
            unique_paths_NS2n, commercial_stock, verbose
        )
    correct_paths_NS2n = find_paths_with_correct_product_and_reactants(
        available_paths_NS2n, true_products, true_reacs, verbose
    )
    return correct_paths_NS2n


def process_paths_post(
    paths_NS2n: PathsProcessedType,
    true_products: List[str],
    true_reacs: List[str],
    commercial_stock: Set[str],
) -> PathsProcessedType:
    unique_paths_NS2n = remove_repetitions_within_beam_result(paths_NS2n)
    available_paths_NS2n = find_paths_with_commercial_sm(
        unique_paths_NS2n, commercial_stock
    )
    correct_paths_NS2n = find_paths_with_correct_product_and_reactants(
        available_paths_NS2n, true_products, true_reacs
    )
    canon_paths_NS2n = canonicalize_paths(correct_paths_NS2n)
    return canon_paths_NS2n


def load_pharma_compounds(
    path_to_json: str,
) -> Tuple[List[str], List[str], List[str], List[int], Dict[str, List[int]]]:
    with open(path_to_json, "r") as file:
        data = json.load(file)
    _products, _sms, _path_strings, _steps_list = [], [], [], []
    nameToIdx: Dict[str, List[int]] = {}
    idx = 0
    for item in data:
        path_dict = eval(item["path"])
        all_sm = find_leaves(path_dict)
        for sm in all_sm:
            nameToIdx.setdefault(item["name"], []).append(idx)
            _path_strings.append(item["path"])
            _products.append(eval(item["path"])["smiles"])
            _sms.append(sm)
            _steps_list.append(max_tree_depth(path_dict))
            idx += 1

    return _products, _sms, _path_strings, _steps_list, nameToIdx
