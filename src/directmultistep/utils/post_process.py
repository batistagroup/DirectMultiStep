from typing import Iterator, TypedDict, cast

from tqdm import tqdm

from directmultistep.utils.logging_config import logger
from directmultistep.utils.pre_process import (
    canonicalize_smiles,
    find_leaves,
    generate_permutations,
    stringify_dict,
)

SHOW_PROGRESS_BARS = False


class FilteredDict(TypedDict, total=False):
    smiles: str
    children: list["FilteredDict"]


BeamResultType = list[list[tuple[str, float]]]
PathReacType = tuple[str, list[str]]
BeamProcessedType = list[PathReacType]
PathsProcessedType = list[BeamProcessedType]
MatchList = list[int | None]


def count_unsolved_targets(beam_results_NS2: BeamResultType | PathsProcessedType) -> int:
    n_empty = 0
    for path_list in beam_results_NS2:
        if len(path_list) == 0:
            n_empty += 1
    return n_empty


def find_valid_paths(beam_results_NS2: BeamResultType) -> PathsProcessedType:
    valid_pathreac_NS2n = []
    iterator = tqdm(beam_results_NS2) if SHOW_PROGRESS_BARS else beam_results_NS2
    for beam_result_S2 in cast(Iterator[list[tuple[str, float]]], iterator):
        valid_pathreac_S2n = []
        for path_string, _ in beam_result_S2:
            try:
                node = eval(path_string)
                reactants = find_leaves(node)
                canon_reactants = [canonicalize_smiles(reactant) for reactant in reactants]
                canon_path = canonicalize_path_string(path_string)
            except:  # noqa: E722
                continue
            valid_pathreac_S2n.append((canon_path, canon_reactants))
        valid_pathreac_NS2n.append(valid_pathreac_S2n)
    return valid_pathreac_NS2n


def find_matching_paths(
    paths_NS2n: PathsProcessedType, correct_paths: list[str], ignore_ids: set[int] | None = None
) -> tuple[MatchList, MatchList]:
    if ignore_ids is None:
        ignore_ids = set()
    match_accuracy_N: MatchList = []
    perm_match_accuracy_N: MatchList = []
    iterator = (
        tqdm(enumerate(zip(paths_NS2n, correct_paths)), total=len(paths_NS2n))
        if SHOW_PROGRESS_BARS
        else enumerate(zip(paths_NS2n, correct_paths))
    )
    for i, (pathreac_S2n, correct_path) in cast(Iterator[tuple[int, tuple[BeamProcessedType, str]]], iterator):
        if i in ignore_ids:
            continue
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


def find_top_n_accuracy(match_accuracy: MatchList, n_vals: list[int], dec_digs: int = 1) -> dict[str, str]:
    n_vals = sorted(n_vals)
    top_counts = {f"Top {n}": 0 for n in n_vals}
    for rank in match_accuracy:
        if rank is None:
            continue
        for n in n_vals:
            if rank <= n:
                top_counts[f"Top {n}"] += 1
    top_fractions = {k: f"{(v / len(match_accuracy)* 100):.{dec_digs}f}" for k, v in top_counts.items()}
    return top_fractions


def remove_repetitions_within_beam_result(
    paths_NS2n: PathsProcessedType,
) -> PathsProcessedType:
    unique_paths_NS2n = []
    iterator = tqdm(paths_NS2n) if SHOW_PROGRESS_BARS else paths_NS2n
    for path_reac_S2 in cast(Iterator[BeamProcessedType], iterator):
        unique_paths_S2n = []
        seen = set()
        for path, reacs_n in path_reac_S2:
            for permuted_pathstring in generate_permutations(data=eval(path), max_perm=None):
                if permuted_pathstring in seen:
                    break
            else:
                seen.add(path)
                unique_paths_S2n.append((path, reacs_n))
        unique_paths_NS2n.append(unique_paths_S2n)
    return unique_paths_NS2n


def find_paths_with_commercial_sm(paths_NS2n: PathsProcessedType, commercial_stock: set[str]) -> PathsProcessedType:
    available_paths_NS2n = []
    iterator = tqdm(paths_NS2n) if SHOW_PROGRESS_BARS else paths_NS2n
    for path_reac_S2 in cast(Iterator[BeamProcessedType], iterator):
        available_paths_S2n = []
        for path, reacs_n in path_reac_S2:
            if all(reactant in commercial_stock for reactant in reacs_n):
                available_paths_S2n.append((path, reacs_n))
        available_paths_NS2n.append(available_paths_S2n)
    return available_paths_NS2n


def find_paths_with_correct_product_and_reactants(
    paths_NS2n: PathsProcessedType,
    true_products: list[str],
    true_reacs: list[str] | None = None,
) -> PathsProcessedType:
    f = canonicalize_smiles
    correct_paths_NS2n = []
    iterator = tqdm(enumerate(paths_NS2n)) if SHOW_PROGRESS_BARS else enumerate(paths_NS2n)
    for idx, path_reac_S2 in cast(Iterator[tuple[int, BeamProcessedType]], iterator):
        correct_paths_S2n = []
        for path, reacs_n in path_reac_S2:
            path_tree = eval(path)
            if f(path_tree["smiles"]) == f(true_products[idx]) and (
                true_reacs is None or f(true_reacs[idx]) in reacs_n
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


def canonicalize_paths(paths_NS2n: PathsProcessedType) -> PathsProcessedType:
    canon_paths_NS2n = []
    counter = 0
    iterator = tqdm(paths_NS2n) if SHOW_PROGRESS_BARS else paths_NS2n
    for path_reac_S2 in cast(Iterator[BeamProcessedType], iterator):
        canon_paths_S2n = []
        for path, reacs_n in path_reac_S2:
            try:
                canon_path = canonicalize_path_string(path)
                canon_paths_S2n.append((canon_path, reacs_n))
            except Exception:
                counter += 1
        canon_paths_NS2n.append(canon_paths_S2n)
    if counter > 0:
        logger.warning(f"Failed to canonicalize {counter} path strings")
    return canon_paths_NS2n


def process_paths(
    paths_NS2n: PathsProcessedType,
    true_products: list[str],
    true_reacs: list[str] | None = None,
    commercial_stock: set[str] | None = None,
) -> tuple[PathsProcessedType, dict[str, int]]:
    canon_paths_NS2n = canonicalize_paths(paths_NS2n)
    unique_paths_NS2n = remove_repetitions_within_beam_result(canon_paths_NS2n)
    if commercial_stock is None:
        available_paths_NS2n = unique_paths_NS2n
    else:
        available_paths_NS2n = find_paths_with_commercial_sm(unique_paths_NS2n, commercial_stock)
    correct_paths_NS2n = find_paths_with_correct_product_and_reactants(available_paths_NS2n, true_products, true_reacs)
    total = len(true_products)
    solvability = {
        "solved (canonicalized)": total - count_unsolved_targets(canon_paths_NS2n),
        "solved (unique)": total - count_unsolved_targets(unique_paths_NS2n),
        "solved (available)": total - count_unsolved_targets(available_paths_NS2n),
        "solved (correct)": total - count_unsolved_targets(correct_paths_NS2n),
    }
    return correct_paths_NS2n, solvability


def process_path_single(
    paths_NS2n: PathsProcessedType,
    true_products: list[str],
    true_reacs: list[str] | None = None,
    commercial_stock: set[str] | None = None,
) -> PathsProcessedType:
    canon_paths_NS2n = canonicalize_paths(paths_NS2n)
    unique_paths_NS2n = remove_repetitions_within_beam_result(canon_paths_NS2n)
    if commercial_stock is None:
        available_paths_NS2n = unique_paths_NS2n
    else:
        available_paths_NS2n = find_paths_with_commercial_sm(unique_paths_NS2n, commercial_stock)
    correct_paths_NS2n = find_paths_with_correct_product_and_reactants(available_paths_NS2n, true_products, true_reacs)
    return correct_paths_NS2n


def process_paths_post(
    paths_NS2n: PathsProcessedType,
    true_products: list[str],
    true_reacs: list[str],
    commercial_stock: set[str],
) -> PathsProcessedType:
    unique_paths_NS2n = remove_repetitions_within_beam_result(paths_NS2n)
    available_paths_NS2n = find_paths_with_commercial_sm(unique_paths_NS2n, commercial_stock)
    correct_paths_NS2n = find_paths_with_correct_product_and_reactants(available_paths_NS2n, true_products, true_reacs)
    canon_paths_NS2n = canonicalize_paths(correct_paths_NS2n)
    return canon_paths_NS2n


def calculate_top_k_counts_by_step_length(
    match_accuracy: list[int | None], n_steps_list: list[int], k_vals: list[int]
) -> dict[int, dict[str, int]]:
    """Calculate accuracy statistics grouped by number of steps.

    Args:
        match_accuracy: List of ranks at which each path was found (None if not found)
        n_steps_list: List of number of steps for each path
        k_vals: List of k values to calculate top-k accuracy for

    Returns:
        Dictionary mapping step count to accuracy statistics
    """
    step_stats: dict[int, dict[str, int]] = {}

    for rank, n_steps in zip(match_accuracy, n_steps_list):
        if n_steps not in step_stats:
            step_stats[n_steps] = {"Total": 0}

        step_stats[n_steps]["Total"] += 1

        if rank is None:
            step_stats[n_steps]["Not Found"] = step_stats[n_steps].get("Not Found", 0) + 1
        else:
            for k in k_vals:
                if rank <= k:
                    step_stats[n_steps][f"Top {k}"] = step_stats[n_steps].get(f"Top {k}", 0) + 1

    return step_stats
