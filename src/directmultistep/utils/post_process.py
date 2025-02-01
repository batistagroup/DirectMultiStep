from typing import Iterator, cast

from tqdm import tqdm

from directmultistep.utils.logging_config import logger
from directmultistep.utils.pre_process import (
    FilteredDict,
    canonicalize_smiles,
    find_leaves,
    generate_permutations,
    stringify_dict,
)

SHOW_PROGRESS_BARS = False


BeamResultType = list[list[tuple[str, float]]]
PathReacType = tuple[str, list[str]]
BeamProcessedType = list[PathReacType]
PathsProcessedType = list[BeamProcessedType]
MatchList = list[int | None]


def count_unsolved_targets(beam_results_NS2: BeamResultType | PathsProcessedType) -> int:
    """Counts the number of unsolved targets in a list of beam results.

    An unsolved target is defined as a target for which the list of paths is empty. Note that this
    differs from the typical definition of a solved target. Typically, solved targets are
    defined as targets with routes where all starting materials (SMs) are in a given stock compound
    set.

    Args:
        beam_results_NS2: A list of beam results, where each beam result is a
            list of paths.

    Returns:
        The number of unsolved targets.
    """
    n_empty = 0
    for path_list in beam_results_NS2:
        if len(path_list) == 0:
            n_empty += 1
    return n_empty


def find_valid_paths(beam_results_NS2: BeamResultType) -> PathsProcessedType:
    """Finds valid paths from beam search results.

    This function processes beam search results, extracts the path string,
    canonicalizes the SMILES strings of the reactants, and returns a list of
    valid paths with canonicalized SMILES.

    Args:
        beam_results_NS2: A list of beam results, where each beam result is a
            list of (path_string, score) tuples.

    Returns:
        A list of valid paths, where each path is a tuple of
        (canonicalized_path_string, list_of_canonicalized_reactant_SMILES).
    """
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
    """Finds matching paths between predicted paths and correct paths.

    This function compares predicted paths with a list of correct paths and
    returns the rank at which the correct path was found. It also checks for
    matches after considering all permutations of the predicted path.

    Args:
        paths_NS2n: A list of predicted paths, where each path is a list of
            (path_string, list_of_reactant_SMILES) tuples.
        correct_paths: A list of correct path strings.
        ignore_ids: A set of indices to ignore during matching.

    Returns:
        A tuple containing two lists:
            - match_accuracy_N: List of ranks at which the correct path was
              found (None if not found).
            - perm_match_accuracy_N: List of ranks at which the correct path
              was found after considering permutations (None if not found).
    """
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
    """Calculates top-n accuracy for a list of match ranks.

    This function calculates the fraction of paths that were found within the
    top-n ranks for a given list of n values.

    Args:
        match_accuracy: A list of ranks at which the correct path was found
            (None if not found).
        n_vals: A list of n values for which to calculate top-n accuracy.
        dec_digs: The number of decimal digits to round the accuracy to.

    Returns:
        A dictionary mapping "Top n" to the corresponding accuracy fraction
        (as a string).
    """
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
    """Removes duplicate paths within each beam result.

    This function iterates through each beam result and removes duplicate paths
    based on their stringified representation after considering all permutations.

    Args:
        paths_NS2n: A list of beam results, where each beam result is a list of
            (path_string, list_of_reactant_SMILES) tuples.

    Returns:
        A list of beam results with duplicate paths removed.
    """
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
    """Finds paths that use only commercially available starting materials.

    This function filters a list of paths, keeping only those where all
    reactants are present in the provided commercial stock.

    Args:
        paths_NS2n: A list of beam results, where each beam result is a list of
            (path_string, list_of_reactant_SMILES) tuples.
        commercial_stock: A set of SMILES strings representing commercially
            available starting materials.

    Returns:
        A list of beam results containing only paths with commercial starting
        materials.
    """
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
    """Finds paths that have the correct product and, optionally, the correct reactants.

    This function filters a list of paths, keeping only those where the product
    SMILES matches the corresponding true product SMILES, and optionally,
    where at least one of the reactants matches the corresponding true reactant
    SMILES.

    Args:
        paths_NS2n: A list of beam results, where each beam result is a list of
            (path_string, list_of_reactant_SMILES) tuples.
        true_products: A list of SMILES strings representing the correct
            products.
        true_reacs: An optional list of SMILES strings representing the correct
            reactants.

    Returns:
        A list of beam results containing only paths with the correct product
        and reactants (if provided).
    """
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
    """Canonicalizes a FilteredDict representing a path.

    This function recursively canonicalizes the SMILES strings in a
    FilteredDict and its children.

    Args:
        path_dict: A FilteredDict representing a path.

    Returns:
        A FilteredDict with canonicalized SMILES strings.
    """
    canon_dict: FilteredDict = {}
    canon_dict["smiles"] = canonicalize_smiles(path_dict["smiles"])
    if "children" in path_dict:
        canon_dict["children"] = []
        for child in path_dict["children"]:
            canon_dict["children"].append(canonicalize_path_dict(child))
    return canon_dict


def canonicalize_path_string(path_string: str) -> str:
    """Canonicalizes a path string.

    This function converts a path string to a FilteredDict, canonicalizes it,
    and then converts it back to a string.

    Args:
        path_string: A string representing a path.

    Returns:
        A canonicalized string representation of the path.
    """
    canon_dict = canonicalize_path_dict(eval(path_string))
    return stringify_dict(canon_dict)


def canonicalize_paths(paths_NS2n: PathsProcessedType) -> PathsProcessedType:
    """Canonicalizes all paths in a list of beam results.

    This function iterates through each beam result and canonicalizes the path
    strings.

    Args:
        paths_NS2n: A list of beam results, where each beam result is a list of
            (path_string, list_of_reactant_SMILES) tuples.

    Returns:
        A list of beam results with canonicalized path strings.
    """
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
    """Processes a list of paths by canonicalizing, removing repetitions, and filtering.

    This function performs a series of processing steps on a list of paths,
    including canonicalization, removal of repetitions, filtering by commercial
    availability, and filtering by correct product and reactants.

    Args:
        paths_NS2n: A list of beam results, where each beam result is a list of
            (path_string, list_of_reactant_SMILES) tuples.
        true_products: A list of SMILES strings representing the correct
            products.
        true_reacs: An optional list of SMILES strings representing the correct
            reactants.
        commercial_stock: An optional set of SMILES strings representing
            commercially available starting materials.

    Returns:
        A tuple containing:
            - A list of beam results containing only the correct paths.
            - A dictionary containing the number of solved targets at each
              stage of processing.
    """
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
    """Processes a list of paths by canonicalizing, removing repetitions, and filtering.

    This function performs a series of processing steps on a list of paths,
    including canonicalization, removal of repetitions, filtering by commercial
    availability, and filtering by correct product and reactants.
    This function is similar to `process_paths` but does not return the
    solvability dictionary.

    Args:
        paths_NS2n: A list of beam results, where each beam result is a list of
            (path_string, list_of_reactant_SMILES) tuples.
        true_products: A list of SMILES strings representing the correct
            products.
        true_reacs: An optional list of SMILES strings representing the correct
            reactants.
        commercial_stock: An optional set of SMILES strings representing
            commercially available starting materials.

    Returns:
        A list of beam results containing only the correct paths.
    """
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
    """Processes a list of paths by removing repetitions, filtering, and canonicalizing.

    This function performs a series of processing steps on a list of paths,
    including removal of repetitions, filtering by commercial availability,
    filtering by correct product and reactants, and canonicalization.

    Args:
        paths_NS2n: A list of beam results, where each beam result is a list of
            (path_string, list_of_reactant_SMILES) tuples.
        true_products: A list of SMILES strings representing the correct
            products.
        true_reacs: A list of SMILES strings representing the correct reactants.
        commercial_stock: A set of SMILES strings representing commercially
            available starting materials.

    Returns:
        A list of beam results containing only the correct paths, canonicalized.
    """
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
        match_accuracy: List of ranks at which each path was found (None if not
            found)
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
