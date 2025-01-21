import json
import pickle
from pathlib import Path
from typing import Any, TypedDict

from directmultistep.utils.pre_process import (
    canonicalize_smiles,
    find_leaves,
    max_tree_depth,
)


class DatasetDict(TypedDict, total=False):
    products: list[str]
    starting_materials: list[str]
    path_strings: list[str]
    n_steps_list: list[int]
    ds_name: str
    nameToIdx: dict[str, list[int]] | None


def load_dataset_sm(path: Path) -> DatasetDict:
    with open(path, "rb") as file:
        products, starting_materials, path_strings, n_steps_list = pickle.load(file)
    ds_name = path.stem.split("_")[0]
    return {
        "products": products,
        "starting_materials": starting_materials,
        "path_strings": path_strings,
        "n_steps_list": n_steps_list,
        "ds_name": ds_name,
    }


def load_dataset_nosm(path: Path) -> DatasetDict:
    with open(path, "rb") as file:
        products, _, path_strings, n_steps_list = pickle.load(file)
    ds_name = path.stem.split("_")[0]
    return {
        "products": products,
        "path_strings": path_strings,
        "n_steps_list": n_steps_list,
        "ds_name": ds_name,
    }


def save_dataset_sm(data: dict[str, Any], path: Path) -> None:
    with open(path, "wb") as file:
        p, sm, ps, ns = data["products"], data.get("starting_materials", []), data["path_strings"], data["n_steps_list"]
        pickle.dump((p, sm, ps, ns), file)


def convert_dict_of_lists_to_list_of_dicts(dict_of_lists: DatasetDict) -> list[dict[str, str]]:
    return [dict(zip(dict_of_lists.keys(), values)) for values in zip(*dict_of_lists.values())]


def convert_list_of_dicts_to_dict_of_lists(list_of_dicts: list[dict[str, str]]) -> dict[str, list[str]]:
    return {key: [item[key] for item in list_of_dicts] for key in list_of_dicts[0].keys()}


def load_pharma_compounds(
    path_to_json: Path,
    load_sm: bool = True,
) -> DatasetDict:
    with open(path_to_json, "r") as file:
        data = json.load(file)
    _products, _sms, _path_strings, _steps_list = [], [], [], []
    name_idx: dict[str, list[int]] = {}
    idx = 0
    for item in data:
        path_dict = eval(item["path"])
        all_sm = find_leaves(path_dict)
        if load_sm:
            for sm in all_sm:
                name_idx.setdefault(item["name"], []).append(idx)
                _path_strings.append(item["path"])
                _products.append(eval(item["path"])["smiles"])
                _sms.append(sm)
                _steps_list.append(max_tree_depth(path_dict))
                idx += 1
        else:
            name_idx.setdefault(item["name"], []).append(idx)
            _path_strings.append(item["path"])
            _products.append(eval(item["path"])["smiles"])
            _steps_list.append(max_tree_depth(path_dict))
            idx += 1

    if load_sm:
        return {
            "products": _products,
            "starting_materials": _sms,
            "path_strings": _path_strings,
            "n_steps_list": _steps_list,
            "nameToIdx": name_idx,
        }
    else:
        return {
            "products": _products,
            "path_strings": _path_strings,
            "n_steps_list": _steps_list,
            "nameToIdx": name_idx,
        }


def load_commercial_stock(path: Path) -> set[str]:
    with open(path, "r") as file:
        stock = file.readlines()
    canonical_stock = set()
    for molecule in stock:
        canonical_stock.add(canonicalize_smiles(molecule.strip()))
    return canonical_stock
