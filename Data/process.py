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

import json
import pickle
from tqdm import tqdm
from DirectMultiStep.Utils.PreProcess import (
    filter_mol_nodes,
    max_tree_depth,
    find_leaves,
    FilteredDict,
    generate_permutations,
)
from pathlib import Path
from typing import List, Tuple, Dict, Union, Set, Optional, cast

data_path = Path(__file__).parent / "PaRoutes"
save_path = Path(__file__).parent / "Processed"

ProductsType = List[str]
FilteredType = List[FilteredDict]
DatasetEntry = Dict[str, Union[str, int, List[str]]]
Dataset = List[DatasetEntry]


class PaRoutesDataset:
    def __init__(self, data_path: Path, filename: str, verbose: bool = True):
        self.data_path = data_path
        self.filename = filename
        self.dataset = json.load(open(data_path.joinpath(filename), "r"))

        self.verbose = verbose

        self.products: List[str] = []
        self.filtered_data: FilteredType = []
        self.path_strings: List[str] = []
        self.max_steps: List[int] = []
        self.SMs: List[List[str]] = []

        self.non_permuted_path_strings: List[str] = []

    def filter_dataset(self):
        if self.verbose:
            print("- Filtering all_routes to remove meta data")
        for route in tqdm(self.dataset):
            filtered_node = filter_mol_nodes(route)
            self.filtered_data.append(filtered_node)
            self.products.append(filtered_node["smiles"])

    def compress_to_string(self):
        if self.verbose:
            print(
                "- Compressing python dictionaries into python strings and generating permutations"
            )

        for filtered_route in tqdm(self.filtered_data):
            permuted_path_strings = generate_permutations(filtered_route)
            # permuted_path_strings = [str(data).replace(" ", "")]
            self.path_strings.append(permuted_path_strings)
            self.non_permuted_path_strings.append(str(filtered_route).replace(" ", ""))

    def find_max_depth(self):
        if self.verbose:
            print("- Finding the max depth of each route tree")
        for filtered_route in tqdm(self.filtered_data):
            self.max_steps.append(max_tree_depth(filtered_route))

    def find_all_leaves(self):
        if self.verbose:
            print("- Finding all leaves of each route tree")
        for filtered_route in tqdm(self.filtered_data):
            self.SMs.append(find_leaves(filtered_route))

    def preprocess(self):
        self.filter_dataset()
        self.compress_to_string()
        self.find_max_depth()
        self.find_all_leaves()

    def prepare_final_datasets(
        self, exclude: Optional[Set[int]] = None
    ) -> Tuple[Dataset, Dataset]:
        if exclude is None:
            exclude = set()
        dataset: Dataset = []
        dataset_each_sm: Dataset = []
        for i in tqdm(range(len(self.products))):
            if i in exclude:
                continue
            entry: DatasetEntry = {
                "train_ID": i,
                "product": self.products[i],
                "path_strings": self.path_strings[i],
                "max_step": self.max_steps[i],
            }
            dataset.append(entry | {"all_SM": self.SMs[i]})
            for sm in self.SMs[i]:
                dataset_each_sm.append({**entry, "SM": sm})
        return (dataset, dataset_each_sm)

    def prepare_final_dataset_v2(
        self,
        save_path: Path,
        n_perms: Optional[int] = None,
        exclude_path_strings: Optional[Set[str]] = None,
        n_sms: Optional[int] = None,
    ) -> Set[str]:
        self.filter_dataset()
        products: List[str] = []
        starting_materials: List[str] = []
        path_strings: List[str] = []
        n_steps_list: List[int] = []
        non_permuted_paths: Set[str] = set()

        if exclude_path_strings is None:
            exclude_path_strings = set()

        for filtered_route in tqdm(self.filtered_data):
            non_permuted_string = str(filtered_route).replace(" ", "")
            non_permuted_paths.add(non_permuted_string)
            permuted_path_strings = generate_permutations(filtered_route, max_perm=None)
            for permuted_path_string in permuted_path_strings:
                if permuted_path_string in exclude_path_strings:
                    break
            else:
                n_steps = max_tree_depth(filtered_route)
                all_SMs = find_leaves(filtered_route)
                if n_perms == 1:
                    permuted_path_strings = [non_permuted_string]
                else:
                    permuted_path_strings = generate_permutations(
                        filtered_route, max_perm=n_perms
                    )

                for path_string in permuted_path_strings:
                    for sm_count, starting_material in enumerate(all_SMs):
                        products.append(cast(str, filtered_route["smiles"]))
                        starting_materials.append(starting_material)
                        path_strings.append(path_string)
                        n_steps_list.append(n_steps)
                        if n_sms is not None and sm_count + 1 >= n_sms:
                            break
        print(f"Created dataset with {len(products)} entries")
        pickle.dump(
            (products, starting_materials, path_strings, n_steps_list),
            open(save_path, "wb"),
        )
        return non_permuted_paths


# ------- Dataset Processing -------
# print("--- Processing of the PaRoutes dataset begins!")
# print("-- starting to process n1 Routes")
# n_perms:Optional[int] = None # None for all
# n_sms:Optional[int] = 1 # None for all
# perm_suffix = "all" if n_perms is None else str(n_perms)
# sm_suffix = "all" if n_sms is None else str(n_sms)
# n1_routes_obj = PaRoutesDataset(data_path, "n1-routes.json")
# n1_path_set = n1_routes_obj.prepare_final_dataset_v2(
#     save_path / f"n1_dataset_nperms={perm_suffix}_nsms={sm_suffix}.pkl", n_perms=n_perms, n_sms=n_sms,
# )
# pickle.dump(n1_path_set, open(save_path / f"n1_nperms={perm_suffix}_nsms={sm_suffix}_path_set.pkl", "wb"))

# print("-- starting to process n5 Routes")
# n5_routes_obj = PaRoutesDataset(data_path, "n5-routes.json")
# n5_path_set = n5_routes_obj.prepare_final_dataset_v2(
#     save_path / f"n5_dataset_nperms={perm_suffix}_nsms={sm_suffix}.pkl", n_perms=n_perms, n_sms=n_sms
# )
# pickle.dump(n5_path_set, open(save_path / f"n5_nperms={perm_suffix}_nsms={sm_suffix}_path_set.pkl", "wb"))

# n1_path_set = pickle.load(open(save_path / "n1_nperms=all_nsms=1_path_set.pkl", "rb"))
# n5_path_set = pickle.load(open(save_path / "n5_nperms=all_nsms=1_path_set.pkl", "rb"))

# print("-- starting to process All Routes")
# all_routes_obj = PaRoutesDataset(data_path, "all_routes.json")
# all_routes_obj.prepare_final_dataset_v2(
#     save_path / "all_dataset_nperms=3_nsms=1.pkl",
#     n_perms=1,
#     n_sms=1,
#     exclude_path_strings=n1_path_set | n5_path_set,
# )


# ------- Prepare Evaluation Subsets -------
# testing_dataset = "n5"

# (_products, _sms, _path_strings, _steps_list) = pickle.load(
#     open(save_path / f"{testing_dataset}_dataset_nperms=1_nsms=1.pkl", "rb")
# )
# combined = [{"product": p, "SM": s, "path_string": ps, "steps": st} for p, s, ps, st in zip(_products, _sms, _path_strings, _steps_list)]

# # shuffle the list
# import random
# random.seed(42)
# random.shuffle(combined)
# _sh_prods = [x["product"] for x in combined]
# _sh_sms = [x["SM"] for x in combined]
# _sh_paths = [x["path_string"] for x in combined]
# _sh_steps = [x["steps"] for x in combined]
# for n_elts in [10, 50,]:
#     pickle.dump((_sh_prods[:n_elts], _sh_sms[:n_elts], _sh_paths[:n_elts], _sh_steps[:n_elts]), open(save_path / f"{testing_dataset}_shuffled_seed42_n{n_elts}.pkl", "wb"))


# ------- Prepare Evaluation Subsets -------
# testing_dataset = "n1"

# (_products, _sms, _path_strings, _steps_list) = pickle.load(
#     open(save_path / f"{testing_dataset}_dataset_nperms=1_nsms=1.pkl", "rb")
# )

# first, second, third, fourth = 2500, 5000, 7500, 10000
# pickle.dump((_products[:first], _sms[:first], _path_strings[:first], _steps_list[:first]), open(save_path / f"{testing_dataset}_dataset_nperms=1_nsms=1_n{first}.pkl", "wb"))
# pickle.dump((_products[first:second], _sms[first:second], _path_strings[first:second], _steps_list[first:second]), open(save_path / f"{testing_dataset}_dataset_nperms=1_nsms=1_n{second}.pkl", "wb"))
# pickle.dump((_products[second:third], _sms[second:third], _path_strings[second:third], _steps_list[second:third]), open(save_path / f"{testing_dataset}_dataset_nperms=1_nsms=1_n{third}.pkl", "wb"))
# pickle.dump((_products[third:fourth], _sms[third:fourth], _path_strings[third:fourth], _steps_list[third:fourth]), open(save_path / f"{testing_dataset}_dataset_nperms=1_nsms=1_n{fourth}.pkl", "wb"))


# ------- Remove SM info from datasets -------


def remove_sm_from_ds(load_path: Path, save_path: Path):
    products, _, path_strings, n_steps_lists = pickle.load(open(load_path, "rb"))
    pickle.dump((products, path_strings, n_steps_lists), open(save_path, "wb"))


# remove_sm_from_ds(load_path=save_path / "all_dataset_nperms=1_nsms=1.pkl", save_path=save_path / "all_dataset_nperms=1_nosm.pkl")
# remove_sm_from_ds(load_path=save_path / "n1_dataset_nperms=1_nsms=1.pkl", save_path=save_path / "n1_dataset_nperms=1_nosm.pkl")
# remove_sm_from_ds(load_path=save_path / "n5_dataset_nperms=1_nsms=1.pkl", save_path=save_path / "n5_dataset_nperms=1_nosm.pkl")
