import json
import pickle
import random
from pathlib import Path

from tqdm import tqdm

from directmultistep.utils.io import (
    convert_dict_of_lists_to_list_of_dicts,
    convert_list_of_dicts_to_dict_of_lists,
    load_dataset_sm,
    save_dataset_sm,
)
from directmultistep.utils.pre_process import (
    FilteredDict,
    canonicalize_smiles,
    filter_mol_nodes,
    find_leaves,
    generate_permutations,
    max_tree_depth,
)

data_path = Path(__file__).parent / "paroutes"
save_path = Path(__file__).parent / "processed"

ProductsType = list[str]
FilteredType = list[FilteredDict]
DatasetEntry = dict[str, str | int | list[str]]
Dataset = list[DatasetEntry]


class PaRoutesDataset:
    def __init__(self, data_path: Path, filename: str, verbose: bool = True) -> None:
        self.data_path = data_path
        self.filename = filename
        with open(data_path.joinpath(filename)) as f:
            self.dataset = json.load(f)

        self.verbose = verbose

        self.products: list[str] = []
        self.filtered_data: FilteredType = []
        # self.path_strings: List[str] = []
        # self.max_steps: List[int] = []
        # self.SMs: List[List[str]] = []

        # self.non_permuted_path_strings: List[str] = []

    def filter_dataset(self) -> None:
        if self.verbose:
            print("- Filtering all_routes to remove meta data")
        for route in tqdm(self.dataset):
            filtered_node = filter_mol_nodes(route)
            self.filtered_data.append(filtered_node)
            self.products.append(filtered_node["smiles"])

    def prepare_final_dataset_v2(
        self,
        save_path: Path,
        n_perms: int | None = None,
        exclude_path_strings: set[str] | None = None,
        n_sms: int | None = None,
    ) -> set[str]:
        self.filter_dataset()
        products: list[str] = []
        starting_materials: list[str] = []
        path_strings: list[str] = []
        n_steps_list: list[int] = []
        non_permuted_paths: set[str] = set()

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
                    permuted_path_strings = generate_permutations(filtered_route, max_perm=n_perms)

                for path_string in permuted_path_strings:
                    for sm_count, starting_material in enumerate(all_SMs):
                        products.append(filtered_route["smiles"])
                        starting_materials.append(starting_material)
                        path_strings.append(path_string)
                        n_steps_list.append(n_steps)
                        if n_sms is not None and sm_count + 1 >= n_sms:
                            break
        print(f"Created dataset with {len(products)} entries")
        with open(save_path, "wb") as f:
            pickle.dump((products, starting_materials, path_strings, n_steps_list), f)
        return non_permuted_paths


# ------- Dataset Processing -------
print("--- Processing of the PaRoutes dataset begins!")
print("-- starting to canonicalize n1 and n5 stocks")
n1_stock = open(data_path / "n1-stock.txt").read().splitlines()  # noqa: SIM115
n5_stock = open(data_path / "n5-stock.txt").read().splitlines()  # noqa: SIM115

n1_stock_canon = [canonicalize_smiles(smi) for smi in n1_stock]
n5_stock_canon = [canonicalize_smiles(smi) for smi in n5_stock]

with open(data_path / "n1-stock.txt", "w") as f:
    f.write("\n".join(n1_stock_canon))

with open(data_path / "n5-stock.txt", "w") as f:
    f.write("\n".join(n5_stock_canon))


print("-- starting to process n1 Routes")
n_perms: int | None = None  # None for all
n_sms: int | None = 1  # None for all
perm_suffix = "all" if n_perms is None else str(n_perms)
sm_suffix = "all" if n_sms is None else str(n_sms)
n1_routes_obj = PaRoutesDataset(data_path, "n1-routes.json")
n1_path_set = n1_routes_obj.prepare_final_dataset_v2(
    save_path / f"n1_dataset_nperms={perm_suffix}_nsms={sm_suffix}.pkl",
    n_perms=n_perms,
    n_sms=n_sms,
)
pickle.dump(n1_path_set, open(save_path / f"n1_nperms={perm_suffix}_nsms={sm_suffix}_path_set.pkl", "wb"))  # noqa: SIM115

print("-- starting to process n5 Routes")
n5_routes_obj = PaRoutesDataset(data_path, "n5-routes.json")
n5_path_set = n5_routes_obj.prepare_final_dataset_v2(
    save_path / f"n5_dataset_nperms={perm_suffix}_nsms={sm_suffix}.pkl", n_perms=n_perms, n_sms=n_sms
)
pickle.dump(n5_path_set, open(save_path / f"n5_nperms={perm_suffix}_nsms={sm_suffix}_path_set.pkl", "wb"))  # noqa: SIM115

n1_path_set = pickle.load(open(save_path / "n1_nperms=all_nsms=1_path_set.pkl", "rb"))  # noqa: SIM115
n5_path_set = pickle.load(open(save_path / "n5_nperms=all_nsms=1_path_set.pkl", "rb"))  # noqa: SIM115

print("-- starting to process All Routes")
all_routes_obj = PaRoutesDataset(data_path, "all_routes.json")
all_routes_obj.prepare_final_dataset_v2(
    save_path / "all_dataset_nperms=3_nsms=1.pkl",
    n_perms=1,
    n_sms=1,
    exclude_path_strings=n1_path_set | n5_path_set,
)


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


def remove_sm_from_ds(load_path: Path, save_path: Path) -> None:
    products, _, path_strings, n_steps_lists = pickle.load(open(load_path, "rb"))  # noqa: SIM115
    pickle.dump((products, path_strings, n_steps_lists), open(save_path, "wb"))  # noqa: SIM115


remove_sm_from_ds(
    load_path=save_path / "all_dataset_nperms=1_nsms=1.pkl", save_path=save_path / "all_dataset_nperms=1_nosm.pkl"
)
remove_sm_from_ds(
    load_path=save_path / "n1_dataset_nperms=1_nsms=1.pkl", save_path=save_path / "n1_dataset_nperms=1_nosm.pkl"
)
remove_sm_from_ds(
    load_path=save_path / "n5_dataset_nperms=1_nsms=1.pkl", save_path=save_path / "n5_dataset_nperms=1_nosm.pkl"
)

# ------- Create train/val partitions -------
train_fname = "unique_dataset_nperms=3_nsms=all_noboth.pkl"
ds_dict = load_dataset_sm(save_path / train_fname)
ds_list = convert_dict_of_lists_to_list_of_dicts(ds_dict)

random.seed(42)
random.shuffle(ds_list)

val_frac = 0.05
train_ds = ds_list[: int(len(ds_list) * (1 - val_frac))]
val_ds = ds_list[int(len(ds_list) * (1 - val_frac)) :]
print(f"Train dataset size: {len(train_ds)}")
print(f"Validation dataset size: {len(val_ds)}")

train_ds_dict = convert_list_of_dicts_to_dict_of_lists(train_ds)
val_ds_dict = convert_list_of_dicts_to_dict_of_lists(val_ds)

save_dataset_sm(train_ds_dict, save_path / f"{train_fname.split('.')[0]}_train={1 - val_frac}.pkl")
save_dataset_sm(val_ds_dict, save_path / f"{train_fname.split('.')[0]}_val={val_frac}.pkl")
