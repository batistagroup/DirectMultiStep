import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import torch.nn as nn
import yaml
from torch import device as torch_device
from torch.utils.data import DataLoader
from tqdm import tqdm

from directmultistep.generation.tensor_gen import BeamSearchOptimized as BeamSearch
from directmultistep.training.config import TrainingConfig
from directmultistep.utils.dataset import RoutesDataset
from directmultistep.utils.io import (
    load_commercial_stock,
    load_dataset_nosm,
    load_dataset_sm,
    load_pharma_compounds,
)
from directmultistep.utils.post_process import (
    BeamResultType,
    canonicalize_paths,
    count_unsolved_targets,
    find_matching_paths,
    find_paths_with_commercial_sm,
    find_paths_with_correct_product_and_reactants,
    find_top_n_accuracy,
    find_valid_paths,
    remove_repetitions_within_beam_result,
)

ds_name_to_fname = {
    "n1_50": "n1_shuffled_seed42_n500_subset50_seed1337.pkl",
    "n1_500": "n1_shuffled_seed42_n500.pkl",
    "n5_50": "n5_shuffled_seed42_n500_subset50_seed42.pkl",
    "n5_500": "n5_shuffled_seed42_n500.pkl",
}


@dataclass
class EvalConfig:
    epoch: int
    data_path: Path
    run_name: str
    eval_dataset: str
    beam_width: int

    use_sm: bool
    use_steps: bool

    enc_active_experts: int | None = None
    dec_active_experts: int | None = None

    batch_size: int = 1
    num_workers: int = 8
    persistent_workers: bool = True

    # post_init values
    _checkpoint_path: Path | None = None
    eval_name: str = ""

    def __post_init__(self) -> None:
        # files = list((self.data_path / "training" / self.run_name).glob(f"epoch={self.epoch}*"))
        # assert len(files) == 1, f"Expected 1 checkpoint file, but found {len(files)}: {files}"
        # self._checkpoint_path: Path = files[0]

        allowed_ds = ["n1_50", "n1_500", "n5_50", "n5_500", "pharma"]
        assert (
            self.eval_dataset in allowed_ds
        ), f"Eval dataset {self.eval_dataset} not in allowed datasets: {allowed_ds}"

        b_str = f"b{self.beam_width}"
        sm_str = "sm" if self.use_sm else "nosm"
        steps_str = "st" if self.use_steps else "nost"
        suffix = ""
        if self.enc_active_experts is not None:
            suffix += f"_ea={self.enc_active_experts}"
        if self.dec_active_experts is not None:
            suffix += f"_da={self.dec_active_experts}"

        self.eval_name = f"{self.eval_dataset}_{b_str}_{sm_str}_{steps_str}" + suffix

    @property
    def checkpoint_path(self) -> Path:
        assert self._checkpoint_path is not None, "Checkpoint path is not set"
        return self._checkpoint_path

    def save(self, path: Path) -> None:
        """Save config to YAML file.

        Args:
            path: Path to save config file
        """
        config_dict = asdict(self)
        config_dict["data_path"] = str(config_dict["data_path"])
        config_dict["_checkpoint_path"] = None

        with open(path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> "EvalConfig":
        """Load config from YAML file.

        Args:
            path: Path to config file

        Returns:
            Loaded config object
        """
        with open(path) as f:
            config_dict = yaml.safe_load(f)

        config_dict["data_path"] = Path(config_dict["data_path"])
        return cls(**config_dict)


class ModelEvaluator:
    def __init__(self, model: nn.Module, ec: EvalConfig, tc: TrainingConfig, device: torch_device):
        self.model = model
        self.model.eval()
        self.device = device
        self.ec = ec
        self.tc = tc

        self.save_path = (
            self.ec.data_path / "evaluation" / self.tc.run_name / f"epoch={self.ec.epoch}" / self.ec.eval_name
        )
        self.save_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _beam_pickle_exists(save_path: Path) -> bool:
        return (save_path / "all_beam_results_NS2.pkl").exists()

    def load_eval_dataset(self) -> None:
        if self.ec.eval_dataset == "pharma":
            data = load_pharma_compounds(self.ec.data_path / "pharma_compounds.json", load_sm=self.ec.use_sm)
            name_idx = data["nameToIdx"]
        else:
            prcsd = self.ec.data_path / "processed"
            name_idx = None
            if self.ec.use_sm:
                data = load_dataset_sm(prcsd / ds_name_to_fname[self.ec.eval_dataset])
            else:
                data = load_dataset_nosm(prcsd / ds_name_to_fname[self.ec.eval_dataset])
        self.ds = RoutesDataset(
            metadata_path=self.ec.data_path / "configs" / self.tc.metadata_fname,
            products=data["products"],
            path_strings=data["path_strings"],
            n_steps_list=data["n_steps_list"],
            starting_materials=data["starting_materials"],
            mode="generation",
            name_idx=name_idx,
        )
        self.dl = DataLoader(
            self.ds,
            batch_size=self.ec.batch_size,
            shuffle=False,
            num_workers=self.ec.num_workers,
            persistent_workers=self.ec.persistent_workers,
        )

    def prepare_beam_search(self) -> None:
        self.beam = BeamSearch(
            model=self.model,
            beam_size=self.ec.beam_width,
            start_idx=self.ds.token_to_idx["<SOS>"],
            pad_idx=self.ds.token_to_idx[" "],
            end_idx=self.ds.token_to_idx["?"],
            max_length=self.ds.seq_out_max_length,
            idx_to_token=self.ds.idx_to_token,
            device=self.device,
        )

    def run_beam_search(self, save_pickle: bool = True, force_rerun: bool = False) -> BeamResultType:
        if self._beam_pickle_exists(self.save_path) and not force_rerun:
            raise FileExistsError(f"Beam search results already exist at {self.save_path / 'all_beam_results_NS2.pkl'}")
        all_beam_results_NS2: list[list[tuple[str, float]]] = []
        for batch_idx, (prod_sm, steps, path) in tqdm(enumerate(self.dl), total=len(self.dl)):
            beam_result_BS2 = self.beam.decode(
                src_BC=prod_sm.to(self.device),
                steps_B1=steps.to(self.device),
                path_start_BL=path.to(self.device),
            )
            all_beam_results_NS2.extend(beam_result_BS2)
        if save_pickle:
            with open(self.save_path / "all_beam_results_NS2.pkl", "wb") as f:
                pickle.dump(all_beam_results_NS2, f)
        return all_beam_results_NS2

    def calculate_top_k_accuracy(
        self,
        k_vals: list[int] | None = None,
        save_pickle: bool = True,
        check_true_reacs: bool = True,
        check_stock: bool = True,
    ) -> dict[str, int | dict[str, str]]:
        return self.recalculate_top_k_accuracy(
            data_path=self.ec.data_path,
            save_path=self.save_path,
            products=self.ds.products,
            path_strings=self.ds.path_strings,
            starting_materials=self.ds.sms,
            eval_ds=self.ec.eval_dataset,
            k_vals=k_vals,
            save_pickle=save_pickle,
            check_true_reacs=self.ec.use_sm,
            check_stock=check_stock,
        )

    @staticmethod
    def calculate_valid_paths_accuracy(
        save_path: Path,
        path_strings: list[str],
        products: list[str],
        k_vals: list[int] | None = None,
        save_pickle: bool = True,
    ) -> dict[str, int | dict[str, str]]:
        if k_vals is None:
            k_vals = [1, 2, 3, 4, 5, 10, 20, 50]
        solvability = {}
        if (save_path / "valid_paths_NS2n.pkl").exists():
            with open(save_path / "valid_paths_NS2n.pkl", "rb") as f:
                valid_paths_NS2n = pickle.load(f)
        else:
            if not ModelEvaluator._beam_pickle_exists(save_path):
                raise FileNotFoundError(f"Beam search results not found at {save_path / 'all_beam_results_NS2.pkl'}")
            with open(save_path / "all_beam_results_NS2.pkl", "rb") as f:
                all_beam_results_NS2 = pickle.load(f)
            valid_paths_NS2n = find_valid_paths(all_beam_results_NS2)
            solvability["solved (all)"] = len(products) - count_unsolved_targets(all_beam_results_NS2)
            if save_pickle:
                with open(save_path / "valid_paths_NS2n.pkl", "wb") as f:
                    pickle.dump(valid_paths_NS2n, f)

        matches_N, perm_matches_N = find_matching_paths(valid_paths_NS2n, path_strings)
        top_ks = {
            "accuracy (valid, no perms)": find_top_n_accuracy(matches_N, k_vals),
            "accuracy (valid, with perms)": find_top_n_accuracy(perm_matches_N, k_vals),
        }
        solvability["solved (valid)"] = len(products) - count_unsolved_targets(valid_paths_NS2n)

        with open(save_path / "top_k_accuracy_valid.yaml", "w") as f:
            yaml.dump({**solvability, **top_ks}, f, sort_keys=False)

        return {**solvability, **top_ks}

    @staticmethod
    def calculate_processed_paths_accuracy(
        data_path: Path,
        save_path: Path,
        products: list[str],
        path_strings: list[str],
        starting_materials: list[str] | None,
        eval_ds: str,
        k_vals: list[int] | None = None,
        save_pickle: bool = True,
        check_true_reacs: bool = True,
        check_stock: bool = True,
        force_rerun: bool = False,
    ) -> dict[str, int | dict[str, str]]:
        if k_vals is None:
            k_vals = [1, 2, 3, 4, 5, 10, 20, 50]

        if not (save_path / "valid_paths_NS2n.pkl").exists():
            raise FileNotFoundError(f"Valid paths not found at {save_path / 'valid_paths_NS2n.pkl'}")

        # Load valid paths
        with open(save_path / "valid_paths_NS2n.pkl", "rb") as f:
            valid_paths_NS2n = pickle.load(f)

        # Step 1: Remove repetitions
        solvability = {}
        unique_paths_fname = "unique_paths_NS2n.pkl"
        if not force_rerun and (save_path / unique_paths_fname).exists():
            with open(save_path / unique_paths_fname, "rb") as f:
                unique_paths_NS2n = pickle.load(f)
        else:
            canon_paths_NS2n = canonicalize_paths(valid_paths_NS2n)
            unique_paths_NS2n = remove_repetitions_within_beam_result(canon_paths_NS2n)
            if save_pickle:
                with open(save_path / unique_paths_fname, "wb") as f:
                    pickle.dump(unique_paths_NS2n, f)
            solvability["solved (canonicalized)"] = len(products) - count_unsolved_targets(canon_paths_NS2n)
        solvability["solved (unique)"] = len(products) - count_unsolved_targets(unique_paths_NS2n)

        # Step 2: Filter by commercial stock if needed
        if check_stock:
            if eval_ds in ["n1", "n5", "n1_50", "n1_500", "n5_50", "n5_500"]:
                eval_ds = eval_ds.split("_")[0]
                stock = load_commercial_stock(data_path / "paroutes" / f"{eval_ds}-stock.txt")
            else:
                stock = None
            available_paths_NS2n = (
                find_paths_with_commercial_sm(unique_paths_NS2n, stock) if stock else unique_paths_NS2n
            )
        else:
            available_paths_NS2n = unique_paths_NS2n

        # Step 3: Find paths with correct products and reactants
        correct_paths_NS2n = find_paths_with_correct_product_and_reactants(
            available_paths_NS2n,
            true_products=products,
            true_reacs=starting_materials if check_true_reacs else None,
        )

        solvability = {
            "solved (available)": len(products) - count_unsolved_targets(available_paths_NS2n),
            "solved (correct)": len(products) - count_unsolved_targets(correct_paths_NS2n),
        }

        matches_N, perm_matches_N = find_matching_paths(correct_paths_NS2n, path_strings)
        top_ks = {
            "accuracy (processed, no perms)": find_top_n_accuracy(matches_N, k_vals),
            "accuracy (processed, with perms)": find_top_n_accuracy(perm_matches_N, k_vals),
        }

        suffix = f"true_reacs={check_true_reacs}_stock={check_stock}"
        if save_pickle:
            with open(save_path / f"processed_paths_NS2n_{suffix}.pkl", "wb") as f:
                pickle.dump(correct_paths_NS2n, f)

        with open(save_path / f"top_k_accuracy_{suffix}.yaml", "w") as f:
            yaml.dump({**solvability, **top_ks}, f, sort_keys=False)

        return {**solvability, **top_ks}

    @staticmethod
    def recalculate_top_k_accuracy(
        data_path: Path,
        save_path: Path,
        products: list[str],
        path_strings: list[str],
        starting_materials: list[str] | None,
        eval_ds: str,
        k_vals: list[int] | None = None,
        save_pickle: bool = True,
        check_true_reacs: bool = True,
        check_stock: bool = True,
    ) -> dict[str, int | dict[str, str]]:
        """Legacy function that combines both valid and processed paths accuracy calculations."""
        valid_results = ModelEvaluator.calculate_valid_paths_accuracy(
            save_path=save_path,
            path_strings=path_strings,
            products=products,
            k_vals=k_vals,
            save_pickle=save_pickle,
        )
        processed_results = ModelEvaluator.calculate_processed_paths_accuracy(
            data_path=data_path,
            save_path=save_path,
            products=products,
            path_strings=path_strings,
            starting_materials=starting_materials,
            eval_ds=eval_ds,
            k_vals=k_vals,
            save_pickle=save_pickle,
            check_true_reacs=check_true_reacs,
            check_stock=check_stock,
        )
        return {**processed_results, **valid_results}

    def prepare_name_to_rank(self) -> dict[str, list[int | None]]:
        fname = f"{self.ec.eval_dataset}_processed_paths_true_reacs={self.ec.use_sm}_stock=False_NS2n.pkl"
        if not (self.save_path / fname).exists():
            raise FileNotFoundError(f"Correct paths not found at {self.save_path / fname}")
        with open(self.save_path / fname, "rb") as f:
            correct_paths_NS2n = pickle.load(f)
        _, perm_matches_N = find_matching_paths(correct_paths_NS2n, self.ds.path_strings)
        assert self.ds.name_idx is not None, "Name index is None"
        name_to_rank = {name: [perm_matches_N[i] for i in idxs] for name, idxs in self.ds.name_idx.items()}
        with open(self.save_path / "name_to_rank.yaml", "w") as f:
            yaml.dump(name_to_rank, f, sort_keys=False)
        return name_to_rank
