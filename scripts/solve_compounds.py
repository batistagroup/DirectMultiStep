import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from directmultistep.generate import generate_routes
from directmultistep.utils.logging_config import logger
from directmultistep.utils.post_process import find_path_strings_with_commercial_sm
from directmultistep.utils.pre_process import canonicalize_smiles

DATA_PATH = Path(__file__).parent.parent / "data"
CKPT_PATH = DATA_PATH / "checkpoints"
FIG_PATH = DATA_PATH / "figures"
CONFIG_PATH = DATA_PATH / "configs" / "dms_dictionary.yaml"
COMPOUND_PATH = DATA_PATH / "compounds"
EVAL_PATH = DATA_PATH / "evaluations"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, required=True, help="Part of the targets to process")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--use_fp16", action="store_true", help="Whether to use FP16")
    parser.add_argument("--num_part", type=int, required=True, help="Number of parts to split the targets into")
    parser.add_argument("--target_name", type=str, required=True, help="Name of the target dataset")
    args = parser.parse_args()
    part = args.part
    model_name = args.model_name
    use_fp16 = args.use_fp16
    num_part = args.num_part
    target_name = args.target_name

    logger.info(f"part: {part}")
    logger.info(f"model_name: {model_name}")
    logger.info(f"use_fp16: {use_fp16}")
    logger.info(f"num_part: {num_part}")
    logger.info(f"target_name: {target_name}")

    logger.info("Loading targets and stock compounds")
    if target_name == "uspto_190":
        targets = open(COMPOUND_PATH / "uspto_190.txt").read().splitlines()
    elif target_name == "chembl":
        targets = json.load(open(COMPOUND_PATH / "chembl_targets.json"))
    else:
        logger.error(f"{target_name} is not a valid target name")
        raise Exception("Not valid target_name")

    emol_stock_set = set(open(COMPOUND_PATH / "eMolecules.txt").read().splitlines())
    buyables_stock_set = set(open(COMPOUND_PATH / "buyables-stock.txt").read().splitlines())

    chunk_size = len(targets) // num_part
    start_index = (part - 1) * chunk_size
    end_index = part * chunk_size if part < num_part else len(targets)
    targets = targets[start_index:end_index]

    folder_name = f"{target_name}_{model_name}_fp16" if use_fp16 else f"{target_name}_{model_name}"
    save_dir = EVAL_PATH / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    SAVED_PATH = save_dir / f"paths_part_{part}.pkl"
    SAVED_COUNT_PATH = save_dir / f"count_part_{part}.json"

    logger.info("Retrosythesis starting")
    start = time.time()

    all_paths = []
    raw_solved_count = 0
    buyable_solved_count = 0
    emol_solved_count = 0

    for target in tqdm(targets):
        target = canonicalize_smiles(target)
        raw_paths = []
        if model_name == "explorer XL" or model_name == "explorer":
            raw_paths += generate_routes(
                target,
                n_steps=None,
                starting_material=None,
                beam_size=50,
                model=model_name,
                config_path=CONFIG_PATH,
                ckpt_dir=CKPT_PATH,
                commercial_stock=None,
                use_fp16=use_fp16,
            )
        else:
            for step in range(2, 9):
                raw_paths += generate_routes(
                    target,
                    n_steps=step,
                    starting_material=None,
                    beam_size=50,
                    model=model_name,
                    config_path=CONFIG_PATH,
                    ckpt_dir=CKPT_PATH,
                    commercial_stock=None,
                    use_fp16=use_fp16,
                )
        buyables_paths = find_path_strings_with_commercial_sm(raw_paths, commercial_stock=buyables_stock_set)
        emol_paths = find_path_strings_with_commercial_sm(raw_paths, commercial_stock=emol_stock_set)
        if len(raw_paths) > 0:
            raw_solved_count += 1
        if len(buyables_paths) > 0:
            buyable_solved_count += 1
        if len(emol_paths) > 0:
            emol_solved_count += 1
        logger.info(f"Current raw solved count: {raw_solved_count}")
        logger.info(f"Current buyable solved count: {buyable_solved_count}")
        logger.info(f"Current emol solved count: {emol_solved_count}")
        all_paths.append([raw_paths, buyables_paths, emol_paths])

    end = time.time()

    results = {
        "raw_solved_count": raw_solved_count,
        "buyable_solved_count": buyable_solved_count,
        "emol_solved_count": emol_solved_count,
        "time_elapsed": end - start,
    }
    logger.info(f"Results: {results}")
    with open(SAVED_COUNT_PATH, "w") as f:
        json.dump(results, f)
