from pathlib import Path

import pandas as pd
from tqdm import tqdm

from directmultistep.utils.logging_config import logger
from directmultistep.utils.pre_process import canonicalize_smiles

DATA_PATH = Path(__file__).parent
COMPOUND_PATH = DATA_PATH / "compounds"

if __name__ == "__main__":
    logger.info("Loading eMolecules csv...")
    # `origin_dict.csv` from `github.com/binghong-ml/retro_star`
    emol_df = pd.read_csv(COMPOUND_PATH / "origin_dict.csv", index_col=0)
    emol_smiles = emol_df["mol"].tolist()
    logger.info(f"Number of eMolecules smiles: {len(emol_smiles)}")
    del emol_df

    logger.info("Canonicalizing SMILES strings...")
    canonicalized_smiles = []
    for smiles in tqdm(emol_smiles):
        try:
            canonicalized_smiles.append(canonicalize_smiles(smiles))
        except ValueError as e:
            logger.error(f"Error canonicalizing SMILES '{smiles}': {e} during canonicalizing buyables")

    logger.info("Saving unique canonicalized SMILES strings...")
    unique_smiles = list(set(canonicalized_smiles))
    logger.info(f"Number of unique eMolecules smiles: {len(unique_smiles)}")
    with open(COMPOUND_PATH / "eMolecules.txt", "w") as f:
        for smiles in unique_smiles:
            f.write(smiles + "\n")
