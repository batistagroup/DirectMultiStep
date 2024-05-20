# Preparing Dataset

This model is trained on the [PaRoutes](https://github.com/MolecularAI/PaRoutes) dataset. To get started, please:

1. Run `python download.py`. This creates a `PaRoutes` folder and downloads `n1-routes.json` (64.2 MB), `n1-stock.txt` (0.4 MB), `n5-routes.json` (82.1 MB), `n5-stock.txt` (0.4 MB), and `all_routes.json` (1.44 GB).
2. Run `python process.py`. This takes roughly 7 minutes and it creates pickle files in `Processed` folder.

Notably [Processed](/Data/Processed/) folder also contains the [character_dictionary.yaml](/Data/Processed/character_dictionary.yaml) which contains tokenToIdx, idxToToken dictionaries, as well as maximal sequence lengths.
