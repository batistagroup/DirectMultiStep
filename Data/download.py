""" Module with script to download public data

Adapted from https://github.com/MolecularAI/PaRoutes/blob/main/data/download_data.py
"""
import os
import sys
from pathlib import Path

import requests
import tqdm

FILES_TO_DOWNLOAD = [
    {
        "filename": "n1-routes.json",
        "url": "https://zenodo.org/record/7341155/files/ref_routes_n1.json?download=1",
    },
    {
        "filename": "n1-stock.txt",
        "url": "https://zenodo.org/record/7341155/files/stock_n1.txt?download=1",
    },
    {
        "filename": "n5-routes.json",
        "url": "https://zenodo.org/record/7341155/files/ref_routes_n5.json?download=1",
    },
    {
        "filename": "n5-stock.txt",
        "url": "https://zenodo.org/record/7341155/files/stock_n5.txt?download=1",
    },
    {
        "filename": "all_routes.json.gz",
        "url": "https://zenodo.org/record/7341155/files/all_loaded_routes.json.gz?download=1",
    },
]


def _download_file(url: str, filename: str) -> None:
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        pbar = tqdm.tqdm(
            total=total_size, desc=os.path.basename(filename), unit="B", unit_scale=True
        )
        with open(filename, "wb") as fileobj:
            for chunk in response.iter_content(chunk_size=1024):
                fileobj.write(chunk)
                pbar.update(len(chunk))
        pbar.close()


def main() -> None:
    """Entry-point for CLI"""
    path = Path(__file__).parent / "PaRoutes"
    path.mkdir(parents=True, exist_ok=True)
    for filespec in FILES_TO_DOWNLOAD:
        try:
            _download_file(
                filespec["url"], path / filespec["filename"]
            )
        except requests.HTTPError as err:
            print(f"Download failed with message {str(err)}")
            sys.exit(1)


if __name__ == "__main__":
    main()