from __future__ import annotations

import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi

BASE = "https://data.pyg.org/datasets/elliptic"
FILES = ["elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"]
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    for filename in FILES:
        archive = RAW / f"{filename}.zip"
        with (
            urllib.request.urlopen(f"{BASE}/{filename}.zip", context=context) as response,
            archive.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(RAW)
        archive.unlink()
        print(f"Downloaded {RAW / filename}")
