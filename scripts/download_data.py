from __future__ import annotations

import shutil
import ssl
import urllib.request
from pathlib import Path

import certifi

# The canonical Kaggle page currently requires a Kaggle login. This public research mirror contains
# the same named BankSim CSV; the profile records its SHA-256 after download.
URL = "https://raw.githubusercontent.com/niravdedhiya/Fraud-Detection-BankSim/main/data/bs140513_032310.csv"
DESTINATION = Path(__file__).resolve().parents[1] / "data" / "raw" / "bs140513_032310.csv"


if __name__ == "__main__":
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(URL, context=context) as response, DESTINATION.open("wb") as output:
        shutil.copyfileobj(response, output)
    print(f"Downloaded {DESTINATION}")
