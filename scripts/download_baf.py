"""Download the BAF dataset from Kaggle."""

from __future__ import annotations

import argparse
from pathlib import Path

DATASET = "sgpjesus/bank-account-fraud-dataset-neurips-2022"
EXPECTED_VERSION = 2
EXPECTED_LICENSE = "CC BY-NC-SA 4.0"
EXPECTED_FILES = [
    "Base.csv",
    "Variant I.csv",
    "Variant II.csv",
    "Variant III.csv",
    "Variant IV.csv",
    "Variant V.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as error:
        raise SystemExit("Install download support first: pip install -e '.[download]'") from error
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as error:
        raise SystemExit(
            "Kaggle authentication failed. Create an API token in Kaggle settings, configure it "
            "as Kaggle documents, then rerun this command."
        ) from error
    metadata = api.dataset_view(DATASET)
    version = int(metadata.currentVersionNumber)
    license_name = str(metadata.licenseName)
    if version != EXPECTED_VERSION or license_name != EXPECTED_LICENSE:
        raise SystemExit(
            f"Dataset metadata changed: version={version}, license={license_name!r}. "
            "Review the upstream record before continuing."
        )
    api.dataset_download_files(DATASET, path=str(args.destination), unzip=True, quiet=False)
    missing = [name for name in EXPECTED_FILES if not (args.destination / name).exists()]
    if missing:
        raise SystemExit(f"Download finished but required files are missing: {missing}")
    print(f"BAF version {version} ({license_name}) is ready in {args.destination}")


if __name__ == "__main__":
    main()
