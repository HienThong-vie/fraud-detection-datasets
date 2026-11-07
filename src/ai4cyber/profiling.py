"""Create reproducible BAF dataset profiles and comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import TrainingConfig
from .constants import BAF_VARIANT_PURPOSES
from .data import dataset_profile, load_baf_csv

KAGGLE_DATASET = "sgpjesus/bank-account-fraud-dataset-neurips-2022"
KAGGLE_VERSION = 2
KAGGLE_UPDATED = "2023-11-29T11:32:12.037Z"
DATA_LICENSE = "CC BY-NC-SA 4.0"
SOURCE_URL = "https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022"
PROJECT_URL = "https://github.com/feedzai/bank-account-fraud"


def _json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profile_file(path: Path, config: TrainingConfig) -> dict:
    frame = load_baf_csv(path, config)
    profile = dataset_profile(frame, config)
    profile.update(
        {
            "dataset": path.stem,
            "file": path.name,
            "file_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "purpose": BAF_VARIANT_PURPOSES.get(path.stem, "Unspecified BAF variant"),
        }
    )
    return profile


def _comparison_rows(profiles: list[dict]) -> list[dict]:
    base = next(profile for profile in profiles if profile["dataset"] == "Base")
    base_dtypes = base["dtypes"]
    rows = []
    for profile in profiles:
        dtype_differences = sorted(
            column
            for column in set(base_dtypes) | set(profile["dtypes"])
            if base_dtypes.get(column) != profile["dtypes"].get(column)
        )
        rows.append(
            {
                "dataset": profile["dataset"],
                "purpose": profile["purpose"],
                "rows": profile["rows"],
                "row_difference_vs_base": profile["rows"] - base["rows"],
                "columns": profile["columns"],
                "column_difference_vs_base": profile["columns"] - base["columns"],
                "fraud_rate": profile["fraud_rate"],
                "fraud_rate_difference_vs_base": profile["fraud_rate"] - base["fraud_rate"],
                "sentinel_minus_one_total": profile["sentinel_minus_one_total"],
                "sentinel_difference_vs_base": (
                    profile["sentinel_minus_one_total"] - base["sentinel_minus_one_total"]
                ),
                "dtype_differences_vs_base": ", ".join(dtype_differences),
                "missing_expected_columns": ", ".join(profile["missing_expected_columns"]),
                "additional_columns": ", ".join(profile["additional_columns"]),
            }
        )
    return rows


def run_profiling(data_dir: Path, config: TrainingConfig, output_dir: Path) -> dict:
    missing = [name for name in config.dataset_files if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing BAF files: {missing}")

    profiles = [profile_file(data_dir / name, config) for name in config.dataset_files]
    output_dir.mkdir(parents=True, exist_ok=True)
    for profile in profiles:
        safe_name = profile["dataset"].lower().replace(" ", "_")
        _json_dump(profile, output_dir / f"{safe_name}_profile.json")

    comparison = pd.DataFrame(_comparison_rows(profiles))
    comparison.to_csv(output_dir / "dataset_comparison.csv", index=False)
    manifest = {
        "source": {
            "kaggle_dataset": KAGGLE_DATASET,
            "kaggle_version": KAGGLE_VERSION,
            "kaggle_updated": KAGGLE_UPDATED,
            "license": DATA_LICENSE,
            "source_url": SOURCE_URL,
            "project_url": PROJECT_URL,
        },
        "profiles": profiles,
    }
    _json_dump(manifest, output_dir / "profile_manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("configs/training.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/profiling"))
    args = parser.parse_args()
    result = run_profiling(args.data_dir, TrainingConfig.from_json(args.config), args.output)
    print(json.dumps(result["source"], indent=2))


if __name__ == "__main__":
    main()
