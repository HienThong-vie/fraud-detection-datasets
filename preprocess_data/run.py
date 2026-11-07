from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ieee_fraud.config import TrainingConfig
from ieee_fraud.data import load_train_data, sampled_splits
from utils.paths import CONFIG, DATASET_RAW, EVALUATION_RESULTS, PREPROCESSED_DF


def main() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), environment.get("PYTHONPATH", "")]
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ieee_fraud.profiling",
            "--data-dir",
            str(DATASET_RAW),
            "--output",
            str(EVALUATION_RESULTS / "dataset_profile.json"),
        ],
        check=True,
        env=environment,
    )
    config = TrainingConfig.from_json(CONFIG)
    splits = sampled_splits(load_train_data(DATASET_RAW), config)
    prepared = pd.concat(
        [
            splits.train.assign(_split="train"),
            splits.validation.assign(_split="validation"),
            splits.test.assign(_split="test"),
        ],
        ignore_index=True,
    )
    PREPROCESSED_DF.mkdir(parents=True, exist_ok=True)
    output = PREPROCESSED_DF / "ieee_model_data.csv.gz"
    prepared.to_csv(output, index=False, compression="gzip")
    manifest = {
        "purpose": "Train-ready sampled IEEE-CIS dataframe",
        "file": output.name,
        "size_bytes": output.stat().st_size,
        "rows": len(prepared),
        "columns": len(prepared.columns),
        "split_counts": prepared["_split"].value_counts().to_dict(),
    }
    (PREPROCESSED_DF / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

