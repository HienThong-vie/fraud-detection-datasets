from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .config import PROJECT_ROOT, load_config
from .data import load_raw


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_profile(frame, dataset_path: Path) -> dict:
    return {
        "source": "https://www.kaggle.com/datasets/ealaxi/paysim1",
        "license": "CC BY-SA 4.0",
        "file": dataset_path.name,
        "file_size_bytes": dataset_path.stat().st_size,
        "sha256": sha256(dataset_path),
        "rows": len(frame),
        "columns_loaded": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "fraud_rows": int(frame["isFraud"].sum()),
        "fraud_rate": float(frame["isFraud"].mean()),
        "flagged_rows": int(frame["isFlaggedFraud"].sum()),
        "step_min": int(frame["step"].min()),
        "step_max": int(frame["step"].max()),
        "transaction_types": {
            str(key): int(value) for key, value in frame["type"].value_counts().items()
        },
        "missing_values": {column: int(value) for column, value in frame.isna().sum().items()},
        "leakage_controls": [
            "isFlaggedFraud is excluded from model inputs",
            "origin and destination balances are excluded",
            "raw account identifiers are excluded",
            "splits are chronological by step",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Profile the PaySim dataset")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "training.json"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "dataset_profile.json"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    dataset_path = Path(config["dataset_path"])
    profile = build_profile(load_raw(dataset_path), dataset_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
