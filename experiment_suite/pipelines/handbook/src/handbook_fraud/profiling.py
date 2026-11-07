from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .config import PROJECT_ROOT, load_config
from .data import load_raw


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_profile(frame, dataset_directory: Path) -> dict:
    files = sorted(dataset_directory.glob("*.pkl"))
    return {
        "source": "https://github.com/Fraud-Detection-Handbook/simulated-data-raw",
        "source_note": "Official daily simulated transaction files; raw-data repository has no explicit licence file.",
        "files": [
            {"name": path.name, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in files
        ],
        "rows": len(frame),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "fraud_rows": int(frame["TX_FRAUD"].sum()),
        "fraud_rate": float(frame["TX_FRAUD"].mean()),
        "date_min": frame["TX_DATETIME"].min().isoformat(),
        "date_max": frame["TX_DATETIME"].max().isoformat(),
        "customers": int(frame["CUSTOMER_ID"].nunique()),
        "terminals": int(frame["TERMINAL_ID"].nunique()),
        "fraud_scenarios": {
            str(key): int(value)
            for key, value in frame["TX_FRAUD_SCENARIO"].value_counts().sort_index().items()
        },
        "missing_values": {column: int(value) for column, value in frame.isna().sum().items()},
        "leakage_controls": [
            "TX_FRAUD_SCENARIO is excluded from model inputs",
            "raw customer and terminal IDs are excluded from model inputs",
            "behavioral features use only earlier transactions",
            "splits are chronological by TX_TIME_DAYS",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Profile the Handbook simulated dataset")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "training.json"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "dataset_profile.json"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    directory = Path(config["dataset_directory"])
    profile = build_profile(load_raw(directory), directory)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
