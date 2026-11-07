from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data import dataset_profile, load_train_data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_profile.json"))
    args = parser.parse_args()
    required = [
        "train_transaction.csv",
        "train_identity.csv",
        "test_transaction.csv",
        "test_identity.csv",
        "sample_submission.csv",
    ]
    missing = [name for name in required if not (args.data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {missing}")
    result = {
        "profile": dataset_profile(load_train_data(args.data_dir)),
        "files": {
            name: {
                "size_bytes": (args.data_dir / name).stat().st_size,
                "sha256": sha256_file(args.data_dir / name),
            }
            for name in required
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["profile"], indent=2))


if __name__ == "__main__":
    main()

