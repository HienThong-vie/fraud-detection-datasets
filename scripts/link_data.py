from __future__ import annotations

import argparse
from pathlib import Path

REQUIRED = [
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
    "sample_submission.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Link IEEE-CIS CSV files without duplicating 1.3 GB")
    parser.add_argument("source", type=Path)
    parser.add_argument("--destination", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    missing = [name for name in REQUIRED if not (args.source / name).is_file()]
    if missing:
        raise SystemExit(f"Source folder is missing: {missing}")
    args.destination.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        link = args.destination / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to((args.source / name).resolve())
        print(f"{link} -> {link.resolve()}")


if __name__ == "__main__":
    main()

