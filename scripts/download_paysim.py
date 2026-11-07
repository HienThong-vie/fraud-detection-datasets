from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "data" / "raw"
EXPECTED_FILE = DESTINATION / "PS_20174392719_1491204439457_log.csv"


def main() -> None:
    if EXPECTED_FILE.exists():
        print(f"Dataset already exists: {EXPECTED_FILE}")
        return
    kaggle_command = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    if not Path(kaggle_command).exists():
        raise SystemExit("Kaggle CLI is not installed. Run: pip install -e '.[dev]'")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            kaggle_command,
            "datasets",
            "download",
            "-d",
            "ealaxi/paysim1",
            "-p",
            str(DESTINATION),
            "--unzip",
        ],
        check=True,
    )
    if not EXPECTED_FILE.exists():
        raise SystemExit(f"Download completed but the expected CSV was not found: {EXPECTED_FILE}")
    print(f"Downloaded {EXPECTED_FILE}")


if __name__ == "__main__":
    main()
