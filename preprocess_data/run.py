from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
            "ai4cyber.profiling",
            "--data-dir",
            str(DATASET_RAW),
            "--config",
            str(CONFIG),
            "--output",
            str(EVALUATION_RESULTS / "profiling"),
        ],
        check=True,
        env=environment,
    )
    files = sorted(PREPROCESSED_DF.glob("*.csv.gz"))
    manifest = {
        "purpose": "Train-ready sampled BAF dataframes",
        "files": [
            {"name": path.name, "size_bytes": path.stat().st_size}
            for path in files
            if path.exists()
        ],
    }
    (PREPROCESSED_DF / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

