from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ai4cyber.config import TrainingConfig
from ai4cyber.training import run_training
from utils.paths import (
    CHECKPOINT,
    CONFIG,
    DATASET_RAW,
    EVALUATION_RESULTS,
    PREPROCESSED_DF,
)


def main() -> None:
    card = run_training(
        DATASET_RAW,
        TrainingConfig.from_json(CONFIG),
        CHECKPOINT,
        EVALUATION_RESULTS,
        PREPROCESSED_DF,
    )
    print(json.dumps({"datasets": len(card["datasets"]), "model": card["production_artifact"]}, indent=2))


if __name__ == "__main__":
    main()

