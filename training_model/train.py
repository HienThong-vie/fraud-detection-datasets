from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ieee_fraud.config import TrainingConfig
from ieee_fraud.training import run_training
from utils.paths import CHECKPOINT, CONFIG, DATASET_RAW, EVALUATION_RESULTS


def main() -> None:
    config = replace(TrainingConfig.from_json(CONFIG), data_dir=str(DATASET_RAW))
    card = run_training(config, CHECKPOINT, EVALUATION_RESULTS, create_submission=True)
    print(json.dumps({"winner": card["winner"], "test_metrics": card["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()

