from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "training.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["dataset_directory"] = str(PROJECT_ROOT / config["dataset_directory"])
    return config
