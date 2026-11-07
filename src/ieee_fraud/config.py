from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: str = "data/raw"
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    max_train_rows: int | None = 120_000
    max_validation_rows: int | None = 40_000
    max_test_rows: int | None = 40_000
    random_seed: int = 42
    target_false_positive_rate: float = 0.05
    hist_gradient_boosting_iterations: int = 120
    xgboost_estimators: int = 500
    xgboost_max_depth: int = 7
    xgboost_learning_rate: float = 0.05
    n_jobs: int = 4

    @classmethod
    def from_json(cls, path: str | Path) -> TrainingConfig:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        config = cls(**values)
        if config.train_fraction <= 0 or config.validation_fraction <= 0:
            raise ValueError("Split fractions must be positive")
        if config.train_fraction + config.validation_fraction >= 1:
            raise ValueError("Train and validation fractions must leave a held-out test period")
        return config

    def to_dict(self) -> dict:
        return asdict(self)

