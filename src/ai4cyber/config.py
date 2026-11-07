"""Training configuration definitions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    dataset_files: list[str] = field(
        default_factory=lambda: [
            "Base.csv",
            "Variant I.csv",
            "Variant II.csv",
            "Variant III.csv",
            "Variant IV.csv",
            "Variant V.csv",
        ]
    )
    target_column: str = "fraud_bool"
    time_column: str = "month"
    train_months: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    validation_months: list[int] = field(default_factory=lambda: [5])
    test_months: list[int] = field(default_factory=lambda: [6, 7])
    protected_features: list[str] = field(
        default_factory=lambda: ["customer_age", "income", "employment_status"]
    )
    excluded_features: list[str] = field(
        default_factory=lambda: ["device_fraud_count", "x1", "x2"]
    )
    max_train_rows: int | None = 250_000
    max_validation_rows: int | None = 50_000
    max_test_rows: int | None = 50_000
    target_false_positive_rate: float = 0.05
    random_seed: int = 42
    tree_estimators: int = 160
    tree_max_depth: int | None = 16
    catboost_iterations: int = 350
    catboost_depth: int = 7
    catboost_learning_rate: float = 0.08
    n_jobs: int = -1
    clustering_sample_rows: int = 30_000
    clustering_k_values: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])

    @classmethod
    def from_json(cls, path: str | Path) -> TrainingConfig:
        with Path(path).open(encoding="utf-8") as handle:
            values: dict[str, Any] = json.load(handle)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
