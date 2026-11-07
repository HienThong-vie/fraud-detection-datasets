"""Model loading and single-record inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .constants import MISSING_SENTINEL_COLUMNS
from .decision import make_review_decision


class ModelService:
    def __init__(self, artifact_path: str | Path):
        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {self.artifact_path}. Run the training command first."
            )
        self.artifact: dict = joblib.load(self.artifact_path)
        self.pipeline = self.artifact["pipeline"]
        self.threshold = float(self.artifact["threshold"])
        self.feature_columns = list(self.artifact["feature_columns"])

    def _frame(self, features: dict[str, Any]) -> pd.DataFrame:
        missing = [column for column in self.feature_columns if column not in features]
        if missing:
            raise ValueError(f"Missing required model fields: {missing}")
        unknown = sorted(set(features) - set(self.feature_columns))
        if unknown:
            raise ValueError(f"Unknown model fields: {unknown}")
        frame = pd.DataFrame([{column: features[column] for column in self.feature_columns}])
        sentinel_columns = set(
            self.artifact.get("missing_sentinel_columns", MISSING_SENTINEL_COLUMNS)
        )
        for column in sentinel_columns & set(frame.columns):
            frame[column] = frame[column].mask(frame[column].eq(-1))
        return frame

    def _local_ablation_factors(self, frame: pd.DataFrame, base_score: float) -> list[dict]:
        """Estimate local feature effects with one-field ablation."""
        factors: list[dict] = []
        for column in self.feature_columns:
            changed = frame.copy()
            changed.loc[0, column] = np.nan
            try:
                ablated_score = float(self.pipeline.predict_proba(changed)[0, 1])
            except (TypeError, ValueError):
                continue
            factors.append(
                {
                    "feature": column,
                    "submitted_value": str(frame.loc[0, column]),
                    "score_change_vs_missing": float(base_score - ablated_score),
                    "direction": "higher" if base_score > ablated_score else "lower",
                }
            )
        return sorted(factors, key=lambda item: abs(item["score_change_vs_missing"]), reverse=True)[
            :5
        ]

    def predict(self, features: dict[str, Any]) -> dict:
        frame = self._frame(features)
        score = float(self.pipeline.predict_proba(frame)[0, 1])
        return {
            **make_review_decision(score, self.threshold),
            "dataset": self.artifact["dataset"],
            "model": self.artifact["model_name"],
            "leading_factors": self._local_ablation_factors(frame, score),
            "factor_note": (
                "Local ablation associations, not causal reasons: each field is replaced by the "
                "training-time imputation value and the score change is measured."
            ),
            "notice": self.artifact["notice"],
        }

    def metadata(self) -> dict:
        return {
            "dataset": self.artifact["dataset"],
            "model": self.artifact["model_name"],
            "threshold": self.threshold,
            "required_features": self.feature_columns,
            "protected_features_excluded": self.artifact["protected_features_excluded"],
            "global_feature_importance": self.artifact["global_feature_importance"],
            "notice": self.artifact["notice"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/production_model.joblib"))
    parser.add_argument("--input", type=Path, required=True, help="JSON object of model features")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    print(json.dumps(ModelService(args.model).predict(payload), indent=2))


if __name__ == "__main__":
    main()
