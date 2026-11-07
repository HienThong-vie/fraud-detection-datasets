from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class ModelService:
    def __init__(self, artifact_path: str | Path = "artifacts/production_model.joblib"):
        artifact = joblib.load(artifact_path)
        self.model = artifact["model"]
        self.feature_columns = artifact["feature_columns"]
        self.threshold = float(artifact["threshold"])
        self.winner = artifact["winner"]

    def predict(self, features: dict) -> dict:
        unknown = sorted(set(features) - set(self.feature_columns))
        if unknown:
            raise ValueError(f"Unknown model fields: {unknown}")
        row = {column: features.get(column, np.nan) for column in self.feature_columns}
        probabilities = np.asarray(self.model.predict_proba(pd.DataFrame([row])))
        probability = float(probabilities[0, 1])
        return {
            "fraud_score": probability,
            "review_recommended": probability >= self.threshold,
            "threshold": self.threshold,
            "model": self.winner,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/production_model.joblib"))
    args = parser.parse_args()
    features = json.loads(args.input.read_text(encoding="utf-8"))
    print(json.dumps(ModelService(args.model).predict(features), indent=2))


if __name__ == "__main__":
    main()

