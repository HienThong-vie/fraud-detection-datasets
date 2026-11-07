from __future__ import annotations

import joblib
import pandas as pd

from ieee_fraud.inference import ModelService


class ConstantModel:
    def predict_proba(self, frame: pd.DataFrame):
        return [[0.75, 0.25] for _ in range(len(frame))]


def test_model_service_accepts_partial_input(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model": ConstantModel(),
            "feature_columns": ["TransactionAmt", "ProductCD"],
            "threshold": 0.2,
            "winner": "test_model",
        },
        path,
    )
    result = ModelService(path).predict({"TransactionAmt": 10.0})
    assert result["fraud_score"] == 0.25
    assert result["review_recommended"] is True

