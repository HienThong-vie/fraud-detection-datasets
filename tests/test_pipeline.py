from __future__ import annotations

import numpy as np
import pandas as pd

from ieee_fraud.config import TrainingConfig
from ieee_fraud.data import chronological_split, normalize_identity_columns
from ieee_fraud.features import TARGET, TIME_COLUMN, TRANSACTION_ID
from ieee_fraud.metrics import binary_metrics, threshold_at_fpr


def test_test_identity_names_are_normalized():
    frame = pd.DataFrame({"TransactionID": [1], "id-01": [-5], "id-38": ["T"]})
    normalized = normalize_identity_columns(frame)
    assert "id_01" in normalized
    assert "id_38" in normalized
    assert "id-01" not in normalized


def test_chronological_split_keeps_future_rows_out_of_training():
    frame = pd.DataFrame(
        {
            TRANSACTION_ID: np.arange(100),
            TIME_COLUMN: np.arange(100) * 100,
            TARGET: [0, 1] * 50,
        }
    )
    split = chronological_split(frame, TrainingConfig())
    assert len(split.train) == 70
    assert len(split.validation) == 15
    assert len(split.test) == 15
    assert split.train[TIME_COLUMN].max() < split.validation[TIME_COLUMN].min()
    assert split.validation[TIME_COLUMN].max() < split.test[TIME_COLUMN].min()


def test_metrics_and_fpr_threshold_are_consistent():
    target = np.array([0, 0, 0, 1, 1, 1])
    probability = np.array([0.01, 0.1, 0.2, 0.6, 0.8, 0.95])
    threshold = threshold_at_fpr(target, probability, 0.05)
    result = binary_metrics(target, probability, threshold)
    assert result["false_positive_rate"] <= 0.05
    assert result["recall"] == 1.0

