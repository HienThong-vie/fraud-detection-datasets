import pandas as pd

from paysim_fraud.data import chronological_split, make_features, prepare_splits


def sample_frame() -> pd.DataFrame:
    rows = []
    for step in range(1, 31):
        for index in range(10):
            rows.append(
                {
                    "step": step,
                    "type": "TRANSFER" if index % 2 else "PAYMENT",
                    "amount": float(step * 10 + index),
                    "isFraud": int(index == 0),
                    "isFlaggedFraud": 0,
                }
            )
    return pd.DataFrame(rows)


def test_feature_columns_and_time_values() -> None:
    features = make_features(sample_frame())
    assert list(features.columns) == ["step", "type", "amount", "hour", "day"]
    assert features.iloc[0]["hour"] == 0
    assert features.iloc[0]["day"] == 0


def test_chronological_split_has_no_time_overlap() -> None:
    split = chronological_split(sample_frame(), 0.6, 0.2)
    assert split["train"]["step"].max() < split["validation"]["step"].min()
    assert split["validation"]["step"].max() < split["test"]["step"].min()


def test_prepare_splits_keeps_both_classes() -> None:
    config = {
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "max_train_rows": 100,
        "max_validation_rows": 50,
        "max_test_rows": 50,
        "random_seed": 42,
    }
    splits = prepare_splits(sample_frame(), config)
    assert all(set(target.unique()) == {0, 1} for _, target in splits.values())
