import pandas as pd

from fraudbench.data import MODEL_COLUMNS, chronological_split, make_features


def test_chronological_split_has_no_time_overlap() -> None:
    frame = pd.DataFrame(
        {
            "step": range(1, 101),
            "amount": range(1, 101),
            "age": ["3"] * 100,
            "gender": ["F"] * 100,
            "category": ["food"] * 100,
            "fraud": [0] * 95 + [1] * 5,
        }
    )
    splits = chronological_split(frame, 0.6, 0.2)
    assert splits["train"]["step"].max() < splits["validation"]["step"].min()
    assert splits["validation"]["step"].max() < splits["test"]["step"].min()
    assert list(make_features(frame).columns) == MODEL_COLUMNS
