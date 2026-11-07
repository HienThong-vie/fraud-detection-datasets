import pandas as pd

from fraudbench.data import MODEL_COLUMNS, temporal_split


def test_temporal_split_has_no_step_overlap() -> None:
    frame = pd.DataFrame(
        {
            "time_step": range(1, 50),
            "tx_id": range(1, 50),
            "is_illicit": [index % 2 for index in range(49)],
            **{column: [0.0] * 49 for column in MODEL_COLUMNS},
        }
    )
    splits = temporal_split(frame, 34, 41)
    assert splits["train"]["time_step"].max() == 34
    assert splits["validation"]["time_step"].min() == 35
    assert splits["test"]["time_step"].min() == 42
