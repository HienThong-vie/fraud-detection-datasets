import pandas as pd

from fraudbench.data import MODEL_COLUMNS, make_features


def test_feature_engineering_removes_identifiers() -> None:
    frame = pd.DataFrame(
        {
            "trans_date_trans_time": ["2020-01-01 02:00:00"],
            "dob": ["1990-01-01"],
            "amt": [100.0],
            "city_pop": [1000],
            "lat": [0.0],
            "long": [0.0],
            "merch_lat": [1.0],
            "merch_long": [1.0],
            "category": ["shopping_net"],
            "gender": ["F"],
        }
    )
    result = make_features(frame)
    assert list(result.columns) == MODEL_COLUMNS
    assert result.loc[0, "distance_km"] > 0
    assert result.loc[0, "hour"] == 2
