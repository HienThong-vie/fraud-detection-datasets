import pandas as pd

from handbook_fraud.data import chronological_masks, engineer_prior_features, prepare_splits


def sample_frame() -> pd.DataFrame:
    rows = []
    transaction_id = 0
    start = pd.Timestamp("2018-04-01")
    for day in range(30):
        for index in range(10):
            seconds = day * 86400 + index * 120
            rows.append(
                {
                    "TRANSACTION_ID": transaction_id,
                    "TX_DATETIME": start + pd.to_timedelta(int(seconds), unit="s"),
                    "CUSTOMER_ID": index % 5,
                    "TERMINAL_ID": index % 3,
                    "TX_AMOUNT": float(10 + index + day),
                    "TX_TIME_SECONDS": seconds,
                    "TX_TIME_DAYS": day,
                    "TX_FRAUD": int(index == 0),
                    "TX_FRAUD_SCENARIO": int(index == 0),
                }
            )
            transaction_id += 1
    return pd.DataFrame(rows)


def test_prior_features_do_not_include_current_amount() -> None:
    ordered, features = engineer_prior_features(sample_frame())
    first_customer = ordered["CUSTOMER_ID"] == 0
    customer_features = features.loc[first_customer]
    assert customer_features.iloc[0]["customer_previous_count"] == 0
    assert customer_features.iloc[0]["customer_mean_amount_previous"] == 0
    assert customer_features.iloc[1]["customer_mean_amount_previous"] == 10.0


def test_chronological_masks_do_not_overlap() -> None:
    frame = sample_frame()
    masks = chronological_masks(frame, 0.6, 0.2)
    train_days = frame.loc[masks["train"], "TX_TIME_DAYS"]
    validation_days = frame.loc[masks["validation"], "TX_TIME_DAYS"]
    test_days = frame.loc[masks["test"], "TX_TIME_DAYS"]
    assert train_days.max() < validation_days.min()
    assert validation_days.max() < test_days.min()


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
