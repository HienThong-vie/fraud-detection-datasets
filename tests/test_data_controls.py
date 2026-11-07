import numpy as np

from ai4cyber.config import TrainingConfig
from ai4cyber.data import (
    build_preprocessor,
    dataset_profile,
    make_temporal_data,
    replace_missing_sentinels,
)
from scripts.make_demo_data import make_frame


def test_missing_sentinels_are_profiled_and_cleaned_selectively():
    frame = make_frame(800, seed=23)
    frame.loc[0, "prev_address_months_count"] = -1
    frame.loc[0, "credit_risk_score"] = -1
    frame.loc[1, "intended_balcon_amount"] = -0.25
    frame.loc[2, "intended_balcon_amount"] = -15.0

    profile = dataset_profile(frame, TrainingConfig())
    cleaned = replace_missing_sentinels(frame)

    assert profile["dtypes"]["prev_address_months_count"] == "int64"
    assert profile["sentinel_minus_one_counts"]["prev_address_months_count"] >= 1
    assert profile["minus_one_counts_all_numeric"]["credit_risk_score"] >= 1
    assert np.isnan(cleaned.loc[0, "prev_address_months_count"])
    assert np.isnan(cleaned.loc[1, "intended_balcon_amount"])
    assert np.isnan(cleaned.loc[2, "intended_balcon_amount"])
    assert cleaned.loc[0, "credit_risk_score"] == -1
    assert profile["negative_missing_counts"]["intended_balcon_amount"] >= 2
    assert profile["effective_missing_sentinel_counts"]["intended_balcon_amount"] >= 2


def test_temporal_development_and_test_policy_is_disjoint():
    config = TrainingConfig()
    frame = replace_missing_sentinels(make_frame(1600, seed=31))
    temporal = make_temporal_data(frame, config)

    assert set(config.train_months + config.validation_months) == set(range(6))
    assert set(config.test_months) == {6, 7}
    assert set(temporal.train["month"].unique()) <= set(config.train_months)
    assert set(temporal.validation["month"].unique()) == {5}
    assert set(temporal.test["month"].unique()) == {6, 7}


def test_preprocessor_statistics_use_training_rows_only():
    config = TrainingConfig()
    frame = make_frame(1600, seed=41)
    frame.loc[frame["month"].isin(config.train_months), "name_email_similarity"] = 0.1
    frame.loc[~frame["month"].isin(config.train_months), "name_email_similarity"] = 0.9
    temporal = make_temporal_data(replace_missing_sentinels(frame), config)

    preprocessor = build_preprocessor(temporal)
    preprocessor.fit(temporal.train[temporal.feature_columns])
    imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    feature_index = temporal.numeric_columns.index("name_email_similarity")

    assert imputer.statistics_[feature_index] == 0.1
