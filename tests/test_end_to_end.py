import json
from dataclasses import replace

from ai4cyber.config import TrainingConfig
from ai4cyber.inference import ModelService
from ai4cyber.training import run_training
from scripts.make_demo_data import make_frame


def test_train_and_predict_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    frame = make_frame(4000, seed=7, shift=0.12)
    # Ensure each month contains both target classes.
    for month in range(8):
        indices = frame.index[frame["month"] == month][:2]
        frame.loc[indices[0], "fraud_bool"] = 0
        frame.loc[indices[1], "fraud_bool"] = 1
    frame.to_csv(raw / "Base.csv", index=False)

    config = replace(
        TrainingConfig(),
        dataset_files=["Base.csv"],
        max_train_rows=1200,
        max_validation_rows=400,
        max_test_rows=400,
        tree_estimators=8,
        tree_max_depth=6,
        n_jobs=1,
    )
    card = run_training(
        raw,
        config,
        tmp_path / "artifacts",
        tmp_path / "outputs",
        tmp_path / "processed",
    )
    assert card["test_data_not_used_for_selection"] is True
    assert card["leakage_controls"]["development_months"] == [0, 1, 2, 3, 4, 5]
    assert card["leakage_controls"]["test_months"] == [6, 7]
    assert (tmp_path / "processed" / "base_model_data.csv.gz").exists()
    profile = json.loads((tmp_path / "outputs" / "base" / "dataset_profile.json").read_text())
    assert "dtypes" in profile
    assert "sentinel_minus_one_counts" in profile

    service = ModelService(tmp_path / "artifacts" / "production_model.joblib")
    sample = frame.iloc[0][service.feature_columns].to_dict()
    prediction = service.predict(sample)
    assert 0.0 <= prediction["score"] <= 1.0
    assert prediction["decision_source"] == "trained ML classifier"
    assert prediction["notice"].startswith("Educational decision-support")
