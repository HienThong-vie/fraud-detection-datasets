from pathlib import Path


def test_team_workflow_folders_exist():
    root = Path(__file__).resolve().parents[1]
    required = {
        "checkpoint",
        "dataset",
        "evaluation",
        "preprocess_data",
        "preprocessed_df",
        "training_model",
        "utils",
    }
    assert all((root / name).is_dir() for name in required)
    assert (root / "dataset" / "Base.csv").resolve() == (root / "data" / "raw" / "Base.csv").resolve()
    assert (root / "checkpoint" / "production_model.joblib").suffix == ".joblib"
    assert (root / "preprocessed_df" / "manifest.json").is_file()
    assert (root / "evaluation" / "results").resolve() == (root / "outputs").resolve()

