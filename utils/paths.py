from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoint"
CONFIG = ROOT / "configs" / "training.json"
DATASET_RAW = ROOT / "dataset"
EVALUATION_RESULTS = ROOT / "evaluation" / "results"
PREPROCESSED_DF = ROOT / "preprocessed_df"

