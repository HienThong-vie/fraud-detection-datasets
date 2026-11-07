from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .config import PROJECT_ROOT
from .data import MODEL_COLUMNS

DEFAULT_MODEL = PROJECT_ROOT / "checkpoint" / "production_model.joblib"


def load_bundle(path: str | Path = DEFAULT_MODEL) -> dict:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")
    return joblib.load(model_path)


def predict_record(record: dict, bundle: dict) -> dict:
    missing = sorted(set(MODEL_COLUMNS) - set(record))
    if missing:
        raise ValueError(f"Input is missing engineered features: {missing}")
    features = pd.DataFrame([record]).loc[:, MODEL_COLUMNS]
    probability = float(bundle["model"].predict_proba(features)[:, 1][0])
    threshold = float(bundle["threshold"])
    return {
        "fraud_probability": probability,
        "threshold": threshold,
        "prediction": int(probability >= threshold),
        "selected_model": bundle["selected_model"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score one engineered Handbook transaction")
    parser.add_argument("json_file")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    args = parser.parse_args(argv)
    record = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
    print(json.dumps(predict_record(record, load_bundle(args.model)), indent=2))


if __name__ == "__main__":
    main()
