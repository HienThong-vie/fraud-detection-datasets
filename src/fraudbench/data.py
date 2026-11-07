from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "fraud"
NUMERIC_COLUMNS = ["step", "amount"]
CATEGORICAL_COLUMNS = ["age", "gender", "category"]
MODEL_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def prepare_catboost_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame[MODEL_COLUMNS].copy()
    for column in CATEGORICAL_COLUMNS:
        prepared[column] = prepared[column].astype("string").fillna("__MISSING__").astype(str)
    return prepared
READ_COLUMNS = ["step", "age", "gender", "category", "amount", TARGET]


def load_file(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"BankSim CSV not found: {dataset_path}")
    frame = pd.read_csv(dataset_path, usecols=READ_COLUMNS)
    frame.columns = [column.strip().strip("'") for column in frame.columns]
    for column in ("age", "gender", "category"):
        frame[column] = frame[column].astype("string").str.strip().str.strip("'")
    frame["step"] = pd.to_numeric(frame["step"], errors="raise")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise").astype("int8")
    if set(frame[TARGET].unique()) - {0, 1}:
        raise ValueError("fraud must be binary")
    return frame.sort_values("step", kind="stable").reset_index(drop=True)


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, MODEL_COLUMNS].copy()


def limit_rows(frame: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    if maximum is None or len(frame) <= maximum:
        return frame.copy()
    selected, _ = train_test_split(
        frame, train_size=maximum, random_state=seed, stratify=frame[TARGET]
    )
    return selected.sort_values("step", kind="stable")


def chronological_split(frame: pd.DataFrame, train_fraction: float, validation_fraction: float):
    steps = np.sort(frame["step"].unique())
    train_cut = steps[max(0, int(len(steps) * train_fraction) - 1)]
    validation_cut = steps[max(0, int(len(steps) * (train_fraction + validation_fraction)) - 1)]
    return {
        "train": frame.loc[frame["step"] <= train_cut].copy(),
        "validation": frame.loc[
            (frame["step"] > train_cut) & (frame["step"] <= validation_cut)
        ].copy(),
        "test": frame.loc[frame["step"] > validation_cut].copy(),
    }


def load_splits(config: dict) -> tuple[dict[str, tuple[pd.DataFrame, pd.Series]], pd.DataFrame]:
    frame = load_file(config["dataset_path"])
    raw_splits = chronological_split(
        frame, float(config["train_fraction"]), float(config["validation_fraction"])
    )
    limits = {
        "train": config.get("max_train_rows"),
        "validation": config.get("max_validation_rows"),
        "test": config.get("max_test_rows"),
    }
    prepared = {}
    for offset, (name, split) in enumerate(raw_splits.items()):
        limited = limit_rows(split, limits[name], int(config["random_seed"]) + offset)
        prepared[name] = (make_features(limited), limited[TARGET].astype("int8"))
    return prepared, frame


def profile(frame: pd.DataFrame, splits: dict) -> dict:
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "fraud_rows": int(frame[TARGET].sum()),
        "fraud_rate": float(frame[TARGET].mean()),
        "missing_values": {key: int(value) for key, value in frame.isna().sum().items()},
        "step_min": int(frame["step"].min()),
        "step_max": int(frame["step"].max()),
        "split_rows": {name: len(values[0]) for name, values in splits.items()},
    }
