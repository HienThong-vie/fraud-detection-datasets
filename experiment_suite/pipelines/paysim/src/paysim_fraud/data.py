from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

READ_COLUMNS = ["step", "type", "amount", "isFraud", "isFlaggedFraud"]
MODEL_COLUMNS = ["step", "type", "amount", "hour", "day"]
TARGET = "isFraud"


def load_raw(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"PaySim CSV not found at {dataset_path}. Run python scripts/download_paysim.py first."
        )
    frame = pd.read_csv(
        dataset_path,
        usecols=READ_COLUMNS,
        dtype={
            "step": "int16",
            "type": "category",
            "amount": "float32",
            "isFraud": "int8",
            "isFlaggedFraud": "int8",
        },
    )
    validate_raw(frame)
    return frame


def validate_raw(frame: pd.DataFrame) -> None:
    missing = sorted(set(READ_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"PaySim dataset is missing columns: {missing}")
    labels = set(frame[TARGET].dropna().astype(int).unique())
    if not labels.issubset({0, 1}) or not labels:
        raise ValueError("isFraud must contain binary labels")


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"step", "type", "amount"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")
    features = frame.loc[:, ["step", "type", "amount"]].copy()
    features["hour"] = ((features["step"].astype("int32") - 1) % 24).astype("int8")
    features["day"] = ((features["step"].astype("int32") - 1) // 24).astype("int16")
    return features.loc[:, MODEL_COLUMNS]


def chronological_split(
    frame: pd.DataFrame, train_fraction: float, validation_fraction: float
) -> dict[str, pd.DataFrame]:
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Split fractions must be positive and sum to less than one")
    steps = np.sort(frame["step"].unique())
    train_index = max(1, int(len(steps) * train_fraction))
    validation_index = max(train_index + 1, int(len(steps) * (train_fraction + validation_fraction)))
    train_end = steps[min(train_index - 1, len(steps) - 1)]
    validation_end = steps[min(validation_index - 1, len(steps) - 1)]
    return {
        "train": frame.loc[frame["step"] <= train_end].copy(),
        "validation": frame.loc[(frame["step"] > train_end) & (frame["step"] <= validation_end)].copy(),
        "test": frame.loc[frame["step"] > validation_end].copy(),
    }


def limit_rows(frame: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    if maximum is None or len(frame) <= maximum:
        return frame.copy()
    selected, _ = train_test_split(
        frame,
        train_size=maximum,
        random_state=seed,
        stratify=frame[TARGET],
    )
    return selected.sort_values("step", kind="stable").copy()


def prepare_splits(frame: pd.DataFrame, config: dict) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    raw_splits = chronological_split(
        frame,
        train_fraction=float(config["train_fraction"]),
        validation_fraction=float(config["validation_fraction"]),
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
    return prepared


def save_processed(
    splits: dict[str, tuple[pd.DataFrame, pd.Series]], destination: str | Path
) -> Path:
    frames = []
    for name, (features, target) in splits.items():
        part = features.copy()
        part[TARGET] = target.to_numpy()
        part["_split"] = name
        frames.append(part)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output, index=False, compression="gzip")
    return output
