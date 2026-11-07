from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "is_illicit"
NUMERIC_COLUMNS = [f"feature_{index}" for index in range(1, 166)]
CATEGORICAL_COLUMNS: list[str] = []
MODEL_COLUMNS = NUMERIC_COLUMNS


def load_file(features_path: str | Path, classes_path: str | Path) -> pd.DataFrame:
    feature_names = ["tx_id", "time_step"] + NUMERIC_COLUMNS
    features = pd.read_csv(features_path, header=None, names=feature_names)
    classes = pd.read_csv(classes_path)
    classes.columns = ["tx_id", "class"]
    classes["class"] = classes["class"].astype("string")
    frame = features.merge(classes, on="tx_id", how="inner", validate="one_to_one")
    frame = frame.loc[frame["class"].isin(["1", "2"])].copy()
    frame[TARGET] = (frame["class"] == "1").astype("int8")
    if frame.empty or set(frame[TARGET].unique()) - {0, 1}:
        raise ValueError("Elliptic labelled rows could not be loaded")
    return frame.sort_values(["time_step", "tx_id"], kind="stable").reset_index(drop=True)


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, MODEL_COLUMNS].copy()


def limit_rows(frame: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    if maximum is None or len(frame) <= maximum:
        return frame.copy()
    selected, _ = train_test_split(
        frame, train_size=maximum, random_state=seed, stratify=frame[TARGET]
    )
    return selected.sort_values(["time_step", "tx_id"], kind="stable")


def temporal_split(frame: pd.DataFrame, train_last_step: int, validation_last_step: int):
    return {
        "train": frame.loc[frame["time_step"] <= train_last_step].copy(),
        "validation": frame.loc[
            (frame["time_step"] > train_last_step) & (frame["time_step"] <= validation_last_step)
        ].copy(),
        "test": frame.loc[frame["time_step"] > validation_last_step].copy(),
    }


def load_splits(config: dict) -> tuple[dict[str, tuple[pd.DataFrame, pd.Series]], pd.DataFrame]:
    frame = load_file(config["features_path"], config["classes_path"])
    raw_splits = temporal_split(
        frame, int(config["train_last_step"]), int(config["validation_last_step"])
    )
    limits = {
        "train": config.get("max_train_rows"),
        "validation": config.get("max_validation_rows"),
        "test": config.get("max_test_rows"),
    }
    prepared = {}
    for offset, (name, split) in enumerate(raw_splits.items()):
        if split[TARGET].nunique() < 2:
            raise ValueError(f"{name} split must contain both licit and illicit labels")
        limited = limit_rows(split, limits[name], int(config["random_seed"]) + offset)
        prepared[name] = (make_features(limited), limited[TARGET].astype("int8"))
    return prepared, frame


def profile(frame: pd.DataFrame, splits: dict) -> dict:
    return {
        "labelled_rows": len(frame),
        "columns": len(frame.columns),
        "illicit_rows": int(frame[TARGET].sum()),
        "illicit_rate": float(frame[TARGET].mean()),
        "missing_values": {key: int(value) for key, value in frame.isna().sum().items()},
        "time_step_min": int(frame["time_step"].min()),
        "time_step_max": int(frame["time_step"].max()),
        "split_rows": {name: len(values[0]) for name, values in splits.items()},
        "baseline_note": "This baseline uses node features only; graph edges are not used.",
    }
