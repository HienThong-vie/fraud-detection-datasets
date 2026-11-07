from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "is_fraud"
NUMERIC_COLUMNS = ["amt", "city_pop", "age", "hour", "weekday", "distance_km"]
CATEGORICAL_COLUMNS = ["category", "gender"]
MODEL_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
READ_COLUMNS = [
    "trans_date_trans_time",
    "category",
    "amt",
    "gender",
    "city_pop",
    "dob",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "unix_time",
    TARGET,
]


def prepare_catboost_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame[MODEL_COLUMNS].copy()
    for column in CATEGORICAL_COLUMNS:
        prepared[column] = prepared[column].astype("string").fillna("__MISSING__").astype(str)
    return prepared


def load_file(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Sparkov CSV not found: {dataset_path}")
    frame = pd.read_csv(dataset_path, usecols=READ_COLUMNS)
    if set(frame[TARGET].dropna().astype(int).unique()) - {0, 1}:
        raise ValueError("is_fraud must be binary")
    return frame.sort_values("unix_time", kind="stable").reset_index(drop=True)


def _haversine_km(frame: pd.DataFrame) -> np.ndarray:
    lat1 = np.radians(frame["lat"].to_numpy(dtype=float))
    lon1 = np.radians(frame["long"].to_numpy(dtype=float))
    lat2 = np.radians(frame["merch_lat"].to_numpy(dtype=float))
    lon2 = np.radians(frame["merch_long"].to_numpy(dtype=float))
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    value = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    transaction_time = pd.to_datetime(frame["trans_date_trans_time"], errors="coerce")
    birth_date = pd.to_datetime(frame["dob"], errors="coerce")
    features = pd.DataFrame(index=frame.index)
    features["amt"] = pd.to_numeric(frame["amt"], errors="coerce")
    features["city_pop"] = pd.to_numeric(frame["city_pop"], errors="coerce")
    features["age"] = ((transaction_time - birth_date).dt.days / 365.25).clip(18, 100)
    features["hour"] = transaction_time.dt.hour
    features["weekday"] = transaction_time.dt.dayofweek
    features["distance_km"] = _haversine_km(frame)
    features["category"] = frame["category"].astype("string")
    features["gender"] = frame["gender"].astype("string")
    return features.loc[:, MODEL_COLUMNS]


def limit_rows(frame: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    if maximum is None or len(frame) <= maximum:
        return frame.copy()
    selected, _ = train_test_split(
        frame, train_size=maximum, random_state=seed, stratify=frame[TARGET]
    )
    return selected.sort_values("unix_time", kind="stable")


def load_splits(config: dict) -> tuple[dict[str, tuple[pd.DataFrame, pd.Series]], pd.DataFrame]:
    development = load_file(config["train_path"])
    test_raw = load_file(config["test_path"])
    cut = int(len(development) * (1 - float(config["validation_fraction"])))
    raw_splits = {
        "train": development.iloc[:cut].copy(),
        "validation": development.iloc[cut:].copy(),
        "test": test_raw,
    }
    limits = {
        "train": config.get("max_train_rows"),
        "validation": config.get("max_validation_rows"),
        "test": config.get("max_test_rows"),
    }
    prepared = {}
    for offset, (name, split) in enumerate(raw_splits.items()):
        limited = limit_rows(split, limits[name], int(config["random_seed"]) + offset)
        prepared[name] = (make_features(limited), limited[TARGET].astype("int8"))
    combined = pd.concat([development, test_raw], ignore_index=True)
    return prepared, combined


def profile(frame: pd.DataFrame, splits: dict) -> dict:
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "fraud_rows": int(frame[TARGET].sum()),
        "fraud_rate": float(frame[TARGET].mean()),
        "missing_values": {key: int(value) for key, value in frame.isna().sum().items()},
        "time_min": str(pd.to_datetime(frame["trans_date_trans_time"]).min()),
        "time_max": str(pd.to_datetime(frame["trans_date_trans_time"]).max()),
        "split_rows": {name: len(values[0]) for name, values in splits.items()},
    }
