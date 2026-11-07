from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_COLUMNS = [
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
]
MODEL_COLUMNS = [
    "TX_AMOUNT",
    "hour",
    "day_of_week",
    "customer_previous_count",
    "customer_mean_amount_previous",
    "customer_seconds_since_previous",
    "terminal_previous_count",
    "terminal_seconds_since_previous",
]
TARGET = "TX_FRAUD"


def load_raw(directory: str | Path) -> pd.DataFrame:
    source = Path(directory)
    files = sorted(source.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(
            f"No Handbook .pkl files found in {source}. Run python scripts/download_handbook.py first."
        )
    frames = []
    for path in files:
        daily = pd.read_pickle(path)
        missing = sorted(set(RAW_COLUMNS) - set(daily.columns))
        if missing:
            raise ValueError(f"{path.name} is missing columns: {missing}")
        frames.append(daily.loc[:, RAW_COLUMNS])
    frame = pd.concat(frames, ignore_index=True)
    frame["TX_DATETIME"] = pd.to_datetime(frame["TX_DATETIME"], errors="raise")
    frame["TRANSACTION_ID"] = pd.to_numeric(frame["TRANSACTION_ID"], errors="raise").astype("int64")
    frame["TX_TIME_SECONDS"] = pd.to_numeric(frame["TX_TIME_SECONDS"], errors="raise").astype("int64")
    frame["TX_TIME_DAYS"] = pd.to_numeric(frame["TX_TIME_DAYS"], errors="raise").astype("int16")
    frame["TX_FRAUD"] = frame["TX_FRAUD"].astype("int8")
    frame["TX_FRAUD_SCENARIO"] = frame["TX_FRAUD_SCENARIO"].astype("int8")
    frame["TX_AMOUNT"] = frame["TX_AMOUNT"].astype("float32")
    validate_raw(frame)
    return frame


def validate_raw(frame: pd.DataFrame) -> None:
    labels = set(frame[TARGET].dropna().astype(int).unique())
    if not labels.issubset({0, 1}) or not labels:
        raise ValueError("TX_FRAUD must contain binary labels")
    if frame["TRANSACTION_ID"].duplicated().any():
        raise ValueError("TRANSACTION_ID must be unique")


def engineer_prior_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["TX_DATETIME", "TRANSACTION_ID"], kind="stable").reset_index(drop=True)
    features = pd.DataFrame(index=ordered.index)
    features["TX_AMOUNT"] = ordered["TX_AMOUNT"].astype("float32")
    features["hour"] = ordered["TX_DATETIME"].dt.hour.astype("int8")
    features["day_of_week"] = ordered["TX_DATETIME"].dt.dayofweek.astype("int8")

    customer_group = ordered.groupby("CUSTOMER_ID", sort=False)
    customer_count = customer_group.cumcount()
    customer_previous_sum = customer_group["TX_AMOUNT"].cumsum() - ordered["TX_AMOUNT"]
    features["customer_previous_count"] = customer_count.astype("int32")
    features["customer_mean_amount_previous"] = (
        customer_previous_sum.div(customer_count.replace(0, np.nan)).fillna(0).astype("float32")
    )
    features["customer_seconds_since_previous"] = (
        customer_group["TX_TIME_SECONDS"].diff().fillna(-1).clip(lower=-1).astype("float32")
    )

    terminal_group = ordered.groupby("TERMINAL_ID", sort=False)
    features["terminal_previous_count"] = terminal_group.cumcount().astype("int32")
    features["terminal_seconds_since_previous"] = (
        terminal_group["TX_TIME_SECONDS"].diff().fillna(-1).clip(lower=-1).astype("float32")
    )
    return ordered, features.loc[:, MODEL_COLUMNS]


def chronological_masks(
    frame: pd.DataFrame, train_fraction: float, validation_fraction: float
) -> dict[str, pd.Series]:
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Split fractions must be positive and sum to less than one")
    days = np.sort(frame["TX_TIME_DAYS"].unique())
    train_index = max(1, int(len(days) * train_fraction))
    validation_index = max(train_index + 1, int(len(days) * (train_fraction + validation_fraction)))
    train_end = days[min(train_index - 1, len(days) - 1)]
    validation_end = days[min(validation_index - 1, len(days) - 1)]
    return {
        "train": frame["TX_TIME_DAYS"] <= train_end,
        "validation": (frame["TX_TIME_DAYS"] > train_end)
        & (frame["TX_TIME_DAYS"] <= validation_end),
        "test": frame["TX_TIME_DAYS"] > validation_end,
    }


def limit_rows(
    features: pd.DataFrame, target: pd.Series, maximum: int | None, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    if maximum is None or len(features) <= maximum:
        return features.copy(), target.copy()
    selected, _ = train_test_split(
        np.arange(len(features)),
        train_size=maximum,
        random_state=seed,
        stratify=target,
    )
    selected = np.sort(selected)
    return features.iloc[selected].copy(), target.iloc[selected].copy()


def prepare_splits(frame: pd.DataFrame, config: dict) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    ordered, features = engineer_prior_features(frame)
    masks = chronological_masks(
        ordered,
        train_fraction=float(config["train_fraction"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    limits = {
        "train": config.get("max_train_rows"),
        "validation": config.get("max_validation_rows"),
        "test": config.get("max_test_rows"),
    }
    splits = {}
    for offset, (name, mask) in enumerate(masks.items()):
        x = features.loc[mask].reset_index(drop=True)
        y = ordered.loc[mask, TARGET].reset_index(drop=True).astype("int8")
        splits[name] = limit_rows(x, y, limits[name], int(config["random_seed"]) + offset)
    return splits


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
