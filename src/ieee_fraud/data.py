from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .config import TrainingConfig
from .features import (
    CATEGORICAL_FEATURES,
    IDENTITY_CATEGORICAL,
    IDENTITY_NUMERIC,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    TEST_TRANSACTION_COLUMNS,
    TIME_COLUMN,
    TRAIN_TRANSACTION_COLUMNS,
    TRANSACTION_ID,
)


@dataclass
class TemporalData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def normalize_identity_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        column: column.replace("id-", "id_")
        for column in frame.columns
        if column.startswith("id-")
    }
    return frame.rename(columns=renamed)


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    seconds = result[TIME_COLUMN]
    result["TransactionDay"] = seconds // 86_400
    result["TransactionHour"] = (seconds // 3_600) % 24
    result["TransactionWeekday"] = (seconds // 86_400) % 7
    return result


def _load_identity(path: Path) -> pd.DataFrame:
    frame = normalize_identity_columns(pd.read_csv(path))
    selected = [TRANSACTION_ID, *IDENTITY_NUMERIC, *IDENTITY_CATEGORICAL]
    return frame[[column for column in selected if column in frame.columns]]


def load_train_data(data_dir: str | Path) -> pd.DataFrame:
    root = Path(data_dir)
    transaction = pd.read_csv(
        root / "train_transaction.csv",
        usecols=TRAIN_TRANSACTION_COLUMNS,
        low_memory=False,
    )
    identity = _load_identity(root / "train_identity.csv")
    return add_time_features(transaction.merge(identity, on=TRANSACTION_ID, how="left"))


def load_competition_test(data_dir: str | Path) -> pd.DataFrame:
    root = Path(data_dir)
    transaction = pd.read_csv(
        root / "test_transaction.csv",
        usecols=TEST_TRANSACTION_COLUMNS,
        low_memory=False,
    )
    identity = _load_identity(root / "test_identity.csv")
    return add_time_features(transaction.merge(identity, on=TRANSACTION_ID, how="left"))


def chronological_split(frame: pd.DataFrame, config: TrainingConfig) -> TemporalData:
    ordered = frame.sort_values([TIME_COLUMN, TRANSACTION_ID]).reset_index(drop=True)
    train_end = int(len(ordered) * config.train_fraction)
    validation_end = int(len(ordered) * (config.train_fraction + config.validation_fraction))
    if train_end == 0 or validation_end <= train_end or validation_end >= len(ordered):
        raise ValueError("Dataset is too small for the configured chronological split")
    return TemporalData(
        train=ordered.iloc[:train_end].copy(),
        validation=ordered.iloc[train_end:validation_end].copy(),
        test=ordered.iloc[validation_end:].copy(),
    )


def bounded_stratified_sample(
    frame: pd.DataFrame, max_rows: int | None, random_seed: int
) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame.copy()
    sampled, _ = train_test_split(
        frame,
        train_size=max_rows,
        stratify=frame[TARGET],
        random_state=random_seed,
    )
    return sampled.sort_values([TIME_COLUMN, TRANSACTION_ID]).reset_index(drop=True)


def sampled_splits(frame: pd.DataFrame, config: TrainingConfig) -> TemporalData:
    split = chronological_split(frame, config)
    return TemporalData(
        train=bounded_stratified_sample(split.train, config.max_train_rows, config.random_seed),
        validation=bounded_stratified_sample(
            split.validation, config.max_validation_rows, config.random_seed
        ),
        test=bounded_stratified_sample(split.test, config.max_test_rows, config.random_seed),
    )


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
        sparse_threshold=0,
    )


def model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in MODEL_FEATURES:
        if column not in result:
            result[column] = np.nan
    return result[MODEL_FEATURES]


def dataset_profile(frame: pd.DataFrame) -> dict:
    identity_fields = IDENTITY_NUMERIC + IDENTITY_CATEGORICAL
    return {
        "rows": len(frame),
        "columns_loaded": len(frame.columns),
        "fraud_count": int(frame[TARGET].sum()),
        "fraud_rate": float(frame[TARGET].mean()),
        "transaction_id_unique": bool(frame[TRANSACTION_ID].is_unique),
        "transaction_dt_min": int(frame[TIME_COLUMN].min()),
        "transaction_dt_max": int(frame[TIME_COLUMN].max()),
        "identity_match_rows": int(frame[identity_fields].notna().any(axis=1).sum()),
        "selected_feature_missing_rate": {
            column: float(frame[column].isna().mean()) for column in MODEL_FEATURES
        },
    }

