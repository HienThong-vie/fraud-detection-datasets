from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import TrainingConfig
from .data import (
    dataset_profile,
    load_competition_test,
    load_train_data,
    model_frame,
    sampled_splits,
)
from .features import CATEGORICAL_FEATURES, MODEL_FEATURES, TARGET, TIME_COLUMN, TRANSACTION_ID
from .metrics import binary_metrics, threshold_at_fpr


def _json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def candidate_models(
    config: TrainingConfig, positive_weight: float = 1.0
) -> dict[str, Pipeline]:
    from .data import build_preprocessor

    return {
        "sgd_logistic": Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                (
                    "model",
                    SGDClassifier(
                        loss="log_loss",
                        class_weight="balanced",
                        max_iter=2_000,
                        tol=1e-3,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=10,
                        random_state=config.random_seed,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        class_weight="balanced",
                        max_iter=config.hist_gradient_boosting_iterations,
                        learning_rate=0.08,
                        max_leaf_nodes=31,
                        l2_regularization=1.0,
                        random_state=config.random_seed,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=config.xgboost_estimators,
                        max_depth=config.xgboost_max_depth,
                        learning_rate=config.xgboost_learning_rate,
                        min_child_weight=5,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=2.0,
                        scale_pos_weight=positive_weight,
                        objective="binary:logistic",
                        eval_metric="aucpr",
                        tree_method="hist",
                        n_jobs=config.n_jobs,
                        random_state=config.random_seed,
                    ),
                ),
            ]
        ),
    }


def train_models(frame: pd.DataFrame, config: TrainingConfig) -> tuple[dict, Pipeline]:
    splits = sampled_splits(frame, config)
    X_train, y_train = model_frame(splits.train), splits.train[TARGET]
    X_validation, y_validation = model_frame(splits.validation), splits.validation[TARGET]
    X_test, y_test = model_frame(splits.test), splits.test[TARGET]

    comparisons = []
    fitted_models: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}
    positive_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    for name, model in candidate_models(config, positive_weight).items():
        model.fit(X_train, y_train)
        validation_probability = model.predict_proba(X_validation)[:, 1]
        threshold = threshold_at_fpr(
            y_validation, validation_probability, config.target_false_positive_rate
        )
        validation_metrics = binary_metrics(y_validation, validation_probability, threshold)
        comparisons.append({"model": name, **validation_metrics})
        fitted_models[name] = model
        thresholds[name] = threshold

    winner = max(comparisons, key=lambda row: row["pr_auc"])["model"]
    champion = fitted_models[winner]
    test_probability = champion.predict_proba(X_test)[:, 1]
    test_metrics = binary_metrics(y_test, test_probability, thresholds[winner])
    evidence = {
        "selection_rule": "Highest validation PR-AUC; internal test period not used for selection",
        "winner": winner,
        "threshold": thresholds[winner],
        "validation_comparison": comparisons,
        "test_metrics": test_metrics,
        "split": {
            "method": "Chronological row-order split after sorting by TransactionDT",
            "train_fraction": config.train_fraction,
            "validation_fraction": config.validation_fraction,
            "test_fraction": 1 - config.train_fraction - config.validation_fraction,
            "train_rows_used": len(splits.train),
            "validation_rows_used": len(splits.validation),
            "test_rows_used": len(splits.test),
            "train_time_max": int(splits.train[TIME_COLUMN].max()),
            "validation_time_min": int(splits.validation[TIME_COLUMN].min()),
            "validation_time_max": int(splits.validation[TIME_COLUMN].max()),
            "test_time_min": int(splits.test[TIME_COLUMN].min()),
        },
    }
    return evidence, champion


def run_training(
    config: TrainingConfig,
    artifact_dir: Path,
    output_dir: Path,
    create_submission: bool = True,
) -> dict:
    data_dir = Path(config.data_dir)
    frame = load_train_data(data_dir)
    profile = dataset_profile(frame)
    evidence, selected_model = train_models(frame, config)

    development = frame.sort_values([TIME_COLUMN, TRANSACTION_ID]).iloc[
        : int(len(frame) * (config.train_fraction + config.validation_fraction))
    ]
    if config.max_train_rows is not None:
        development_limit = config.max_train_rows + (config.max_validation_rows or 0)
        if len(development) > development_limit:
            from .data import bounded_stratified_sample

            development = bounded_stratified_sample(
                development, development_limit, config.random_seed
            )
    production_model = clone(selected_model).fit(model_frame(development), development[TARGET])

    artifact = {
        "model": production_model,
        "feature_columns": MODEL_FEATURES,
        "categorical_columns": CATEGORICAL_FEATURES,
        "threshold": evidence["threshold"],
        "winner": evidence["winner"],
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_dir / "production_model.joblib")

    card = {
        **evidence,
        "config": config.to_dict(),
        "dataset_profile": profile,
        "production_refit_rows": len(development),
        "feature_count": len(MODEL_FEATURES),
        "notice": "Educational fraud-screening model; predictions require human review.",
    }
    _json_dump(card, artifact_dir / "model_card.json")
    _json_dump({"test_metrics": evidence["test_metrics"]}, output_dir / "metrics.json")
    pd.DataFrame(evidence["validation_comparison"]).to_csv(
        output_dir / "model_comparison.csv", index=False
    )

    if create_submission:
        competition = load_competition_test(data_dir)
        probability = production_model.predict_proba(model_frame(competition))[:, 1]
        pd.DataFrame(
            {TRANSACTION_ID: competition[TRANSACTION_ID].astype(int), TARGET: probability}
        ).to_csv(output_dir / "test_predictions.csv", index=False)
        card["competition_prediction_rows"] = len(competition)
        _json_dump(card, artifact_dir / "model_card.json")
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training.json"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--skip-submission", action="store_true")
    args = parser.parse_args()
    config = TrainingConfig.from_json(args.config)
    result = run_training(config, args.artifacts, args.outputs, not args.skip_submission)
    print(json.dumps({"winner": result["winner"], "test_metrics": result["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()

