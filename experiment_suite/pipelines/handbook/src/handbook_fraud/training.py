from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import PROJECT_ROOT, load_config
from .data import MODEL_COLUMNS, load_raw, prepare_splits, save_processed
from .metrics import best_f1_threshold, binary_metrics
from .profiling import build_profile


def candidate_models(config: dict, positive_weight: float = 1.0) -> dict[str, Pipeline]:
    seed = int(config["random_seed"])
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=int(config["logistic_max_iter"]),
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=int(config["random_forest_trees"]),
                        max_depth=int(config["random_forest_max_depth"]),
                        min_samples_leaf=4,
                        class_weight="balanced_subsample",
                        n_jobs=2,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=int(config["xgboost_estimators"]),
                        max_depth=int(config["xgboost_max_depth"]),
                        learning_rate=float(config["xgboost_learning_rate"]),
                        min_child_weight=5,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=2.0,
                        scale_pos_weight=positive_weight,
                        objective="binary:logistic",
                        eval_metric="aucpr",
                        tree_method="hist",
                        n_jobs=4,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def fit_candidates(
    splits: dict[str, tuple[pd.DataFrame, pd.Series]], config: dict
) -> tuple[dict[str, Pipeline], pd.DataFrame, dict[str, np.ndarray], dict[str, float]]:
    x_train, y_train = splits["train"]
    x_validation, y_validation = splits["validation"]
    fitted = {}
    probabilities = {}
    thresholds = {}
    rows = []
    positive_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    for name, model in candidate_models(config, positive_weight).items():
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_validation)[:, 1]
        threshold = best_f1_threshold(y_validation, probability)
        metrics = binary_metrics(y_validation, probability, threshold)
        fitted[name] = model
        probabilities[name] = probability
        thresholds[name] = threshold
        rows.append({"model": name, **metrics})
    comparison = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    return fitted, comparison, probabilities, thresholds


def save_plots(
    comparison: pd.DataFrame,
    validation_target: pd.Series,
    validation_probabilities: dict[str, np.ndarray],
    test_target: pd.Series,
    test_probability: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> None:
    matplotlib_directory = PROJECT_ROOT / ".matplotlib"
    matplotlib_directory.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_directory))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(
        comparison["model"], comparison["pr_auc"], color=["#7c3aed", "#0f766e", "#b45309"]
    )
    axis.set_ylabel("Validation PR-AUC")
    axis.set_title("Handbook simulated-data model comparison")
    figure.tight_layout()
    figure.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    for name, probability in validation_probabilities.items():
        PrecisionRecallDisplay.from_predictions(validation_target, probability, name=name, ax=axis)
    axis.set_title("Validation precision-recall curves")
    figure.tight_layout()
    figure.savefig(output_dir / "precision_recall_curves.png", dpi=160)
    plt.close(figure)

    prediction = (test_probability >= threshold).astype(int)
    figure, axis = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        test_target, prediction, ax=axis, colorbar=False, values_format="d"
    )
    axis.set_title("Held-out test confusion matrix")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(test_probability[test_target.to_numpy() == 0], bins=40, alpha=0.7, label="legitimate")
    axis.hist(test_probability[test_target.to_numpy() == 1], bins=40, alpha=0.7, label="fraud")
    axis.axvline(threshold, color="black", linestyle="--", label="threshold")
    axis.set_xlabel("Fraud score")
    axis.set_ylabel("Transactions")
    axis.set_title("Held-out score distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "prediction_distribution.png", dpi=160)
    plt.close(figure)


def train(config_path: str | Path) -> dict:
    config = load_config(config_path)
    output_dir = PROJECT_ROOT / "outputs"
    artifact_dir = PROJECT_ROOT / "artifacts"
    checkpoint_dir = PROJECT_ROOT / "checkpoint"
    dataset_directory = Path(config["dataset_directory"])
    frame = load_raw(dataset_directory)
    splits = prepare_splits(frame, config)
    save_processed(splits, PROJECT_ROOT / "preprocessed_df" / "handbook_model_data.csv.gz")

    fitted, comparison, validation_probabilities, thresholds = fit_candidates(splits, config)
    winner = str(comparison.iloc[0]["model"])
    selected_model = fitted[winner]
    threshold = float(thresholds[winner])
    x_test, y_test = splits["test"]
    test_probability = selected_model.predict_proba(x_test)[:, 1]
    test_metrics = binary_metrics(y_test, test_probability, threshold)

    x_train, y_train = splits["train"]
    x_validation, y_validation = splits["validation"]
    development_target = pd.concat([y_train, y_validation], ignore_index=True)
    development_weight = float(
        (development_target == 0).sum() / max(1, (development_target == 1).sum())
    )
    production_model = candidate_models(config, development_weight)[winner]
    production_model.fit(
        pd.concat([x_train, x_validation], ignore_index=True),
        development_target,
    )
    bundle = {
        "model": production_model,
        "threshold": threshold,
        "feature_columns": MODEL_COLUMNS,
        "selected_model": winner,
        "dataset": "Fraud Detection Handbook simulated data",
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_dir / "production_model.joblib")
    joblib.dump(bundle, checkpoint_dir / "production_model.joblib")

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    pd.DataFrame(
        {"TX_FRAUD": y_test.to_numpy(), "fraud_probability": test_probability}
    ).to_csv(output_dir / "test_predictions.csv", index=False)
    results = {
        "selected_model": winner,
        "selection_metric": "validation_pr_auc",
        "validation": json.loads(comparison.iloc[0].to_json()),
        "test": test_metrics,
        "split_rows": {name: len(values[0]) for name, values in splits.items()},
        "leakage_note": "Historical features use only earlier events; test data was not used for selection.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    profile = build_profile(frame, dataset_directory)
    (output_dir / "dataset_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    recorded_config = dict(config)
    recorded_config["dataset_directory"] = str(
        Path(config["dataset_directory"]).relative_to(PROJECT_ROOT)
    )
    model_card = {"dataset_source": profile["source"], "config": recorded_config, **results}
    (artifact_dir / "model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    (checkpoint_dir / "model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    save_plots(
        comparison,
        y_validation,
        validation_probabilities,
        y_test,
        test_probability,
        threshold,
        output_dir,
    )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Handbook simulated-data fraud models")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "training.json"))
    args = parser.parse_args(argv)
    result = train(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
