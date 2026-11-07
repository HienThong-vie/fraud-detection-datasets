"""Train, evaluate, and select fraud classifiers."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .config import TrainingConfig
from .constants import (
    ALL_NEGATIVE_MISSING_COLUMNS,
    MISSING_SENTINEL_COLUMNS,
    SYNTHETIC_DATA_NOTICE,
)
from .data import (
    build_preprocessor,
    dataset_profile,
    load_baf_csv,
    make_temporal_data,
    prepare_catboost_frame,
    replace_missing_sentinels,
)
from .metrics import binary_metrics, fairness_by_age, threshold_at_fpr


def build_models(config: TrainingConfig) -> dict[str, object]:
    """Create the classifier set used in the experiment."""
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=config.random_seed,
        ),
        "sgd_logistic": SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            penalty="elasticnet",
            l1_ratio=0.15,
            max_iter=1500,
            tol=1e-3,
            average=True,
            random_state=config.random_seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=config.tree_estimators,
            max_depth=config.tree_max_depth,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=config.n_jobs,
            random_state=config.random_seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=config.tree_estimators,
            max_depth=config.tree_max_depth,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=config.n_jobs,
            random_state=config.random_seed,
        ),
    }


def build_catboost_pipeline(config: TrainingConfig, temporal) -> Pipeline:
    """Build the native-categorical boosted-tree candidate."""
    prepare = FunctionTransformer(
        prepare_catboost_frame,
        validate=False,
        kw_args={
            "feature_columns": temporal.feature_columns,
            "categorical_columns": temporal.categorical_columns,
        },
        feature_names_out="one-to-one",
    )
    model = CatBoostClassifier(
        iterations=config.catboost_iterations,
        depth=config.catboost_depth,
        learning_rate=config.catboost_learning_rate,
        loss_function="Logloss",
        eval_metric="PRAUC",
        auto_class_weights="Balanced",
        random_seed=config.random_seed,
        thread_count=config.n_jobs,
        allow_writing_files=False,
        verbose=False,
    )
    return Pipeline([("prepare", prepare), ("model", model)])


def _json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _feature_importance(pipeline: Pipeline, raw_features: list[str]) -> list[dict]:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_)).reshape(-1)
    else:
        return []
    if "preprocessor" not in pipeline.named_steps:
        total = float(np.abs(values).sum()) or 1.0
        return [
            {"feature": feature, "importance": float(abs(value) / total)}
            for feature, value in sorted(
                zip(raw_features, values, strict=False),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
            if value != 0
        ]
    transformed = pipeline.named_steps["preprocessor"].get_feature_names_out()
    aggregated = {feature: 0.0 for feature in raw_features}
    for name, value in zip(transformed, values, strict=False):
        token = name.split("__", 1)[-1]
        matches = [
            feature
            for feature in raw_features
            if token == feature or token.startswith(feature + "_")
        ]
        if matches:
            aggregated[max(matches, key=len)] += float(abs(value))
    total = sum(aggregated.values()) or 1.0
    return [
        {"feature": feature, "importance": value / total}
        for feature, value in sorted(aggregated.items(), key=lambda item: item[1], reverse=True)
        if value > 0
    ]


def _write_plots(dataset_name: str, results: list[dict], y_test, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 5))
    for result in results:
        PrecisionRecallDisplay.from_predictions(
            y_test, result.pop("_test_scores"), name=result["model"], ax=axis
        )
    axis.set_title(f"Precision-recall curves - {dataset_name}")
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curves.png", dpi=180)
    plt.close(fig)

    table = pd.DataFrame(
        [
            {
                "model": row["model"],
                "PR-AUC": row["test"]["pr_auc"],
                "ROC-AUC": row["test"]["roc_auc"],
                "Recall": row["test"]["recall"],
                "Precision": row["test"]["precision"],
            }
            for row in results
        ]
    ).melt("model", var_name="metric", value_name="value")
    fig, axis = plt.subplots(figsize=(9, 5))
    sns.barplot(table, x="model", y="value", hue="metric", ax=axis)
    axis.tick_params(axis="x", rotation=18)
    axis.set_title(f"Held-out month 7 model comparison - {dataset_name}")
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close(fig)


def train_dataset(
    dataset_path: Path,
    config: TrainingConfig,
    artifact_root: Path,
    output_root: Path,
    processed_root: Path,
) -> dict:
    dataset_name = dataset_path.stem
    safe_name = dataset_name.lower().replace(" ", "_")
    artifact_dir = artifact_root / safe_name
    output_dir = output_root / safe_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_frame = load_baf_csv(dataset_path, config)
    profile = dataset_profile(raw_frame, config)
    frame = replace_missing_sentinels(raw_frame)
    temporal = make_temporal_data(frame, config)
    X_train = temporal.train[temporal.feature_columns]
    y_train = temporal.train[config.target_column]
    X_validation = temporal.validation[temporal.feature_columns]
    y_validation = temporal.validation[config.target_column]
    X_test = temporal.test[temporal.feature_columns]
    y_test = temporal.test[config.target_column]

    # Save the exact rows used in each split.
    processed_root.mkdir(parents=True, exist_ok=True)
    used_columns = list(
        dict.fromkeys(
            [
                *temporal.feature_columns,
                config.target_column,
                config.time_column,
                *[c for c in config.protected_features if c in frame.columns],
            ]
        )
    )
    processed_parts = []
    for split_name, split_frame in (
        ("train", temporal.train),
        ("validation", temporal.validation),
        ("test", temporal.test),
    ):
        part = split_frame[used_columns].copy()
        part.insert(0, "_split", split_name)
        processed_parts.append(part)
    pd.concat(processed_parts, ignore_index=True).to_csv(
        processed_root / f"{safe_name}_model_data.csv.gz", index=False, compression="gzip"
    )

    candidates = {
        model_name: Pipeline(
            [("preprocessor", copy.deepcopy(build_preprocessor(temporal))), ("model", estimator)]
        )
        for model_name, estimator in build_models(config).items()
    }
    candidates["catboost"] = build_catboost_pipeline(config, temporal)

    results: list[dict] = []
    for model_name, pipeline in candidates.items():
        fit_params = (
            {"model__cat_features": temporal.categorical_columns}
            if model_name == "catboost"
            else {}
        )
        pipeline.fit(X_train, y_train, **fit_params)
        validation_scores = pipeline.predict_proba(X_validation)[:, 1]
        threshold = threshold_at_fpr(
            y_validation, validation_scores, config.target_false_positive_rate
        )
        test_scores = pipeline.predict_proba(X_test)[:, 1]
        importance = _feature_importance(pipeline, temporal.feature_columns)
        artifact = {
            "pipeline": pipeline,
            "threshold": threshold,
            "dataset": dataset_name,
            "model_name": model_name,
            "feature_columns": temporal.feature_columns,
            "numeric_columns": temporal.numeric_columns,
            "categorical_columns": temporal.categorical_columns,
            "protected_features_excluded": config.protected_features,
            "missing_sentinel_columns": sorted(MISSING_SENTINEL_COLUMNS),
            "missing_value_policy": {
                "all_negative_values": sorted(ALL_NEGATIVE_MISSING_COLUMNS),
                "exact_minus_one": sorted(MISSING_SENTINEL_COLUMNS - ALL_NEGATIVE_MISSING_COLUMNS),
            },
            "split_policy": {
                "train_months": config.train_months,
                "validation_months": config.validation_months,
                "test_months": config.test_months,
            },
            "global_feature_importance": importance[:20],
            "notice": SYNTHETIC_DATA_NOTICE,
        }
        model_path = artifact_dir / f"{model_name}.joblib"
        joblib.dump(artifact, model_path, compress=3)
        row = {
            "dataset": dataset_name,
            "model": model_name,
            "artifact": str(model_path),
            "train_rows": len(temporal.train),
            "validation_rows": len(temporal.validation),
            "test_rows": len(temporal.test),
            "validation": binary_metrics(y_validation, validation_scores, threshold),
            "test": binary_metrics(y_test, test_scores, threshold),
            "fairness": fairness_by_age(temporal.test, y_test, test_scores, threshold),
            "top_features": importance[:15],
            "_test_scores": test_scores,
        }
        results.append(row)

    # Select by validation PR-AUC only.
    winner = max(results, key=lambda item: item["validation"]["pr_auc"])
    champion_path = artifact_dir / "champion.joblib"
    shutil.copy2(winner["artifact"], champion_path)
    _write_plots(dataset_name, results, y_test, output_dir)
    serializable = [
        {key: value for key, value in row.items() if not key.startswith("_")} for row in results
    ]
    _json_dump(profile, output_dir / "dataset_profile.json")
    _json_dump(serializable, output_dir / "metrics.json")
    pd.DataFrame(
        [{"dataset": row["dataset"], "model": row["model"], **row["test"]} for row in serializable]
    ).drop(columns=["confusion_matrix"]).to_csv(output_dir / "model_comparison.csv", index=False)
    return {
        "dataset": dataset_name,
        "winner": winner["model"],
        "champion_artifact": str(champion_path),
        "validation_pr_auc": winner["validation"]["pr_auc"],
        "test_metrics": winner["test"],
        "profile": profile,
    }


def run_training(
    data_dir: Path,
    config: TrainingConfig,
    artifact_root: Path,
    output_root: Path,
    processed_root: Path | None = None,
) -> dict:
    processed_root = processed_root or data_dir.parent / "processed"
    summaries = [
        train_dataset(data_dir / filename, config, artifact_root, output_root, processed_root)
        for filename in config.dataset_files
    ]
    base = next((item for item in summaries if item["dataset"].lower() == "base"), summaries[0])
    production_path = artifact_root / "production_model.joblib"
    shutil.copy2(base["champion_artifact"], production_path)
    leakage_controls = {
        "development_months": sorted(set(config.train_months + config.validation_months)),
        "test_months": sorted(config.test_months),
        "temporal_partitions_disjoint": not (
            set(config.train_months) & set(config.validation_months)
            or set(config.train_months) & set(config.test_months)
            or set(config.validation_months) & set(config.test_months)
        ),
        "preprocessing_fit_scope": "sklearn Pipeline fitted on training rows only",
        "imbalance_control": "class_weight applied by estimators during training only",
        "target_excluded_from_features": True,
        "time_column_excluded_from_features": True,
        "protected_features_excluded_from_models": config.protected_features,
        "missing_value_policy": {
            "all_negative_values": sorted(ALL_NEGATIVE_MISSING_COLUMNS),
            "exact_minus_one": sorted(MISSING_SENTINEL_COLUMNS - ALL_NEGATIVE_MISSING_COLUMNS),
        },
        "age_fairness_groups": {
            "younger": "customer_age < 50",
            "older": "customer_age >= 50",
        },
        "test_data_used_for_selection": False,
        "fixed_random_seed": config.random_seed,
    }
    model_card = {
        "selection_rule": "Best validation PR-AUC per dataset; production model is Base champion.",
        "test_data_not_used_for_selection": True,
        "production_artifact": str(production_path),
        "datasets": summaries,
        "config": config.to_dict(),
        "leakage_controls": leakage_controls,
        "notice": SYNTHETIC_DATA_NOTICE,
    }
    _json_dump(model_card, artifact_root / "model_card.json")
    _json_dump(leakage_controls, output_root / "leakage_checklist.json")
    return model_card


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("configs/training.json"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    card = run_training(
        args.data_dir,
        TrainingConfig.from_json(args.config),
        args.artifacts,
        args.outputs,
        args.processed,
    )
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
