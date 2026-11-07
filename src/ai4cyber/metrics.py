"""Binary classification metrics and threshold selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def threshold_at_fpr(
    y_true: pd.Series | np.ndarray, scores: np.ndarray, target_fpr: float
) -> float:
    """Select the highest-recall threshold within the FPR limit."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    valid = np.flatnonzero((fpr <= target_fpr) & np.isfinite(thresholds))
    if valid.size == 0:
        # No observed threshold meets the limit; choose a no-alert boundary.
        return float(np.nextafter(np.max(scores), np.inf))
    best_tpr = tpr[valid].max()
    best = valid[np.flatnonzero(tpr[valid] == best_tpr)]
    # Prefer the stricter threshold when recall is tied.
    return float(thresholds[best].max())


def binary_metrics(y_true, scores: np.ndarray, threshold: float) -> dict[str, float | list]:
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "confusion_matrix": matrix.tolist(),
    }


def fairness_by_age(frame: pd.DataFrame, y_true, scores: np.ndarray, threshold: float) -> dict:
    """Calculate metrics for the configured age groups."""
    if "customer_age" not in frame.columns:
        return {"available": False, "reason": "customer_age is absent"}
    groups = np.where(
        frame["customer_age"].to_numpy() >= 50,
        "age_50_or_over",
        "age_under_50",
    )
    results: dict[str, dict] = {}
    for group in sorted(set(groups)):
        mask = groups == group
        if len(set(np.asarray(y_true)[mask])) < 2:
            continue
        results[group] = binary_metrics(np.asarray(y_true)[mask], scores[mask], threshold)
        results[group]["rows"] = int(mask.sum())
    if len(results) == 2:
        younger = results["age_under_50"]["false_positive_rate"]
        older = results["age_50_or_over"]["false_positive_rate"]
        results["fpr_ratio_50_or_over_to_under_50"] = (
            float(older / younger) if younger else None
        )
    return {"available": bool(results), "groups": results}
