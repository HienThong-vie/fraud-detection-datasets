from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import CHECKPOINT, EVALUATION_RESULTS


def comparison_chart() -> None:
    comparison = pd.read_csv(EVALUATION_RESULTS / "model_comparison.csv")
    chart = comparison.set_index("model")[["roc_auc", "pr_auc"]].plot(
        kind="bar", figsize=(8, 5), color=["#315D9A", "#E58F3A"]
    )
    chart.set_title("Validation model comparison")
    chart.set_ylabel("Score")
    chart.set_ylim(0, 1)
    chart.tick_params(axis="x", rotation=0)
    chart.figure.tight_layout()
    chart.figure.savefig(EVALUATION_RESULTS / "model_comparison.png", dpi=160)
    plt.close(chart.figure)


def confusion_chart(metrics: dict) -> None:
    matrix = [
        [metrics["true_negative"], metrics["false_positive"]],
        [metrics["false_negative"], metrics["true_positive"]],
    ]
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{matrix[row][column]:,}", ha="center", va="center")
    axis.set_xticks([0, 1], labels=["Legitimate", "Fraud"])
    axis.set_yticks([0, 1], labels=["Legitimate", "Fraud"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Chronological holdout confusion matrix")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(EVALUATION_RESULTS / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def prediction_chart() -> None:
    predictions = pd.read_csv(EVALUATION_RESULTS / "test_predictions.csv", usecols=["isFraud"])
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(predictions["isFraud"], bins=40, color="#315D9A", edgecolor="white")
    axis.set_title("Competition fraud-score distribution")
    axis.set_xlabel("Predicted fraud score")
    axis.set_ylabel("Transactions")
    figure.tight_layout()
    figure.savefig(EVALUATION_RESULTS / "prediction_distribution.png", dpi=160)
    plt.close(figure)


def main() -> None:
    card = json.loads((CHECKPOINT / "model_card.json").read_text(encoding="utf-8"))
    metrics = card["test_metrics"]
    comparison_chart()
    confusion_chart(metrics)
    prediction_chart()
    print(f"Selected model: {card['winner']}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"False-positive rate: {metrics['false_positive_rate']:.4f}")
    if "competition_prediction_rows" in card:
        print(f"Competition predictions: {card['competition_prediction_rows']:,}")
    print(f"Charts: {EVALUATION_RESULTS}")


if __name__ == "__main__":
    main()

