from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import CHECKPOINT


def main() -> None:
    card = json.loads((CHECKPOINT / "model_card.json").read_text(encoding="utf-8"))
    print("Dataset | Model | PR-AUC | ROC-AUC | Recall | FPR")
    print("--- | --- | ---: | ---: | ---: | ---:")
    for dataset in card["datasets"]:
        metrics = dataset["test_metrics"]
        print(
            f"{dataset['dataset']} | {dataset['winner']} | {metrics['pr_auc']:.4f} | "
            f"{metrics['roc_auc']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['false_positive_rate']:.4f}"
        )


if __name__ == "__main__":
    main()

