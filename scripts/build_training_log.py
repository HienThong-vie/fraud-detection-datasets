"""Build the team training log from generated machine-readable evidence."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path


def _metric(value: float) -> str:
    return f"{value:.4f}"


def build_log(model_card_path: Path, profile_path: Path) -> str:
    card = json.loads(model_card_path.read_text(encoding="utf-8"))
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))
    source = profiles["source"]
    lines = [
        "# AI4Cyber ML Training Run Log",
        "",
        "This log is generated from the model card and dataset profile manifest.",
        "",
        "## Run identity",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Log generated | {datetime.now().astimezone().isoformat(timespec='seconds')} |",
        "| Architecture | ML-only |",
        "| Selection rule | Highest validation PR-AUC; test data not used for selection |",
        "| Random seed | 42 |",
        "| Development period | Months 0-5 (0-4 train; 5 validation) |",
        "| Held-out test period | Months 6-7 |",
        "",
        "## Environment",
        "",
        "| Component | Version |",
        "| --- | ---: |",
        f"| Python | {platform.python_version()} |",
        f"| pandas | {version('pandas')} |",
        f"| NumPy | {version('numpy')} |",
        f"| scikit-learn | {version('scikit-learn')} |",
        f"| FastAPI | {version('fastapi')} |",
        f"| Matplotlib | {version('matplotlib')} |",
        "",
        "## Source-data verification",
        "",
        f"- Kaggle dataset: `{source['kaggle_dataset']}`",
        f"- Version: {source['kaggle_version']} ({source['kaggle_updated']})",
        f"- Licence: {source['license']}",
        f"- Source: {source['source_url']}",
        "",
        "| Dataset | Rows | Columns | Fraud rate | Effective missing codes | SHA-256 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for profile in profiles["profiles"]:
        lines.append(
            f"| {profile['dataset']} | {profile['rows']:,} | {profile['columns']} | "
            f"{profile['fraud_rate']:.4%} | "
            f"{sum(profile['effective_missing_sentinel_counts'].values()):,} | "
            f"`{profile['sha256']}` |"
        )

    lines.extend(
        [
            "",
            (
                "Per-column dtypes, raw null counts, sentinel counts, monthly counts, and "
                "differences from Base are stored in `outputs/profiling/`."
            ),
            "",
            "## Data preparation and leakage controls",
            "",
            "- Every negative `intended_balcon_amount` value is converted to missing before preprocessing.",
            "- Exact `-1` sentinels are converted for the other documented fields.",
            "- Median numeric imputation and most-frequent categorical imputation are fitted on training rows only.",
            "- Categorical fields use one-hot encoding with unknown-category handling.",
            "- Class imbalance is handled through estimator class weights; no pre-split resampling is used.",
            "- Target, month, protected diagnostics, `device_fraud_count`, `x1`, and `x2` are excluded from predictors.",
            "- Sampling is stratified by month and label with fixed seed 42.",
            "- Thresholds and model winners use validation data only.",
            "- The age fairness diagnostic compares customers under 50 with customers aged 50 or over.",
            "- Full checklist: `docs/LEAKAGE_CHECKLIST.md` and `outputs/leakage_checklist.json`.",
            "",
            "## Champion results on held-out months 6-7",
            "",
            "| Dataset | Selected model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset in card["datasets"]:
        metric = dataset["test_metrics"]
        lines.append(
            f"| {dataset['dataset']} | {dataset['winner'].replace('_', ' ').title()} | "
            f"{_metric(metric['threshold'])} | {_metric(metric['accuracy'])} | "
            f"{_metric(metric['precision'])} | {_metric(metric['recall'])} | "
            f"{_metric(metric['f1'])} | {_metric(metric['roc_auc'])} | "
            f"{_metric(metric['pr_auc'])} | {_metric(metric['false_positive_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Commands",
            "",
            "```bash",
            "python -m ai4cyber.profiling",
            "python -m ai4cyber.training",
            "python -m ai4cyber.clustering",
            "python -m ai4cyber.eda",
            "pytest -q",
            "ruff check .",
            "```",
            "",
            "## Evidence files",
            "",
            "- `docs/DATASET_PROVENANCE.md`",
            "- `docs/LEAKAGE_CHECKLIST.md`",
            "- `outputs/profiling/profile_manifest.json`",
            "- `outputs/profiling/dataset_comparison.csv`",
            "- `artifacts/model_card.json`",
            "- `outputs/leakage_checklist.json`",
            "- `outputs/<dataset>/metrics.json`",
            "- `outputs/<dataset>/model_comparison.csv`",
            "- `data/processed/*_model_data.csv.gz` (local, Git-ignored)",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-card", type=Path, default=Path("artifacts/model_card.json"))
    parser.add_argument(
        "--profiles", type=Path, default=Path("outputs/profiling/profile_manifest.json")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/TRAINING_RUN_LOG.md"))
    args = parser.parse_args()
    args.output.write_text(build_log(args.model_card, args.profiles), encoding="utf-8")
    print(f"Training log written to {args.output}")


if __name__ == "__main__":
    main()
