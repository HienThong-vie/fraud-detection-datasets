"""Generate exploratory data analysis charts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import TrainingConfig
from .data import load_baf_csv


def run_eda(dataset: Path, config: TrainingConfig, output: Path) -> None:
    frame = load_baf_csv(dataset, config)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    counts = frame[config.target_column].value_counts().rename(index={0: "Legitimate", 1: "Fraud"})
    fig, axis = plt.subplots(figsize=(7, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=axis)
    axis.set(title="Class balance", ylabel="Applications", xlabel="Label")
    fig.tight_layout()
    fig.savefig(output / "class_balance.png", dpi=180)
    plt.close(fig)

    monthly = frame.groupby(config.time_column)[config.target_column].mean().mul(100).reset_index()
    fig, axis = plt.subplots(figsize=(8, 4))
    sns.lineplot(monthly, x=config.time_column, y=config.target_column, marker="o", ax=axis)
    axis.set(title="Fraud rate by month", ylabel="Fraud rate (%)", xlabel="Month")
    fig.tight_layout()
    fig.savefig(output / "fraud_rate_by_month.png", dpi=180)
    plt.close(fig)

    if "device_os" in frame.columns:
        category = frame.groupby("device_os")[config.target_column].mean().mul(100).sort_values()
        fig, axis = plt.subplots(figsize=(8, 4))
        sns.barplot(x=category.index, y=category.values, ax=axis)
        axis.set(title="Fraud rate by device OS", ylabel="Fraud rate (%)", xlabel="Device OS")
        fig.tight_layout()
        fig.savefig(output / "fraud_rate_by_device_os.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/Base.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/training.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/base/eda"))
    args = parser.parse_args()
    run_eda(args.dataset, TrainingConfig.from_json(args.config), args.output)


if __name__ == "__main__":
    main()
