"""Exploratory clustering pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

from .config import TrainingConfig
from .data import build_preprocessor, load_baf_csv, make_temporal_data, replace_missing_sentinels


def run_clustering(dataset: Path, config: TrainingConfig, artifacts: Path, outputs: Path) -> dict:
    frame = replace_missing_sentinels(load_baf_csv(dataset, config))
    temporal = make_temporal_data(frame, config)
    sample_n = min(config.clustering_sample_rows, len(temporal.train))
    sample = temporal.train.sample(n=sample_n, random_state=config.random_seed).reset_index(
        drop=True
    )
    preprocessor = build_preprocessor(temporal)
    matrix = preprocessor.fit_transform(sample[temporal.feature_columns])
    dimensions = max(2, min(12, matrix.shape[1] - 1))
    reducer = TruncatedSVD(n_components=dimensions, random_state=config.random_seed)
    embedding = reducer.fit_transform(matrix)

    candidates: list[dict] = []
    fitted: dict[int, MiniBatchKMeans] = {}
    silhouette_n = min(5000, len(sample))
    for k in config.clustering_k_values:
        model = MiniBatchKMeans(
            n_clusters=k, batch_size=2048, n_init=10, random_state=config.random_seed
        )
        labels = model.fit_predict(embedding)
        score = silhouette_score(
            embedding, labels, sample_size=silhouette_n, random_state=config.random_seed
        )
        candidates.append({"k": int(k), "silhouette": float(score)})
        fitted[k] = model

    best = max(candidates, key=lambda item: item["silhouette"])
    winner = fitted[best["k"]]
    labels = winner.predict(embedding)
    analysis = sample[[config.target_column]].copy()
    analysis["cluster"] = labels
    summary = (
        analysis.groupby("cluster")[config.target_column]
        .agg(rows="size", fraud_count="sum", fraud_rate="mean")
        .reset_index()
    )
    payload = {
        "dataset": dataset.stem,
        "rows": int(sample_n),
        "best_k": int(best["k"]),
        "best_silhouette": float(best["silhouette"]),
        "candidates": candidates,
        "clusters": summary.to_dict(orient="records"),
        "interpretation_limit": (
            "Clusters are exploratory patterns, not fraud labels and not used by the decision endpoint."
        ),
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"preprocessor": preprocessor, "reducer": reducer, "model": winner, "metadata": payload},
        artifacts / "clustering.joblib",
        compress=3,
    )
    (outputs / "clustering_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    summary.to_csv(outputs / "cluster_summary.csv", index=False)

    plot = pd.DataFrame(
        {"component_1": embedding[:, 0], "component_2": embedding[:, 1], "cluster": labels}
    )
    if len(plot) > 8000:
        plot = plot.sample(8000, random_state=config.random_seed)
    fig, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        plot, x="component_1", y="component_2", hue="cluster", palette="tab10", s=16, ax=axis
    )
    axis.set_title(f"Exploratory clusters - {dataset.stem}")
    fig.tight_layout()
    fig.savefig(outputs / "cluster_projection.png", dpi=180)
    plt.close(fig)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/Base.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/training.json"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/base"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs/base"))
    args = parser.parse_args()
    result = run_clustering(
        args.dataset, TrainingConfig.from_json(args.config), args.artifacts, args.outputs
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
