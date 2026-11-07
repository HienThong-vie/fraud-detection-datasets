# ULB: exploratory clustering

MiniBatch K-means selected **k=2** using training-only geometry. Training silhouette: 0.259; mean seed ARI: 0.008.

Weak/unstable geometry: no candidate satisfied both quality floors; do not claim meaningful fraud groups.

This is the second method type alongside the four supervised classifiers. It is not a fifth fraud classifier, and cluster IDs do not mean fraud/legitimate.

| Cluster | Train share | Train fraud rate | Test share* | Test fraud rate* |
|---|---:|---:|---:|---:|
| 0 | 9.8% | 0.92% | 9.2% | 0.30% |
| 1 | 90.2% | 0.13% | 90.8% | 0.11% |

*Previously viewed test periods; interpretation only. Neither fraud rates nor test data selected k, preprocessing or seed. Differences are associations, not causal evidence.

## Training feature interpretation

- Cluster 0: Amount +2.03 training SD, V20 +1.63 training SD, V1 -1.24 training SD.
- Cluster 1: Amount -0.22 training SD, V20 -0.17 training SD, V1 +0.13 training SD.

Profiles describe training numeric means relative to the training-fit scale. Inspect categorical profiles as well; numeric summaries alone do not fully explain mixed-feature distances.

## Reproduce

Run from the suite root:

```bash
python training_model/clustering.py --dataset ulb
```

Completed folders are preserved. Use a new versioned directory for a new experiment.

## Evidence

[Selection and seed stability](silhouette_stability.png) · [Cluster sizes](cluster_sizes.png) · [Numeric profiles](feature_profiles.png) · [Fraud-rate interpretation](fraud_rates.png) · [Metrics](cluster_metrics.csv) · [Full protocol](report.json)

## Limitations

Exploratory segmentation, not a fraud classifier. Silhouette is an internal geometry score, not accuracy. Mixed categorical/numeric distances, scaling, sampling and compression affect geometry. Repeated seeds measure initialization sensitivity, not temporal stability. Previously viewed test periods are not a fresh holdout. Elliptic uses only labelled-subset records and local anonymous features; availability cannot all be independently verified. Demographic features, where present, may drive groups; no adverse decisions should be made from clusters. Fraud-rate Wilson intervals are descriptive and assume independent cases; temporal dependence can make them optimistic.
