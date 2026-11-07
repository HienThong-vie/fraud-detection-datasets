# BAF: exploratory clustering

MiniBatch K-means selected **k=2** using training-only geometry. Training silhouette: 0.079; mean seed ARI: 0.080.

Weak/unstable geometry: no candidate satisfied both quality floors; do not claim meaningful fraud groups.

This is the second method type alongside the four supervised classifiers. It is not a fifth fraud classifier, and cluster IDs do not mean fraud/legitimate.

| Cluster | Train share | Train fraud rate | Test share* | Test fraud rate* |
|---|---:|---:|---:|---:|
| 0 | 35.3% | 1.17% | 9.9% | 1.28% |
| 1 | 64.7% | 0.85% | 90.1% | 1.42% |

*Previously viewed test periods; interpretation only. Neither fraud rates nor test data selected k, preprocessing or seed. Differences are associations, not causal evidence.

## Training feature interpretation

- Cluster 0: current_address_months_count +0.72 training SD, phone_home_valid +0.46 training SD, bank_branch_count_8w +0.44 training SD.
- Cluster 1: current_address_months_count -0.40 training SD, bank_branch_count_8w -0.25 training SD, phone_home_valid -0.24 training SD.

Profiles describe training numeric means relative to the training-fit scale. Inspect categorical profiles as well; numeric summaries alone do not fully explain mixed-feature distances.

## Reproduce

Run from the suite root:

```bash
python training_model/clustering.py --dataset baf
```

Completed folders are preserved. Use a new versioned directory for a new experiment.

## Evidence

[Selection and seed stability](silhouette_stability.png) · [Cluster sizes](cluster_sizes.png) · [Numeric profiles](feature_profiles.png) · [Fraud-rate interpretation](fraud_rates.png) · [Metrics](cluster_metrics.csv) · [Full protocol](report.json)

## Limitations

Exploratory segmentation, not a fraud classifier. Silhouette is an internal geometry score, not accuracy. Mixed categorical/numeric distances, scaling, sampling and compression affect geometry. Repeated seeds measure initialization sensitivity, not temporal stability. Previously viewed test periods are not a fresh holdout. Elliptic uses only labelled-subset records and local anonymous features; availability cannot all be independently verified. Demographic features, where present, may drive groups; no adverse decisions should be made from clusters. Fraud-rate Wilson intervals are descriptive and assume independent cases; temporal dependence can make them optimistic.
