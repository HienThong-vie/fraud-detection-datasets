# SPARKOV: exploratory clustering

MiniBatch K-means selected **k=5** using training-only geometry. Training silhouette: 0.141; mean seed ARI: 0.281.

Weak/unstable geometry: no candidate satisfied both quality floors; do not claim meaningful fraud groups.

This is the second method type alongside the four supervised classifiers. It is not a fifth fraud classifier, and cluster IDs do not mean fraud/legitimate.

| Cluster | Train share | Train fraud rate | Test share* | Test fraud rate* |
|---|---:|---:|---:|---:|
| 0 | 4.7% | 0.24% | 4.7% | 0.00% |
| 1 | 3.5% | 12.36% | 3.3% | 9.12% |
| 2 | 30.1% | 0.20% | 30.2% | 0.12% |
| 3 | 34.1% | 0.14% | 28.3% | 0.13% |
| 4 | 27.7% | 0.10% | 33.6% | 0.04% |

*Previously viewed test periods; interpretation only. Neither fraud rates nor test data selected k, preprocessing or seed. Differences are associations, not causal evidence.

## Training feature interpretation

- Cluster 0: city_pop +4.03 training SD, age -0.25 training SD, amt -0.09 training SD.
- Cluster 1: amt +3.97 training SD, hour +0.13 training SD, age -0.08 training SD.
- Cluster 2: hour -1.17 training SD, age +0.49 training SD, city_pop -0.22 training SD.
- Cluster 3: weekday +0.80 training SD, hour +0.61 training SD, amt -0.23 training SD.
- Cluster 4: weekday -1.06 training SD, hour +0.50 training SD, amt -0.21 training SD.

Profiles describe training numeric means relative to the training-fit scale. Inspect categorical profiles as well; numeric summaries alone do not fully explain mixed-feature distances.

## Reproduce

Run from the suite root:

```bash
python training_model/clustering.py --dataset sparkov
```

Completed folders are preserved. Use a new versioned directory for a new experiment.

## Evidence

[Selection and seed stability](silhouette_stability.png) · [Cluster sizes](cluster_sizes.png) · [Numeric profiles](feature_profiles.png) · [Fraud-rate interpretation](fraud_rates.png) · [Metrics](cluster_metrics.csv) · [Full protocol](report.json)

## Limitations

Exploratory segmentation, not a fraud classifier. Silhouette is an internal geometry score, not accuracy. Mixed categorical/numeric distances, scaling, sampling and compression affect geometry. Repeated seeds measure initialization sensitivity, not temporal stability. Previously viewed test periods are not a fresh holdout. Elliptic uses only labelled-subset records and local anonymous features; availability cannot all be independently verified. Demographic features, where present, may drive groups; no adverse decisions should be made from clusters. Fraud-rate Wilson intervals are descriptive and assume independent cases; temporal dependence can make them optimistic.
