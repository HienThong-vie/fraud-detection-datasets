# IEEE: exploratory clustering

MiniBatch K-means selected **k=3** using training-only geometry. Training silhouette: 0.467; mean seed ARI: 0.868.

Passed the stated size/repeatability floors; this does not establish real-world or fraud relevance.

This is the second method type alongside the four supervised classifiers. It is not a fifth fraud classifier, and cluster IDs do not mean fraud/legitimate.

| Cluster | Train share | Train fraud rate | Test share* | Test fraud rate* |
|---|---:|---:|---:|---:|
| 0 | 84.3% | 2.42% | 83.3% | 2.30% |
| 1 | 11.8% | 12.04% | 12.9% | 11.59% |
| 2 | 4.0% | 1.61% | 3.8% | 1.78% |

*Previously viewed test periods; interpretation only. Neither fraud rates nor test data selected k, preprocessing or seed. Differences are associations, not causal evidence.

## Training feature interpretation

- Cluster 0: id_17 -0.32 training SD, C7 -0.27 training SD, C12 -0.24 training SD.
- Cluster 1: id_17 +2.35 training SD, C7 +1.86 training SD, C12 +1.62 training SD.
- Cluster 2: C13 +4.74 training SD, C14 +4.72 training SD, C6 +4.68 training SD.

Profiles describe training numeric means relative to the training-fit scale. Inspect categorical profiles as well; numeric summaries alone do not fully explain mixed-feature distances.

## Reproduce

Run from the suite root:

```bash
python training_model/clustering.py --dataset ieee
```

Completed folders are preserved. Use a new versioned directory for a new experiment.

## Evidence

[Selection and seed stability](silhouette_stability.png) · [Cluster sizes](cluster_sizes.png) · [Numeric profiles](feature_profiles.png) · [Fraud-rate interpretation](fraud_rates.png) · [Metrics](cluster_metrics.csv) · [Full protocol](report.json)

## Limitations

Exploratory segmentation, not a fraud classifier. Silhouette is an internal geometry score, not accuracy. Mixed categorical/numeric distances, scaling, sampling and compression affect geometry. Repeated seeds measure initialization sensitivity, not temporal stability. Previously viewed test periods are not a fresh holdout. Elliptic uses only labelled-subset records and local anonymous features; availability cannot all be independently verified. Demographic features, where present, may drive groups; no adverse decisions should be made from clusters. Fraud-rate Wilson intervals are descriptive and assume independent cases; temporal dependence can make them optimistic.
