# PAYSIM: exploratory clustering

MiniBatch K-means selected **k=8** using training-only geometry. Training silhouette: 0.446; mean seed ARI: 0.864.

Passed the stated size/repeatability floors; this does not establish real-world or fraud relevance.

This is the second method type alongside the four supervised classifiers. It is not a fifth fraud classifier, and cluster IDs do not mean fraud/legitimate.

| Cluster | Train share | Train fraud rate | Test share* | Test fraud rate* |
|---|---:|---:|---:|---:|
| 0 | 10.7% | 0.00% | 9.9% | 0.00% |
| 1 | 18.5% | 0.00% | 21.6% | 0.01% |
| 2 | 1.9% | 1.23% | 1.6% | 25.57% |
| 3 | 17.9% | 0.06% | 17.7% | 0.89% |
| 4 | 12.8% | 0.06% | 15.6% | 0.80% |
| 5 | 17.5% | 0.12% | 14.4% | 2.35% |
| 6 | 16.2% | 0.02% | 14.8% | 0.51% |
| 7 | 4.4% | 0.35% | 4.6% | 5.21% |

*Previously viewed test periods; interpretation only. Neither fraud rates nor test data selected k, preprocessing or seed. Differences are associations, not causal evidence.

## Training feature interpretation

- Cluster 0: hour -0.90 training SD, amount +0.05 training SD.
- Cluster 1: hour +0.91 training SD, amount -0.59 training SD.
- Cluster 2: amount +5.37 training SD, hour -0.29 training SD.
- Cluster 3: hour +0.75 training SD, amount -0.02 training SD.
- Cluster 4: hour +0.81 training SD, amount +0.00 training SD.
- Cluster 5: hour -0.90 training SD, amount +0.10 training SD.
- Cluster 6: hour -0.93 training SD, amount -0.58 training SD.
- Cluster 7: amount +1.86 training SD, hour -0.07 training SD.

Profiles describe training numeric means relative to the training-fit scale. Inspect categorical profiles as well; numeric summaries alone do not fully explain mixed-feature distances.

## Reproduce

Run from the suite root:

```bash
python training_model/clustering.py --dataset paysim
```

Completed folders are preserved. Use a new versioned directory for a new experiment.

## Evidence

[Selection and seed stability](silhouette_stability.png) · [Cluster sizes](cluster_sizes.png) · [Numeric profiles](feature_profiles.png) · [Fraud-rate interpretation](fraud_rates.png) · [Metrics](cluster_metrics.csv) · [Full protocol](report.json)

## Limitations

Exploratory segmentation, not a fraud classifier. Silhouette is an internal geometry score, not accuracy. Mixed categorical/numeric distances, scaling, sampling and compression affect geometry. Repeated seeds measure initialization sensitivity, not temporal stability. Previously viewed test periods are not a fresh holdout. Elliptic uses only labelled-subset records and local anonymous features; availability cannot all be independently verified. Demographic features, where present, may drive groups; no adverse decisions should be made from clusters. Fraud-rate Wilson intervals are descriptive and assume independent cases; temporal dependence can make them optimistic.
