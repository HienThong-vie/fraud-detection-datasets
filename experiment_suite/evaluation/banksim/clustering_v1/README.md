# BANKSIM: exploratory clustering

MiniBatch K-means selected **k=8** using training-only geometry. Training silhouette: 0.312; mean seed ARI: 0.697.

Passed the stated size/repeatability floors; this does not establish real-world or fraud relevance.

This is the second method type alongside the four supervised classifiers. It is not a fifth fraud classifier, and cluster IDs do not mean fraud/legitimate.

| Cluster | Train share | Train fraud rate | Test share* | Test fraud rate* |
|---|---:|---:|---:|---:|
| 0 | 10.7% | 0.07% | 11.1% | 0.07% |
| 1 | 22.6% | 0.13% | 22.7% | 0.16% |
| 2 | 5.9% | 3.16% | 5.9% | 2.28% |
| 3 | 2.0% | 49.41% | 1.9% | 43.24% |
| 4 | 14.0% | 0.11% | 13.9% | 0.12% |
| 5 | 13.2% | 0.09% | 12.8% | 0.11% |
| 6 | 4.6% | 0.07% | 4.5% | 0.04% |
| 7 | 26.9% | 0.15% | 27.2% | 0.14% |

*Previously viewed test periods; interpretation only. Neither fraud rates nor test data selected k, preprocessing or seed. Differences are associations, not causal evidence.

## Training feature interpretation

- Cluster 0: amount -0.19 training SD.
- Cluster 1: amount -0.39 training SD.
- Cluster 2: amount +1.54 training SD.
- Cluster 3: amount +5.29 training SD.
- Cluster 4: amount -0.20 training SD.
- Cluster 5: amount -0.20 training SD.
- Cluster 6: amount -0.19 training SD.
- Cluster 7: amount -0.09 training SD.

Profiles describe training numeric means relative to the training-fit scale. Inspect categorical profiles as well; numeric summaries alone do not fully explain mixed-feature distances.

## Reproduce

Run from the suite root:

```bash
python training_model/clustering.py --dataset banksim
```

Completed folders are preserved. Use a new versioned directory for a new experiment.

## Evidence

[Selection and seed stability](silhouette_stability.png) · [Cluster sizes](cluster_sizes.png) · [Numeric profiles](feature_profiles.png) · [Fraud-rate interpretation](fraud_rates.png) · [Metrics](cluster_metrics.csv) · [Full protocol](report.json)

## Limitations

Exploratory segmentation, not a fraud classifier. Silhouette is an internal geometry score, not accuracy. Mixed categorical/numeric distances, scaling, sampling and compression affect geometry. Repeated seeds measure initialization sensitivity, not temporal stability. Previously viewed test periods are not a fresh holdout. Elliptic uses only labelled-subset records and local anonymous features; availability cannot all be independently verified. Demographic features, where present, may drive groups; no adverse decisions should be made from clusters. Fraud-rate Wilson intervals are descriptive and assume independent cases; temporal dependence can make them optimistic.
