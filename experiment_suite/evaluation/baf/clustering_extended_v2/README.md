# BAF extended clustering

Selected k=5; silhouette 0.0669, Davies–Bouldin 2.7712. Weak or unstable geometry; no candidate met both size and repeatability floors.

Training only: minimum1% cluster size and mean seed ARI>=0.6; if none pass, explicit relaxed-quality fallback. Minimize .5 silhouette descending rank + .3 Davies-Bouldin ascending rank + .2 distance-to-inertia-elbow rank. Ties prefer smaller k. Elbow maximizes distance from normalized inertia chord. Weights are engineering heuristics, not proven optimal.

Training-only median/mode imputation, percentile clipping, scaling and encoding; 20D train-fitted SVD where needed. Silhouette/DB use fixed uniform 2000-row training sample; inertia uses training-fit sample (maximum80000).

[Selection diagnostics](selection_diagnostics.png) · [All k metrics](k_comparison.csv) · [All cluster counts and rates](cluster_metrics_all_k.csv) · [Feature profiles](feature_profiles.png)

Illicit/fraud rates use labelled rows only; unknown records are not negatives. Training, validation and test assignments are reported separately.

Descriptive clusters, not fraud predictions. Unknown nodes are never treated as licit. Illicit rates divide by labelled rows only. Initialisation stability is not temporal stability. Test periods previously viewed. Anonymous local-feature availability cannot all be verified. Geometry depends on preprocessing and SVD.

Run `python training_model/extended_clustering.py --dataset baf` from the suite root.
