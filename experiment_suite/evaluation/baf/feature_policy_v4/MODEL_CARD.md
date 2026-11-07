# baf: feature and threshold-policy experiment v4

Selected champion: **catboost**, `features_setting0_windowNone`.

Default display is the 1% calibration-FPR point policy without a review-capacity limit. This is an experimental display choice, not a bank deployment recommendation.

| Family | Train AP | Selection AP | Test AP* | Test recall* | Test FPR* |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.1191 | 0.1496 | 0.1746 | 16.0% | 0.515% |
| random_forest | 0.1211 | 0.1180 | 0.1534 | 12.0% | 0.507% |
| extra_trees | 0.1516 | 0.1214 | 0.1525 | 10.1% | 0.347% |
| catboost | 0.1605 | 0.1820 | 0.2127 | 21.8% | 0.619% |

*Previously inspected test period; exploratory, not a fresh final holdout. For case-control PaySim training, train AP is inverse-inclusion-weighted, not the inflated unweighted minority-enriched-sample AP.

## Candidate feature and data changes

These changes were tested; selected families can retain a v3 incumbent if validation does not justify replacement. The variant and saved pipeline identify the actual recipe.

Missingness flags, nonnegative log transforms, velocity ratios and device/payment interactions. No protected columns reintroduced; month-level timing only.

## Threshold policy choices

Five FPR budgets: 0.5%, 1%, 2%, 3%, 5%. Optional calibration review limits: 50, 100, 200, 500 alerts per 10,000 cases, or no capacity constraint. These are scenario comparisons; the team must agree a business budget before choosing one.

`point` uses the lowest tie-safe calibration threshold meeting the empirical limits. `ucb95` additionally uses an approximate one-sided 95% Wilson upper calibration-FPR bound. Sparse-data infeasibility is reported, not hidden. Adaptive thresholds, temporal dependence and drift prevent interpreting this as a future 95% FPR guarantee.

Threshold changes do not improve AP; they trade recall, false positives, precision and review volume for a fixed score ranking. Actual test/selection limits can be exceeded.

## Temporal validation

Two forward refits inside the training period are performed for each selected family. Fit, threshold-calibration and evaluation time groups are disjoint and ordered. Backtest fitting is capped at 100,000 rows, preserving training positives when feasible and correcting negative inclusion weights; forward calibration/evaluation use natural prevalence samples capped at 50,000 rows. These diagnostics do not select the champion and are not an independent outer validation after repeated tuning. When fewer than three training groups exist (BAF month-level folds), boosting uses a fixed <=300-round cap rather than inventing within-month timestamps or random early-stopping splits.

Reassess v3 incumbents and new feature candidates on identical calibration/selection rows. Maximize weakest chronological selection-block utility (65% AP + 35% mean recall at five FPR budgets minus FPR-excess penalty), minus 0.15 positive training/validation AP-above-random gap and 0.10 ROC-AUC gap. New candidates must retain >=95% incumbent validation AP, >=85% recall at 1% budget, and utility within 0.03. No test selection. Two forward training-period refits per selected family are diagnostic, not a fresh outer holdout or a champion selection criterion.

Previously viewed test periods and reused validation are exploratory. No guarantee of optimal fitting, future FPR, precision or review capacity. Wilson bounds are approximate one-sided calibration diagnostics, not future-period guarantees under drift or dependence. No deployment threshold is recommended without business constraints. PaySim training metrics are inverse-inclusion-weighted; unweighted enriched-training prevalence differs. Supplied anonymous feature availability cannot all be independently verified.

[Four-model comparison](model_comparison.png) · [Candidate feature comparison](candidate_feature_comparison.png) · [Validation-only candidate table](candidate_validation.csv) · [Temporal validation](temporal_validation.png) · [Threshold trade-offs](threshold_policy_tradeoffs.png) · [Review-capacity trade-offs](review_capacity_tradeoffs.png) · [All policy metrics CSV](threshold_policies.csv) · [Full report](report.json) · [Frozen selection](selection.json) · [Independent verification](verification.json)
