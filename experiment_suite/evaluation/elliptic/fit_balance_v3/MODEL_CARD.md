# elliptic: fitting audit v3

Validation-selected champion: **xgboost**, `v2_incumbent_d3_weightpower0_recent_steps20plus`.

no large observed training/validation gap; this does not prove optimal fit.

Large validation-to-test AP drop: investigate temporal/prevalence changes. This exploratory diagnostic does not identify the cause and did not select the model.

AP values below use unchanged temporal partitions. Default threshold is calibrated at 1% FPR; actual test FPR may differ. Test results did not select the model.

| Family | Train AP | Validation AP | Test AP | Test recall | Test FPR | Gap assessment |
|---|---:|---:|---:|---:|---:|---|
| logistic_regression | 0.6085 | 0.5320 | 0.1579 | 35.8% | 5.775% | no large observed training/validation gap; this does not prove optimal fit |
| random_forest | 0.9986 | 0.8984 | 0.5449 | 47.8% | 1.767% | large training/validation gap; overfitting or temporal shift remains |
| extra_trees | 0.9778 | 0.8970 | 0.5366 | 48.0% | 2.146% | no large observed training/validation gap; this does not prove optimal fit |
| xgboost | 0.9939 | 0.9049 | 0.5333 | 48.0% | 1.210% | no large observed training/validation gap; this does not prove optimal fit |

## Evaluation support

Selected champion training rows: 11,905. Prepared partition counts follow; a retained recent-window incumbent may use fewer training rows.

| Prepared partition | Rows | Fraud cases |
|---|---:|---:|
| train | 29,894 | 3,462 |
| calibration | 3,049 | 215 |
| selection | 4,780 | 460 |
| test | 8,841 | 408 |

## Selection safeguards

v2 weakest temporal-block utility minus 0.25 positive train/selection AP-above-random gap, 0.15 positive ROC-AUC gap and 0.10 validation block AP drop. Per-family and champion floors: at least 95% incumbent validation AP, 85% incumbent recall at 1% calibration budget, and utility no more than 0.03 below incumbent. Incumbents remain eligible. No test-based selection.

Early stopping for new boosting candidates used only the last 15% of training time groups; the chosen iteration count was refitted on the full training period. Calibration and selection were not used for early stopping. Incumbents were evaluated under the same frozen policy.

The floors and gap penalties are declared heuristics, not statistical guarantees. A low gap does not establish that a model is not underfitting. Training AP is optimistic; AP also depends on fraud prevalence. AP-above-random is not a complete prevalence adjustment.

Previously inspected test periods are exploratory, not a fresh final holdout. Validation FPR budgets are not guarantees on test or deployment. AP is average_precision_score, not trapezoidal integration. Historical label features, where used, require operational availability assumptions. Fit-gap diagnostics are warnings, not guarantees of no over/underfitting. AP-above-random rescales the random baseline but does not fully correct prevalence shift. Reused validation can itself be overfit; use a new future holdout for confirmation. Learning capacity is screened, not mathematically proven optimal.

[All metrics CSV](model_comparison.csv) · [All FPR operating points](operating_points.csv) · [Fit diagnostics](fit_diagnostics.png) · [Four-model comparison](model_comparison.png) · [Frozen selection](selection.json) · [Independent verification](verification.json)
