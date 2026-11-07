# baf: fitting audit v3

Validation-selected champion: **catboost**, `d5_l230_earlystop`.

no large observed training/validation gap; this does not prove optimal fit.



AP values below use unchanged temporal partitions. Default threshold is calibrated at 1% FPR; actual test FPR may differ. Test results did not select the model.

| Family | Train AP | Validation AP | Test AP | Test recall | Test FPR | Gap assessment |
|---|---:|---:|---:|---:|---:|---|
| logistic_regression | 0.1089 | 0.1336 | 0.1580 | 10.4% | 0.292% | no large observed training/validation gap; this does not prove optimal fit |
| random_forest | 0.0880 | 0.0964 | 0.1391 | 13.8% | 0.499% | no large observed training/validation gap; this does not prove optimal fit |
| extra_trees | 0.1516 | 0.1214 | 0.1525 | 10.1% | 0.347% | no large observed training/validation gap; this does not prove optimal fit |
| catboost | 0.1698 | 0.1827 | 0.2109 | 20.9% | 0.607% | no large observed training/validation gap; this does not prove optimal fit |

## Evaluation support

Selected champion training rows: 500,000. Prepared partition counts follow; a retained recent-window incumbent may use fewer training rows.

| Prepared partition | Rows | Fraud cases |
|---|---:|---:|
| train | 500,000 | 4,825 |
| calibration | 50,000 | 569 |
| selection | 50,000 | 591 |
| test | 50,000 | 702 |

## Selection safeguards

v2 weakest temporal-block utility minus 0.25 positive train/selection AP-above-random gap, 0.15 positive ROC-AUC gap and 0.10 validation block AP drop. Per-family and champion floors: at least 95% incumbent validation AP, 85% incumbent recall at 1% calibration budget, and utility no more than 0.03 below incumbent. Incumbents remain eligible. No test-based selection.

Early stopping for new boosting candidates used only the last 15% of training time groups; the chosen iteration count was refitted on the full training period. Calibration and selection were not used for early stopping. Incumbents were evaluated under the same frozen policy.

The floors and gap penalties are declared heuristics, not statistical guarantees. A low gap does not establish that a model is not underfitting. Training AP is optimistic; AP also depends on fraud prevalence. AP-above-random is not a complete prevalence adjustment.

Previously inspected test periods are exploratory, not a fresh final holdout. Validation FPR budgets are not guarantees on test or deployment. AP is average_precision_score, not trapezoidal integration. Historical label features, where used, require operational availability assumptions. Fit-gap diagnostics are warnings, not guarantees of no over/underfitting. AP-above-random rescales the random baseline but does not fully correct prevalence shift. Reused validation can itself be overfit; use a new future holdout for confirmation. Learning capacity is screened, not mathematically proven optimal.

[All metrics CSV](model_comparison.csv) · [All FPR operating points](operating_points.csv) · [Fit diagnostics](fit_diagnostics.png) · [Four-model comparison](model_comparison.png) · [Frozen selection](selection.json) · [Independent verification](verification.json)
