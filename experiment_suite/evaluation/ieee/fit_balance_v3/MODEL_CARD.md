# ieee: fitting audit v3

Validation-selected champion: **xgboost**, `d5_child40_earlystop`.

large training/validation gap; overfitting or temporal shift remains.



AP values below use unchanged temporal partitions. Default threshold is calibrated at 1% FPR; actual test FPR may differ. Test results did not select the model.

| Family | Train AP | Validation AP | Test AP | Test recall | Test FPR | Gap assessment |
|---|---:|---:|---:|---:|---:|---|
| logistic_regression | 0.3163 | 0.2525 | 0.2378 | 26.9% | 1.922% | no large observed training/validation gap; this does not prove optimal fit |
| random_forest | 0.6272 | 0.4153 | 0.4627 | 39.9% | 1.132% | large training/validation gap; overfitting or temporal shift remains |
| extra_trees | 0.4618 | 0.3053 | 0.3020 | 24.8% | 1.067% | large training/validation gap; overfitting or temporal shift remains |
| xgboost | 0.6841 | 0.4810 | 0.4933 | 44.0% | 1.370% | large training/validation gap; overfitting or temporal shift remains |

## Evaluation support

Selected champion training rows: 300,000. Prepared partition counts follow; a retained recent-window incumbent may use fewer training rows.

| Prepared partition | Rows | Fraud cases |
|---|---:|---:|
| train | 300,000 | 10,551 |
| calibration | 16,011 | 555 |
| selection | 23,989 | 819 |
| test | 40,000 | 1,392 |

## Selection safeguards

v2 weakest temporal-block utility minus 0.25 positive train/selection AP-above-random gap, 0.15 positive ROC-AUC gap and 0.10 validation block AP drop. Per-family and champion floors: at least 95% incumbent validation AP, 85% incumbent recall at 1% calibration budget, and utility no more than 0.03 below incumbent. Incumbents remain eligible. No test-based selection.

Early stopping for new boosting candidates used only the last 15% of training time groups; the chosen iteration count was refitted on the full training period. Calibration and selection were not used for early stopping. Incumbents were evaluated under the same frozen policy.

The floors and gap penalties are declared heuristics, not statistical guarantees. A low gap does not establish that a model is not underfitting. Training AP is optimistic; AP also depends on fraud prevalence. AP-above-random is not a complete prevalence adjustment.

Previously inspected test periods are exploratory, not a fresh final holdout. Validation FPR budgets are not guarantees on test or deployment. AP is average_precision_score, not trapezoidal integration. Historical label features, where used, require operational availability assumptions. Fit-gap diagnostics are warnings, not guarantees of no over/underfitting. AP-above-random rescales the random baseline but does not fully correct prevalence shift. Reused validation can itself be overfit; use a new future holdout for confirmation. Learning capacity is screened, not mathematically proven optimal.

[All metrics CSV](model_comparison.csv) · [All FPR operating points](operating_points.csv) · [Fit diagnostics](fit_diagnostics.png) · [Four-model comparison](model_comparison.png) · [Frozen selection](selection.json) · [Independent verification](verification.json)
