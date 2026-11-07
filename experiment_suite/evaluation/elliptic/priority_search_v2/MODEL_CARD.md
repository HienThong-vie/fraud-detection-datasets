# elliptic: AP, recall and FPR experiment

Validation-selected family: xgboost. Default calibration FPR budget: 1%.

| Model | Test AP | Test recall | Actual test FPR |
|---|---:|---:|---:|
| logistic_regression | 0.2289 | 25.2% | 2.075% |
| random_forest | 0.5449 | 47.8% | 1.767% |
| extra_trees | 0.5366 | 48.0% | 2.146% |
| xgboost | 0.5333 | 48.0% | 1.210% |

Maximize weakest chronological-block utility: 65% AP + 35% mean recall across three calibrated FPR budgets, minus selection FPR excess penalty and 0.10 positive training/selection AP gap. No ROC-AUC or accuracy weight.

Previously inspected test periods are exploratory, not a fresh final holdout. Validation FPR budgets are not guarantees on test or deployment. AP is average_precision_score, not trapezoidal integration. Historical label features, where used, require operational availability assumptions.

Elliptic includes a recent-history training candidate (steps >=20) to examine temporal stability; no future labels or graph outcomes added.

[All three operating budgets](operating_points.csv) · [Four-model chart](model_comparison.png) · [Recall / FPR chart](recall_at_fpr_budgets.png) · [Full report](report.json)

## Reproducibility correction

Numerical correction only: preserve the previously validation-selected configurations and champion; deterministic forest prediction and 12-decimal score precision. No test-based reselection, no refitting. Original selection utilities retained for audit.

## Evaluation support

| Period | Rows | Fraud cases |
|---|---:|---:|
| train | 29894 | 3462 |
| calibration | 3049 | 215 |
| selection | 4780 | 460 |
| test | 8841 | 408 |

Scores support analyst review and do not establish fraud. The 1% default budget is not a deployment recommendation; agree customer-friction and review-capacity limits before adoption.
