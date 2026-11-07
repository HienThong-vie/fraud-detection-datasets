# baf: AP, recall and FPR experiment

Validation-selected family: catboost. Default calibration FPR budget: 1%.

| Model | Test AP | Test recall | Actual test FPR |
|---|---:|---:|---:|
| logistic_regression | 0.1530 | 9.0% | 0.270% |
| random_forest | 0.1368 | 13.7% | 0.531% |
| extra_trees | 0.1525 | 10.1% | 0.347% |
| catboost | 0.2074 | 19.5% | 0.586% |

Maximize weakest chronological-block utility: 65% AP + 35% mean recall across three calibrated FPR budgets, minus selection FPR excess penalty and 0.10 positive training/selection AP gap. No ROC-AUC or accuracy weight.

Previously inspected test periods are exploratory, not a fresh final holdout. Validation FPR budgets are not guarantees on test or deployment. AP is average_precision_score, not trapezoidal integration. Historical label features, where used, require operational availability assumptions.

BAF train months 0-3, calibration month 4, selection month 5, test months 6-7. No finer timestamps available. This changes the training boundary from v1.

[All three operating budgets](operating_points.csv) · [Four-model chart](model_comparison.png) · [Recall / FPR chart](recall_at_fpr_budgets.png) · [Full report](report.json)

## Reproducibility correction

Numerical correction only: preserve the previously validation-selected configurations and champion; deterministic forest prediction and 12-decimal score precision. No test-based reselection, no refitting. Original selection utilities retained for audit.

## Evaluation support

| Period | Rows | Fraud cases |
|---|---:|---:|
| train | 500000 | 4825 |
| calibration | 50000 | 569 |
| selection | 50000 | 591 |
| test | 50000 | 702 |

Scores support analyst review and do not establish fraud. The 1% default budget is not a deployment recommendation; agree customer-friction and review-capacity limits before adoption.
