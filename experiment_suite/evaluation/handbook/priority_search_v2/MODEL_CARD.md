# handbook: AP, recall and FPR experiment

Validation-selected family: random_forest. Default calibration FPR budget: 1%.

| Model | Test AP | Test recall | Actual test FPR |
|---|---:|---:|---:|
| logistic_regression | 0.5628 | 74.3% | 0.959% |
| random_forest | 0.7699 | 78.8% | 0.979% |
| extra_trees | 0.7232 | 77.5% | 0.967% |
| xgboost | 0.7465 | 78.1% | 0.888% |

Maximize weakest chronological-block utility: 65% AP + 35% mean recall across three calibrated FPR budgets, minus selection FPR excess penalty and 0.10 positive training/selection AP gap. No ROC-AUC or accuracy weight.

Previously inspected test periods are exploratory, not a fresh final holdout. Validation FPR budgets are not guarantees on test or deployment. AP is average_precision_score, not trapezoidal integration. Historical label features, where used, require operational availability assumptions.

Handbook rolling 1/7/30-day history excludes current and simultaneous transactions. Terminal fraud labels are used only after >7 days in an explicit online-replay scenario; past validation/test outcomes can enter later features after this delay. Not a frozen no-feedback deployment.

[All three operating budgets](operating_points.csv) · [Four-model chart](model_comparison.png) · [Recall / FPR chart](recall_at_fpr_budgets.png) · [Full report](report.json)

## Reproducibility correction

Numerical correction only: preserve the previously validation-selected configurations and champion; deterministic forest prediction and 12-decimal score precision. No test-based reselection, no refitting. Original selection utilities retained for audit.

## Evaluation support

| Period | Rows | Fraud cases |
|---|---:|---:|
| train | 300000 | 2417 |
| calibration | 20002 | 156 |
| selection | 29998 | 279 |
| test | 50000 | 448 |

Scores support analyst review and do not establish fraud. The 1% default budget is not a deployment recommendation; agree customer-friction and review-capacity limits before adoption.
