# ulb: AP, recall and FPR experiment

Validation-selected family: xgboost. Default calibration FPR budget: 1%.

| Model | Test AP | Test recall | Actual test FPR |
|---|---:|---:|---:|
| logistic_regression | 0.7579 | 86.4% | 0.713% |
| random_forest | 0.8167 | 87.9% | 0.833% |
| extra_trees | 0.7984 | 87.9% | 0.779% |
| xgboost | 0.8027 | 87.9% | 0.761% |

Maximize weakest chronological-block utility: 65% AP + 35% mean recall across three calibrated FPR budgets, minus selection FPR excess penalty and 0.10 positive training/selection AP gap. No ROC-AUC or accuracy weight.

Previously inspected test periods are exploratory, not a fresh final holdout. Validation FPR budgets are not guarantees on test or deployment. AP is average_precision_score, not trapezoidal integration. Historical label features, where used, require operational availability assumptions.


[All three operating budgets](operating_points.csv) · [Four-model chart](model_comparison.png) · [Recall / FPR chart](recall_at_fpr_budgets.png) · [Full report](report.json)

## Reproducibility correction

Numerical correction only: preserve the previously validation-selected configurations and champion; deterministic forest prediction and 12-decimal score precision. No test-based reselection, no refitting. Original selection utilities retained for audit.

## Evaluation support

| Period | Rows | Fraud cases |
|---|---:|---:|
| train | 170884 | 360 |
| calibration | 20082 | 19 |
| selection | 29918 | 31 |
| test | 50000 | 66 |

Validation has fewer than 100 fraud cases in at least one stage. Small count changes can materially change recall/AP; do not treat small score differences as definitive.

Scores support analyst review and do not establish fraud. The 1% default budget is not a deployment recommendation; agree customer-friction and review-capacity limits before adoption.
