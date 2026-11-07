# sparkov: AP, recall and FPR experiment

Validation-selected family: catboost. Default calibration FPR budget: 1%.

| Model | Test AP | Test recall | Actual test FPR |
|---|---:|---:|---:|
| logistic_regression | 0.1413 | 53.9% | 0.988% |
| random_forest | 0.8093 | 93.8% | 1.000% |
| extra_trees | 0.8062 | 86.0% | 1.142% |
| catboost | 0.8859 | 94.8% | 1.076% |

Maximize weakest chronological-block utility: 65% AP + 35% mean recall across three calibrated FPR budgets, minus selection FPR excess penalty and 0.10 positive training/selection AP gap. No ROC-AUC or accuracy weight.

Previously inspected test periods are exploratory, not a fresh final holdout. Validation FPR budgets are not guarantees on test or deployment. AP is average_precision_score, not trapezoidal integration. Historical label features, where used, require operational availability assumptions.


[All three operating budgets](operating_points.csv) · [Four-model chart](model_comparison.png) · [Recall / FPR chart](recall_at_fpr_budgets.png) · [Full report](report.json)

## Reproducibility correction

Numerical correction only: preserve the previously validation-selected configurations and champion; deterministic forest prediction and 12-decimal score precision. No test-based reselection, no refitting. Original selection utilities retained for audit.

## Evaluation support

| Period | Rows | Fraud cases |
|---|---:|---:|
| train | 400000 | 2301 |
| calibration | 20001 | 117 |
| selection | 29999 | 180 |
| test | 50000 | 193 |

Scores support analyst review and do not establish fraud. The 1% default budget is not a deployment recommendation; agree customer-friction and review-capacity limits before adoption.
