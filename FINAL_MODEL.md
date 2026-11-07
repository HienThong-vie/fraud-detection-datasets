# ULB selected model

Active classifier: **xgboost**, `fit_balance_v3`.

Selection uses the existing chronological-validation winner. Other families remain comparison evidence, not active selected models.

| Calibration FPR budget | Threshold | Test AP* | Test recall* | Actual test FPR* | Precision* |
|---|---:|---:|---:|---:|---:|
| 5% | 0.398901045322 | 0.7644 | 89.39% | 4.127% | 2.78% |
| 1% | 0.494893312454 | 0.7644 | 84.85% | 0.845% | 11.72% |

*Previously inspected test periods. Headline: 5% calibration budget; 1% secondary. Actual later FPR can differ. No automatic customer decision is authorised.

Checkpoint SHA-256: `21fc9d21f97716aa24f0c294e86f2f6242e52e29701dcd7f4052d7d011dd29f0`.

[Conda setup and commands](experiment_suite/CONDA_GUIDE.md)

The `selected` API/CLI variant resolves only this dataset’s frozen champion. Prediction examples and input requirements are available in the dashboard.
