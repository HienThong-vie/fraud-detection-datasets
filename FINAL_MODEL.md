# IEEE selected model

Active classifier: **xgboost**, `feature_policy_v4`.

Selection uses the existing chronological-validation winner. Other families remain comparison evidence, not active selected models.

| Calibration FPR budget | Threshold | Test AP* | Test recall* | Actual test FPR* | Precision* |
|---|---:|---:|---:|---:|---:|
| 5% | 0.263438373804 | 0.5235 | 65.45% | 5.828% | 28.82% |
| 1% | 0.532332658768 | 0.5235 | 45.19% | 1.409% | 53.62% |

*Previously inspected test periods. Headline: 5% calibration budget; 1% secondary. Actual later FPR can differ. No automatic customer decision is authorised.

Checkpoint SHA-256: `d1d6c5e52d98795f921b855d5556691e6c58686369a30bdfdb9e0445f35bcfa8`.

[Conda setup and commands](experiment_suite/CONDA_GUIDE.md)

The `selected` API/CLI variant resolves only this dataset’s frozen champion. Prediction examples and input requirements are available in the dashboard.
