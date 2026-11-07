# PAYSIM selected model

Active classifier: **random_forest**, `feature_policy_v4`.

Selection uses the existing chronological-validation winner. Other families remain comparison evidence, not active selected models.

| Calibration FPR budget | Threshold | Test AP* | Test recall* | Actual test FPR* | Precision* |
|---|---:|---:|---:|---:|---:|
| 5% | 0.004092325896 | 0.5261 | 83.56% | 9.992% | 10.19% |
| 1% | 0.05660013187 | 0.5261 | 46.49% | 0.598% | 51.32% |

*Previously inspected test periods. Headline: 5% calibration budget; 1% secondary. Actual later FPR can differ. No automatic customer decision is authorised.

Checkpoint SHA-256: `ea67b343f9ad118ab5f120f380db67dad8e2b9bef8a99c0b9496f1254fc797ef`.

[Conda setup and commands](experiment_suite/CONDA_GUIDE.md)

The `selected` API/CLI variant resolves only this dataset’s frozen champion. Prediction examples and input requirements are available in the dashboard.
