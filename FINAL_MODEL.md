# HANDBOOK selected model

Active classifier: **random_forest**, `priority_search_v2`.

Selection uses the existing chronological-validation winner. Other families remain comparison evidence, not active selected models.

| Calibration FPR budget | Threshold | Test AP* | Test recall* | Actual test FPR* | Precision* |
|---|---:|---:|---:|---:|---:|
| 5% | 0.237218133143 | 0.7699 | 80.58% | 3.671% | 16.56% |
| 1% | 0.386661616201 | 0.7699 | 78.79% | 0.979% | 42.12% |

*Previously inspected test periods. Headline: 5% calibration budget; 1% secondary. Actual later FPR can differ. No automatic customer decision is authorised.

Checkpoint SHA-256: `ec150d98357dcf05c1342028d167116dd645e75f4003ece4ae0784c84bf82c4b`.

[Conda setup and commands](experiment_suite/CONDA_GUIDE.md)

The `selected` API/CLI variant resolves only this dataset’s frozen champion. Prediction examples and input requirements are available in the dashboard.
