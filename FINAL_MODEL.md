# BANKSIM selected model

Active classifier: **catboost**, `priority_search_v2`.

Selection uses the existing chronological-validation winner. Other families remain comparison evidence, not active selected models.

| Calibration FPR budget | Threshold | Test AP* | Test recall* | Actual test FPR* | Precision* |
|---|---:|---:|---:|---:|---:|
| 5% | 0.009907873289 | 0.8026 | 94.26% | 5.212% | 16.49% |
| 1% | 0.071567336827 | 0.8026 | 80.56% | 0.827% | 51.54% |

*Previously inspected test periods. Headline: 5% calibration budget; 1% secondary. Actual later FPR can differ. No automatic customer decision is authorised.

Checkpoint SHA-256: `0c54a822f4272513b5570c581e136449bbf1f8a4013a2d9aed0735e51c15d56c`.

[Conda setup and commands](experiment_suite/CONDA_GUIDE.md)

The `selected` API/CLI variant resolves only this dataset’s frozen champion. Prediction examples and input requirements are available in the dashboard.
