# SPARKOV selected model

Active classifier: **catboost**, `fit_balance_v3`.

Selection uses the existing chronological-validation winner. Other families remain comparison evidence, not active selected models.

| Calibration FPR budget | Threshold | Test AP* | Test recall* | Actual test FPR* | Precision* |
|---|---:|---:|---:|---:|---:|
| 5% | 0.022885146267 | 0.8859 | 97.93% | 4.939% | 7.13% |
| 1% | 0.116677166993 | 0.8859 | 94.82% | 1.076% | 25.45% |

*Previously inspected test periods. Headline: 5% calibration budget; 1% secondary. Actual later FPR can differ. No automatic customer decision is authorised.

Checkpoint SHA-256: `f4e13f1757bb24b7c71be80eff995dfad875ae99605345713e039b80a6365532`.

[Conda setup and commands](experiment_suite/CONDA_GUIDE.md)

The `selected` API/CLI variant resolves only this dataset’s frozen champion. Prediction examples and input requirements are available in the dashboard.
