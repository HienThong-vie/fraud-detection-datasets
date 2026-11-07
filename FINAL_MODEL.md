# ELLIPTIC selected model

Active classifier: **xgboost**, `feature_policy_v4`.

Selection uses the existing chronological-validation winner. Other families remain comparison evidence, not active selected models.

| Calibration FPR budget | Threshold | Test AP* | Test recall* | Actual test FPR* | Precision* |
|---|---:|---:|---:|---:|---:|
| 5% | 0.101034425199 | 0.5333 | 51.47% | 6.724% | 27.03% |
| 1% | 0.392538309097 | 0.5333 | 48.04% | 1.210% | 65.77% |

*Previously inspected test periods. Headline: 5% calibration budget; 1% secondary. Actual later FPR can differ. No automatic customer decision is authorised.

Checkpoint SHA-256: `010494ec8e74205e4e8da0464b27e9e532b742a85f7f428eb1286b18e3221f8f`.

[Conda setup and commands](experiment_suite/CONDA_GUIDE.md)

The `selected` API/CLI variant resolves only this dataset’s frozen champion. Prediction examples and input requirements are available in the dashboard.

The selected classifier remains an offline benchmark. The new clustering experiment includes all 203,769 nodes, including 157,205 unknown labels; unknown is never counted as licit.

[Extended clustering diagnostics](experiment_suite/evaluation/elliptic/clustering_extended_v2/README.md)
