# BAF selected model

Active classifier: **catboost**, `feature_policy_v4`.

Selection uses the existing chronological-validation winner. Other families remain comparison evidence, not active selected models.

| Calibration FPR budget | Threshold | Test AP* | Test recall* | Actual test FPR* | Precision* |
|---|---:|---:|---:|---:|---:|
| 5% | 0.300830880968 | 0.2127 | 51.14% | 3.584% | 16.89% |
| 1% | 0.567526347484 | 0.2127 | 21.79% | 0.619% | 33.41% |

*Previously inspected test periods. Headline: 5% calibration budget; 1% secondary. Actual later FPR can differ. No automatic customer decision is authorised.

Checkpoint SHA-256: `6655ca84af79c1a6bbdf8b5a61e5fd8178320476ef257c34da7b4f1f8a2927a3`.

[Conda setup and commands](experiment_suite/CONDA_GUIDE.md)

The `selected` API/CLI variant resolves only this dataset’s frozen champion. Prediction examples and input requirements are available in the dashboard.

The production model and card now match v4 CatBoost. The previous model is preserved under `legacy_7919ca0/`. Both original 27-field inputs and the 60-field engineered schema are supported by the BAF production service.

[Age-group fairness](experiment_suite/evaluation/baf/final_evidence/fairness_age.csv) · [Processed-data manifest](dataset/Base_final_model_data_manifest.json)

Observed FPR remains higher in the age≥50 group; age exclusion does not establish fairness.

[Extended clustering diagnostics](experiment_suite/evaluation/baf/clustering_extended_v2/README.md)
