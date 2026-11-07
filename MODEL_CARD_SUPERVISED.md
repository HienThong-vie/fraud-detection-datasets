# Four-supervised-model experiment card

Dataset: elliptic. Selected family: xgboost.

| Model | Setting | Test PR-AUC | Test ROC-AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| logistic_regression | C0.1 | 0.1989 | 0.8508 | 0.2263 | 0.4804 | 0.3077 |
| random_forest | d12_leaf8 | 0.5449 | 0.8555 | 0.5552 | 0.4804 | 0.5151 |
| extra_trees | d12_leaf8 | 0.5366 | 0.8287 | 0.5199 | 0.4804 | 0.4994 |
| xgboost | prior_accepted_configuration | 0.5556 | 0.8547 | 0.5650 | 0.4902 | 0.5249 |

Maximize worst utility across two selection-validation subsets minus 0.03 * positive train/validation PR-AUC gap. Utility: 45% F1, 15% min(precision,recall), 20% PR-AUC, 20% ROC-AUC. Calibration threshold within 95% of best F1, favor balanced precision/recall, calibration FPR <= 5%.

Existing temporal test periods have already been inspected. Exploratory, not an untouched final evaluation. Validation subsets are stratified because exported tables do not consistently retain event timestamps. No test metric chooses a candidate. BAF is Base only; variants retain original reports. No production checkpoint replacement.

[Full protocol and results](outputs/supervised_search_v1/report.json)
[Training instructions](training_model/FOUR_SUPERVISED.md)

The original README retains dataset source, licence and attribution. Scores support review; they do not establish fraud. Existing production artifacts were retained separately.
