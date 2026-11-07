# Four-supervised-model experiment card

Dataset: ieee. Selected family: xgboost.

| Model | Setting | Test PR-AUC | Test ROC-AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| logistic_regression | C0.1 | 0.2125 | 0.8061 | 0.2365 | 0.3017 | 0.2652 |
| random_forest | d12_leaf8 | 0.4455 | 0.8631 | 0.3981 | 0.4871 | 0.4381 |
| extra_trees | d12_leaf8 | 0.2364 | 0.8012 | 0.2655 | 0.3254 | 0.2924 |
| xgboost | prior_accepted_configuration | 0.4795 | 0.8876 | 0.4072 | 0.4935 | 0.4462 |

Maximize worst utility across two selection-validation subsets minus 0.03 * positive train/validation PR-AUC gap. Utility: 45% F1, 15% min(precision,recall), 20% PR-AUC, 20% ROC-AUC. Calibration threshold within 95% of best F1, favor balanced precision/recall, calibration FPR <= 5%.

Existing temporal test periods have already been inspected. Exploratory, not an untouched final evaluation. Validation subsets are stratified because exported tables do not consistently retain event timestamps. No test metric chooses a candidate. BAF is Base only; variants retain original reports. No production checkpoint replacement.

[Full protocol and results](outputs/supervised_search_v1/report.json)
[Training instructions](training_model/FOUR_SUPERVISED.md)

The original README retains dataset source, licence and attribution. Scores support review; they do not establish fraud. Existing production artifacts were retained separately.
