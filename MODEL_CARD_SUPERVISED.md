# Four-supervised-model experiment card

Dataset: handbook. Selected family: random_forest.

| Model | Setting | Test PR-AUC | Test ROC-AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| logistic_regression | C1 | 0.2297 | 0.6337 | 0.2552 | 0.2478 | 0.2514 |
| random_forest | prior_accepted_configuration | 0.2511 | 0.6316 | 0.9065 | 0.2165 | 0.3495 |
| extra_trees | d12_leaf8 | 0.1345 | 0.6447 | 0.5506 | 0.1094 | 0.1825 |
| xgboost | d5_weightpower0.5 | 0.2381 | 0.6491 | 0.5139 | 0.2879 | 0.3691 |

Maximize worst utility across two selection-validation subsets minus 0.03 * positive train/validation PR-AUC gap. Utility: 45% F1, 15% min(precision,recall), 20% PR-AUC, 20% ROC-AUC. Calibration threshold within 95% of best F1, favor balanced precision/recall, calibration FPR <= 5%.

Existing temporal test periods have already been inspected. Exploratory, not an untouched final evaluation. Validation subsets are stratified because exported tables do not consistently retain event timestamps. No test metric chooses a candidate. BAF is Base only; variants retain original reports. No production checkpoint replacement.

[Full protocol and results](outputs/supervised_search_v1/report.json)
[Training instructions](training_model/FOUR_SUPERVISED.md)

The original README retains dataset source, licence and attribution. Scores support review; they do not establish fraud. Existing production artifacts were retained separately.
