# Four-supervised-model experiment card

Dataset: baf. Selected family: catboost.

| Model | Setting | Test PR-AUC | Test ROC-AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| logistic_regression | C0.1 | 0.1555 | 0.8830 | 0.2421 | 0.1852 | 0.2098 |
| random_forest | d12_leaf8 | 0.1452 | 0.8863 | 0.2513 | 0.1353 | 0.1759 |
| extra_trees | d12_leaf8 | 0.1504 | 0.8720 | 0.2232 | 0.1895 | 0.2049 |
| catboost | prior_accepted_configuration | 0.1832 | 0.8823 | 0.2757 | 0.2293 | 0.2504 |

Maximize worst utility across two selection-validation subsets minus 0.03 * positive train/validation PR-AUC gap. Utility: 45% F1, 15% min(precision,recall), 20% PR-AUC, 20% ROC-AUC. Calibration threshold within 95% of best F1, favor balanced precision/recall, calibration FPR <= 5%.

Existing temporal test periods have already been inspected. Exploratory, not an untouched final evaluation. Validation subsets are stratified because exported tables do not consistently retain event timestamps. No test metric chooses a candidate. BAF is Base only; variants retain original reports. No production checkpoint replacement.

[Full protocol and results](outputs/supervised_search_v1/report.json)
[Training instructions](training_model/FOUR_SUPERVISED.md)

The original README retains dataset source, licence and attribution. Scores support review; they do not establish fraud. Existing production artifacts were retained separately.
