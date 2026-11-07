# Four-supervised-model experiment card

Dataset: ulb. Selected family: xgboost.

| Model | Setting | Test PR-AUC | Test ROC-AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| logistic_regression | C1 | 0.7432 | 0.9830 | 0.9348 | 0.6515 | 0.7679 |
| random_forest | d12_leaf8 | 0.8134 | 0.9742 | 0.8475 | 0.7576 | 0.8000 |
| extra_trees | d8_leaf16 | 0.7644 | 0.9839 | 0.7619 | 0.7273 | 0.7442 |
| xgboost | d3_weightpower1 | 0.7983 | 0.9775 | 0.8197 | 0.7576 | 0.7874 |

Maximize worst utility across two selection-validation subsets minus 0.03 * positive train/validation PR-AUC gap. Utility: 45% F1, 15% min(precision,recall), 20% PR-AUC, 20% ROC-AUC. Calibration threshold within 95% of best F1, favor balanced precision/recall, calibration FPR <= 5%.

Existing temporal test periods have already been inspected. Exploratory, not an untouched final evaluation. Validation subsets are stratified because exported tables do not consistently retain event timestamps. No test metric chooses a candidate. BAF is Base only; variants retain original reports. No production checkpoint replacement.

[Full protocol and results](outputs/supervised_search_v1/report.json)
[Training instructions](training_model/FOUR_SUPERVISED.md)

The original README retains dataset source, licence and attribution. Scores support review; they do not establish fraud. Existing production artifacts were retained separately.
