# Four-supervised-model experiment card

Dataset: paysim. Selected family: xgboost.

| Model | Setting | Test PR-AUC | Test ROC-AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| logistic_regression | C1 | 0.1855 | 0.9158 | 0.3787 | 0.1704 | 0.2351 |
| random_forest | prior_accepted_configuration | 0.2997 | 0.8502 | 0.7352 | 0.2407 | 0.3626 |
| extra_trees | d12_leaf8 | 0.1884 | 0.8860 | 0.4735 | 0.1868 | 0.2680 |
| xgboost | d3_weightpower1 | 0.3685 | 0.9060 | 0.4707 | 0.3842 | 0.4230 |

Maximize worst utility across two selection-validation subsets minus 0.03 * positive train/validation PR-AUC gap. Utility: 45% F1, 15% min(precision,recall), 20% PR-AUC, 20% ROC-AUC. Calibration threshold within 95% of best F1, favor balanced precision/recall, calibration FPR <= 5%.

Existing temporal test periods have already been inspected. Exploratory, not an untouched final evaluation. Validation subsets are stratified because exported tables do not consistently retain event timestamps. No test metric chooses a candidate. BAF is Base only; variants retain original reports. No production checkpoint replacement.

[Full protocol and results](outputs/supervised_search_v1/report.json)
[Training instructions](training_model/FOUR_SUPERVISED.md)

The original README retains dataset source, licence and attribution. Scores support review; they do not establish fraud. Existing production artifacts were retained separately.
