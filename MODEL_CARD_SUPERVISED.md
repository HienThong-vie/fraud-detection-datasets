# Four-supervised-model experiment card

Dataset: sparkov. Selected family: catboost.

| Model | Setting | Test PR-AUC | Test ROC-AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| logistic_regression | C0.1 | 0.1359 | 0.9222 | 0.2321 | 0.3368 | 0.2748 |
| random_forest | d12_leaf8 | 0.8203 | 0.9924 | 0.7075 | 0.7772 | 0.7407 |
| extra_trees | d12_leaf8 | 0.6207 | 0.9596 | 0.5959 | 0.5959 | 0.5959 |
| catboost | d5_SqrtBalanced | 0.8669 | 0.9943 | 0.7225 | 0.8497 | 0.7810 |

Maximize worst utility across two selection-validation subsets minus 0.03 * positive train/validation PR-AUC gap. Utility: 45% F1, 15% min(precision,recall), 20% PR-AUC, 20% ROC-AUC. Calibration threshold within 95% of best F1, favor balanced precision/recall, calibration FPR <= 5%.

Existing temporal test periods have already been inspected. Exploratory, not an untouched final evaluation. Validation subsets are stratified because exported tables do not consistently retain event timestamps. No test metric chooses a candidate. BAF is Base only; variants retain original reports. No production checkpoint replacement.

[Full protocol and results](outputs/supervised_search_v1/report.json)
[Training instructions](training_model/FOUR_SUPERVISED.md)

The original README retains dataset source, licence and attribution. Scores support review; they do not establish fraud. Existing production artifacts were retained separately.
