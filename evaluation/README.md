# Latest: four supervised classifiers

[Open all latest metrics and charts](../outputs/supervised_search_v1/)

[Model comparison](../outputs/supervised_search_v1/model_comparison.png) · [PR curves](../outputs/supervised_search_v1/precision_recall_curves.png) · [ROC curves](../outputs/supervised_search_v1/roc_curves.png) · [Metrics CSV](../outputs/supervised_search_v1/model_comparison.csv)

---

# Evaluation

After training, open the [outputs folder](../outputs/) to find:

- `model_comparison.csv`
- `metrics.json`
- `dataset_profile.json`
- `test_predictions.csv`
- `model_comparison.png`
- `precision_recall_curves.png`
- `confusion_matrix.png`
- `prediction_distribution.png`

Run `python evaluation/evaluate.py` to print the held-out results in the terminal.

## Current real-data run

Random forest was selected by validation PR-AUC. On the 50,000-row held-out test sample it achieved
PR-AUC 0.2511, ROC-AUC 0.6316, precision 0.9697, and recall 0.2143.

### Confusion matrix

![Handbook confusion matrix](../outputs/confusion_matrix.png)

### Model comparison

![Handbook model comparison](../outputs/model_comparison.png)

### Precision-recall curves

![Handbook precision-recall curves](../outputs/precision_recall_curves.png)

### Prediction distribution

![Handbook prediction distribution](../outputs/prediction_distribution.png)
