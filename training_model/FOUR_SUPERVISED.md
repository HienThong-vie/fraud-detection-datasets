# Four supervised classifiers per dataset

Run on the local Mac CPU from the project root:

```bash
python training_model/supervised_search.py --dataset baf
```

In a standalone dataset branch, use that branch's dataset ID in the same command. The script detects
the branch's project root. It reads `preprocessed_df/` and uses the existing artifact's feature list.
Raw records must first be processed by the original dataset-specific preparation pipeline.

Exactly four families are compared: logistic regression, random forest, extra trees, and gradient
boosted trees. The boosted learner is CatBoost for BAF, Sparkov and BankSim; XGBoost for IEEE, PaySim,
Handbook, ULB and Elliptic. All four learn from the binary target labels. Clustering is not run.

Each family has two prespecified configurations. Logistic regression tests C=0.1 and C=1. Random
forest and extra trees use 140 trees with depth 8/minimum leaf 16 or depth 12/minimum leaf 8. Boosted
trees use 350 iterations and regularized depth-3 or depth-5 configurations. Class weighting is computed
from training data. Full exact trial evidence is saved in `trials.json`.

After that search, `--include-prior` screens the previously accepted configuration for its existing
family using the same training, calibration and selection rows. That adds one fit per dataset (nine
fits per dataset, 72 across all eight) and can retain existing settings. It does not introduce a fifth
model family. Use this screening flag once after the initial search.

The temporal train/validation/test partitions are preserved. Validation is divided into 40% threshold
calibration and 60% model selection; the selection set is divided into two stratified subsets. This
subdivision is explicitly stratified, not claimed chronological, because prepared exports do not
consistently preserve timestamps. The calibration threshold stays within 95% of the best calibration
F1, prefers balanced precision and recall, and requires calibration false-positive rate <=5%.

For each candidate, utility is 45% F1 + 15% min(precision,recall) + 20% PR-AUC + 20% ROC-AUC.
Selection uses the worse utility across the two selection subsets, minus 0.03 times the positive
train/selection PR-AUC gap. Accuracy is reported but does not select a model. This weighting is a stated
engineering preference, not a claim of a universally optimal business objective.

One configuration is selected per family, and the four-family champion is frozen in `selection.json`
before test scores are computed. The exact evaluated model and threshold are saved without a
train-plus-validation refit. Test results are never used to pick candidates or revise thresholds.
These temporal test periods have already been inspected in earlier experiments, so the new results
remain exploratory. Confirmation requires fresh later observations. A low loss or high training AUC
does not establish good generalization. No minimum precision/recall is claimed when the data cannot
support it. BAF tuning in this round covers Base; original variant analyses remain available separately.

Outputs under `outputs/supervised_search_v1/`:

- `report.json`, `protocol.json`, `selection.json`, `trials.json`
- `model_comparison.csv`: train, validation, test metrics for all four models
- `test_predictions.csv`: labels and four sets of held-out scores
- Four named `.joblib` pipelines and `champion.joblib`
- `model_comparison.png`: PR-AUC, ROC-AUC, precision, recall, F1 and FPR
- `precision_recall_curves.png`, `roc_curves.png`
- `confusion_matrices.png`, `generalization_gap.png`, `threshold_tradeoffs.png`

Existing production artifacts are preserved until the team reviews the trade-offs. The smoke-test
directory uses tiny data and is software verification only, never report evidence.
