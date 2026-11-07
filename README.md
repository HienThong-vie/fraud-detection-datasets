## Two method types: classification + clustering

The four supervised classifiers are unchanged. A separate exploratory MiniBatch K-means study groups similar records; clusters are not fraud predictions.

Selected k: **2**; training silhouette: **0.259**; mean seed ARI: **0.008**.

Weak/unstable geometry: no candidate satisfied both quality floors; do not claim meaningful fraud groups.

Preprocessing, k selection and repeatability checks use training inputs only. Labels, identifiers, label-derived features, absolute event-time columns and Elliptic neighbour aggregates are excluded. Fraud rates are interpreted afterward, never used to choose clusters. Validation/test records are assigned without refitting. Previously viewed test periods remain exploratory.

- [Interpretation and limitations](experiment_suite/evaluation/ulb/clustering_v1/README.md)
- [Silhouette and seed stability](experiment_suite/evaluation/ulb/clustering_v1/silhouette_stability.png)
- [Cluster sizes](experiment_suite/evaluation/ulb/clustering_v1/cluster_sizes.png)
- [Feature profiles](experiment_suite/evaluation/ulb/clustering_v1/feature_profiles.png)
- [Post-hoc fraud rates](experiment_suite/evaluation/ulb/clustering_v1/fraud_rates.png)
- [Metrics and saved clustering pipeline](experiment_suite/evaluation/ulb/clustering_v1/)

Run `python training_model/cluster.py` from the branch root after supplying the raw data as described below. This entrypoint preserves completed runs. The local dashboard shows a separate clustering section. The main training entrypoint still runs classification only.

Silhouette is not accuracy, and seed repeatability does not establish temporal stability. Weak clustering is a documented finding, not evidence of successful fraud detection.

---

# ULB: latest supervised experiment

Latest: `fit_balance_v3`. Four supervised families; selected champion: **xgboost**. Training ran locally on the Mac CPU. Selection uses chronological validation, not test scores.

| Model | Test AP* | Recall* | Actual FPR* | Precision* | F1* | ROC-AUC* |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.7579 | 86.4% | 0.713% | 13.8% | 0.2380 | 0.9842 |
| random_forest | 0.7747 | 86.4% | 0.861% | 11.7% | 0.2061 | 0.9795 |
| extra_trees | 0.7871 | 87.9% | 0.737% | 13.6% | 0.2358 | 0.9856 |
| xgboost | 0.7644 | 84.8% | 0.845% | 11.7% | 0.2059 | 0.9796 |

*Previously inspected test periods are exploratory, not a fresh holdout. The displayed operating point uses a 1% calibration FPR budget; future FPR can differ. Scores support review and do not establish fraud. No business deployment threshold has been approved.

- [Model comparison](experiment_suite/evaluation/ulb/fit_balance_v3/model_comparison.png)
- [Precision–recall curves](experiment_suite/evaluation/ulb/fit_balance_v3/precision_recall_curves.png)
- [All metrics](experiment_suite/evaluation/ulb/fit_balance_v3/model_comparison.csv)
- [Protocol and results](experiment_suite/evaluation/ulb/fit_balance_v3/report.json)
- [Saved models and all graphs](experiment_suite/evaluation/ulb/fit_balance_v3/)

This branch retains its previously completed experiment; the latest feature-policy retraining targeted BAF, IEEE, PaySim and Elliptic only.

## Run in PyCharm

Open this branch as a project, create a Python 3.12 virtual environment, then use the terminal:

```bash
python -m pip install -r experiment_suite/requirements.txt
python experiment_suite/app.py
```

Open http://127.0.0.1:8050/. Stop another suite server first if the port is in use. Saved models work without raw data; inputs must contain the engineered columns shown by the dashboard.

## Reproduce training

Put downloaded source files in `experiment_suite/dataset/ulb/`. Do not commit raw data. Run from `experiment_suite/`:

```bash
python -c "import sys; sys.path.insert(0, 'training_model'); from priority_data import prepare; prepare('ulb')"
```

```bash
python training_model/fit_balance_search.py --dataset ulb
```

Completed experiment folders are preserved. A completed run may report already complete or reuse cached fits. For a genuinely new search, use a separate working copy and a new versioned experiment directory; do not overwrite the published evidence. Mac/Linux is required for the training locks (`fcntl`); dashboard inference does not need those locks.

The full implementation lives in `experiment_suite/training_model/`; original project code and documentation remain below. Production baseline artifacts are retained separately.

---

## Earlier project documentation, sources and licences

# ULB: four supervised-model comparison

The latest experiment compares **logistic regression, random forest, extra trees, and XGBoost**. Each family has two regularized settings; validation selects its settings and threshold. The four-model winner is frozen before test evaluation. The run was performed locally on the Mac CPU.

Run `python training_model/train.py` from this branch's project root after preparing its raw data. This entrypoint now runs only the four supervised families.

- [Four-model comparison chart](outputs/supervised_search_v1/model_comparison.png)
- [Precision–recall curves](outputs/supervised_search_v1/precision_recall_curves.png)
- [ROC curves](outputs/supervised_search_v1/roc_curves.png)
- [Confusion matrices](outputs/supervised_search_v1/confusion_matrices.png)
- [Training/validation/test gaps](outputs/supervised_search_v1/generalization_gap.png)
- [Threshold trade-offs](outputs/supervised_search_v1/threshold_tradeoffs.png)
- [All metric values](outputs/supervised_search_v1/model_comparison.csv)
- [Protocol, exact results and limitations](outputs/supervised_search_v1/report.json)
- [Training method and commands](training_model/FOUR_SUPERVISED.md)

Four fitted pipelines and `champion.joblib` are in `outputs/supervised_search_v1/`. Existing production checkpoints are retained for comparison. All figures in this section are from the new four-model experiment; the original implementation notes below describe the earlier baseline. Test periods were inspected in earlier experiments; these are exploratory comparisons.

---

## Original project documentation and provenance

# ULB Real Credit-Card Fraud ML

Standalone PyCharm project for the real, anonymized ULB/Worldline European card-transaction dataset. It
compares class-weighted logistic regression, random forest, and XGBoost under a time-ordered evaluation.

## Dataset

- [Original Kaggle distribution](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 real transactions, including 492 confirmed fraud cases
- `V1`–`V28` are anonymized PCA components; `Time` and `Amount` retain their published meaning

The anonymization prevents business-level interpretation of the PCA components. This project must not invent meanings for them.

## PyCharm setup and run

Open the whole folder and create a Python 3.11 virtual environment:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/download_data.py
python preprocess_data/run.py
python training_model/train.py
python evaluation/evaluate.py
python -m fraudbench.inference examples/sample_transaction.json
python -m uvicorn fraudbench.api:app --host 127.0.0.1 --port 8060
```

The download script uses TensorFlow's public mirror and saves `data/raw/creditcard.csv`. Raw data remains ignored by Git.

## Folder guide

| Folder | Contents |
|---|---|
| `dataset` | Source, limitations and expected filename |
| `preprocess_data` | Time-ordered split entrypoint |
| `preprocessed_df` | Generated model table |
| `training_model` | Training entrypoint |
| `checkpoint` | Selected `.joblib` model |
| `evaluation` | Evaluation entrypoint |
| `outputs` | Metrics, predictions and charts |
| `utils` | Team helper scripts |

## Evaluation design

- Rows are ordered by `Time`, then divided 60%/20%/20%.
- Sampling occurs after splitting and preserves the target ratio.
- Scalers and imputers are fitted only inside the model pipeline.
- Validation PR-AUC selects the model; test data is evaluated once.
- Accuracy is reported but is not the main metric because only 0.172% of rows are fraud.

## Citation

Dal Pozzolo, A. et al., “Calibrating Probability with Undersampling for Unbalanced Classification,” IEEE SSCI, 2015.

## Current trained result

XGBoost was selected with validation PR-AUC **0.7660**. On the later 50,000-row real-transaction sample it
achieved PR-AUC **0.8155**, ROC-AUC **0.9725**, precision **0.9038**, recall **0.7121**, and F1 **0.7966**.
See [`outputs/metrics.json`](outputs/metrics.json) and
[`outputs/model_comparison.csv`](outputs/model_comparison.csv).

![ULB model comparison](outputs/model_comparison.png)

![ULB confusion matrix](outputs/confusion_matrix.png)
