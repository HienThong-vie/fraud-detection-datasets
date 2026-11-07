## Two method types: classification + clustering

The four supervised classifiers are unchanged. A separate exploratory MiniBatch K-means study groups similar records; clusters are not fraud predictions.

Selected k: **8**; training silhouette: **0.312**; mean seed ARI: **0.697**.

Passed the stated size/repeatability floors; this does not establish real-world or fraud relevance.

Preprocessing, k selection and repeatability checks use training inputs only. Labels, identifiers, label-derived features, absolute event-time columns and Elliptic neighbour aggregates are excluded. Fraud rates are interpreted afterward, never used to choose clusters. Validation/test records are assigned without refitting. Previously viewed test periods remain exploratory.

- [Interpretation and limitations](experiment_suite/evaluation/banksim/clustering_v1/README.md)
- [Silhouette and seed stability](experiment_suite/evaluation/banksim/clustering_v1/silhouette_stability.png)
- [Cluster sizes](experiment_suite/evaluation/banksim/clustering_v1/cluster_sizes.png)
- [Feature profiles](experiment_suite/evaluation/banksim/clustering_v1/feature_profiles.png)
- [Post-hoc fraud rates](experiment_suite/evaluation/banksim/clustering_v1/fraud_rates.png)
- [Metrics and saved clustering pipeline](experiment_suite/evaluation/banksim/clustering_v1/)

Run `python training_model/cluster.py` from the branch root after supplying the raw data as described below. This entrypoint preserves completed runs. The local dashboard shows a separate clustering section. The main training entrypoint still runs classification only.

Silhouette is not accuracy, and seed repeatability does not establish temporal stability. Weak clustering is a documented finding, not evidence of successful fraud detection.

---

# BANKSIM: latest supervised experiment

Latest: `priority_search_v2`. Four supervised families; selected champion: **catboost**. Training ran locally on the Mac CPU. Selection uses chronological validation, not test scores.

| Model | Test AP* | Recall* | Actual FPR* | Precision* | F1* | ROC-AUC* |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.7711 | 78.3% | 0.873% | 49.5% | 0.6065 | 0.9925 |
| random_forest | 0.8034 | 79.4% | 0.843% | 50.7% | 0.6190 | 0.9926 |
| extra_trees | 0.7093 | 77.6% | 0.857% | 49.7% | 0.6059 | 0.9909 |
| catboost | 0.8026 | 80.6% | 0.827% | 51.5% | 0.6286 | 0.9911 |

*Previously inspected test periods are exploratory, not a fresh holdout. The displayed operating point uses a 1% calibration FPR budget; future FPR can differ. Scores support review and do not establish fraud. No business deployment threshold has been approved.

- [Model comparison](experiment_suite/evaluation/banksim/priority_search_v2/model_comparison.png)
- [Precision–recall curves](experiment_suite/evaluation/banksim/priority_search_v2/precision_recall_curves.png)
- [All metrics](experiment_suite/evaluation/banksim/priority_search_v2/model_comparison.csv)
- [Protocol and results](experiment_suite/evaluation/banksim/priority_search_v2/report.json)
- [Saved models and all graphs](experiment_suite/evaluation/banksim/priority_search_v2/)

This branch retains its previously completed experiment; the latest feature-policy retraining targeted BAF, IEEE, PaySim and Elliptic only.

## Run in PyCharm

Open this branch as a project, create a Python 3.12 virtual environment, then use the terminal:

```bash
python -m pip install -r experiment_suite/requirements.txt
python experiment_suite/app.py
```

Open http://127.0.0.1:8050/. Stop another suite server first if the port is in use. Saved models work without raw data; inputs must contain the engineered columns shown by the dashboard.

## Reproduce training

Put downloaded source files in `experiment_suite/dataset/banksim/`. Do not commit raw data. Run from `experiment_suite/`:

```bash
python -c "import sys; sys.path.insert(0, 'training_model'); from priority_data import prepare; prepare('banksim')"
```

```bash
python training_model/priority_search.py --dataset banksim
```

Completed experiment folders are preserved. A completed run may report already complete or reuse cached fits. For a genuinely new search, use a separate working copy and a new versioned experiment directory; do not overwrite the published evidence. Mac/Linux is required for the training locks (`fcntl`); dashboard inference does not need those locks.

The full implementation lives in `experiment_suite/training_model/`; original project code and documentation remain below. Production baseline artifacts are retained separately.

---

## Earlier project documentation, sources and licences

# BANKSIM: four supervised-model comparison

The latest experiment compares **logistic regression, random forest, extra trees, and CatBoost**. Each family has two regularized settings; validation selects its settings and threshold. The four-model winner is frozen before test evaluation. The run was performed locally on the Mac CPU.

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

# BankSim Synthetic Payment Fraud ML

Standalone PyCharm project for the BankSim agent-based bank-payment dataset. It compares class-weighted
logistic regression, random forest, and CatBoost using a chronological train/validation/test split.
CatBoost keeps the dataset's categorical fields native instead of one-hot expanding them.

## Dataset

- [Canonical Kaggle source](https://www.kaggle.com/datasets/ealaxi/banksim1)
- Synthetic data generated by the BankSim simulator from aggregated Spanish-bank patterns
- Approximately 594,643 transactions with 7,200 fraud labels

The CSV is not redistributed by this repository. Customer IDs, merchant IDs and constant ZIP fields are excluded from the baseline to reduce memorisation and useless features.

## PyCharm setup and run

Open the whole folder, create a Python 3.11 virtual environment, then:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/download_data.py
python preprocess_data/run.py
python training_model/train.py
python evaluation/evaluate.py
python -m fraudbench.inference examples/sample_transaction.json
python -m uvicorn fraudbench.api:app --host 127.0.0.1 --port 8050
```

Expected raw file:

```text
data/raw/bs140513_032310.csv
```

## Folder guide

| Folder | Contents |
|---|---|
| `dataset` | Source, citation and expected filename |
| `preprocess_data` | Validation and temporal splitting |
| `preprocessed_df` | Generated model table |
| `training_model` | Training entrypoint |
| `checkpoint` | Selected `.joblib` model |
| `evaluation` | Evaluation entrypoint |
| `outputs` | Metrics, predictions and charts |
| `utils` | Team helper scripts |

## Evaluation design

- Simulation steps are split chronologically before sampling.
- Preprocessing is fitted within the training pipeline.
- Selection and threshold tuning use validation data only.
- The final period is untouched until held-out evaluation.
- PR-AUC is the selection metric because fraud is rare.

## Citation

E. A. López-Rojas and S. Axelsson, “BankSim: A Bank Payment Simulation for Fraud Detection Research,” 2014.

## Current trained result

Logistic regression was selected. On the latest 50,000-row simulation-period sample it achieved PR-AUC **0.7716**, ROC-AUC **0.9923**, precision **0.8757**, recall **0.5870**, and F1 **0.7029**. See [`outputs/metrics.json`](outputs/metrics.json) and [`outputs/model_comparison.csv`](outputs/model_comparison.csv).
CatBoost was evaluated as a third candidate, but its validation PR-AUC (**0.7753**) did not exceed logistic
regression (**0.7850**), so it was not selected.

![BankSim model comparison](outputs/model_comparison.png)

![BankSim confusion matrix](outputs/confusion_matrix.png)
