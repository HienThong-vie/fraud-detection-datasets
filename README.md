## Two method types: classification + clustering

The four supervised classifiers are unchanged. A separate exploratory MiniBatch K-means study groups similar records; clusters are not fraud predictions.

Selected k: **2**; training silhouette: **0.379**; mean seed ARI: **0.906**.

Passed the stated size/repeatability floors; this does not establish real-world or fraud relevance.

Preprocessing, k selection and repeatability checks use training inputs only. Labels, identifiers, label-derived features, absolute event-time columns and Elliptic neighbour aggregates are excluded. Fraud rates are interpreted afterward, never used to choose clusters. Validation/test records are assigned without refitting. Previously viewed test periods remain exploratory.

- [Interpretation and limitations](experiment_suite/evaluation/elliptic/clustering_v1/README.md)
- [Silhouette and seed stability](experiment_suite/evaluation/elliptic/clustering_v1/silhouette_stability.png)
- [Cluster sizes](experiment_suite/evaluation/elliptic/clustering_v1/cluster_sizes.png)
- [Feature profiles](experiment_suite/evaluation/elliptic/clustering_v1/feature_profiles.png)
- [Post-hoc fraud rates](experiment_suite/evaluation/elliptic/clustering_v1/fraud_rates.png)
- [Metrics and saved clustering pipeline](experiment_suite/evaluation/elliptic/clustering_v1/)

Run `python training_model/cluster.py` from the branch root after supplying the raw data as described below. This entrypoint preserves completed runs. The local dashboard shows a separate clustering section. The main training entrypoint still runs classification only.

Silhouette is not accuracy, and seed repeatability does not establish temporal stability. Weak clustering is a documented finding, not evidence of successful fraud detection.

---

# ELLIPTIC: latest supervised experiment

Latest: `feature_policy_v4`. Four supervised families; selected champion: **xgboost**. Training ran locally on the Mac CPU. Selection uses chronological validation, not test scores.

| Model | Test AP* | Recall* | Actual FPR* | Precision* | F1* | ROC-AUC* |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.1579 | 35.8% | 5.775% | 23.1% | 0.2805 | 0.8207 |
| random_forest | 0.5449 | 47.8% | 1.767% | 56.7% | 0.5186 | 0.8555 |
| extra_trees | 0.5366 | 48.0% | 2.146% | 52.0% | 0.4994 | 0.8287 |
| xgboost | 0.5333 | 48.0% | 1.210% | 65.8% | 0.5552 | 0.8032 |

*Previously inspected test periods are exploratory, not a fresh holdout. The displayed operating point uses a 1% calibration FPR budget; future FPR can differ. Scores support review and do not establish fraud. No business deployment threshold has been approved.

- [Model comparison](experiment_suite/evaluation/elliptic/feature_policy_v4/model_comparison.png)
- [Precision–recall curves](experiment_suite/evaluation/elliptic/feature_policy_v4/precision_recall_curves.png)
- [All metrics](experiment_suite/evaluation/elliptic/feature_policy_v4/model_comparison.csv)
- [Protocol and results](experiment_suite/evaluation/elliptic/feature_policy_v4/report.json)
- [Saved models and all graphs](experiment_suite/evaluation/elliptic/feature_policy_v4/)

- [Threshold policies](experiment_suite/evaluation/elliptic/feature_policy_v4/threshold_policies.csv)
- [Feature comparison](experiment_suite/evaluation/elliptic/feature_policy_v4/candidate_feature_comparison.png)
- [Temporal checks](experiment_suite/evaluation/elliptic/feature_policy_v4/temporal_validation.png)
- [Model card and limitations](experiment_suite/evaluation/elliptic/feature_policy_v4/MODEL_CARD.md)

The dashboard offers 0.5%, 1%, 2%, 3% and 5% calibration FPR budgets, an approximate one-sided 95% calibration FPR bound, and review-capacity limits. These constraints apply to calibration data and do not guarantee future compliance.

**Elliptic warning:** the selected incumbent uses supplied forward-neighbour aggregates and remains an offline benchmark, not a validated at-transaction-time model. Incoming-only candidates were tested but did not win the declared validation criterion.

## Run in PyCharm

Open this branch as a project, create a Python 3.12 virtual environment, then use the terminal:

```bash
python -m pip install -r experiment_suite/requirements.txt
python experiment_suite/app.py
```

Open http://127.0.0.1:8050/. Stop another suite server first if the port is in use. Saved models work without raw data; inputs must contain the engineered columns shown by the dashboard.

## Reproduce training

Put downloaded source files in `experiment_suite/dataset/elliptic/`. Do not commit raw data. Run from `experiment_suite/`:

```bash
python -c "import sys; sys.path.insert(0, 'training_model'); from priority_data import prepare; prepare('elliptic')"
```

```bash
python training_model/feature_policy_search.py --dataset elliptic
```

Completed experiment folders are preserved. A completed run may report already complete or reuse cached fits. For a genuinely new search, use a separate working copy and a new versioned experiment directory; do not overwrite the published evidence. Mac/Linux is required for the training locks (`fcntl`); dashboard inference does not need those locks.

The full implementation lives in `experiment_suite/training_model/`; original project code and documentation remain below. Production baseline artifacts are retained separately.

---

## Earlier project documentation, sources and licences

# ELLIPTIC: four supervised-model comparison

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

# Elliptic Real Bitcoin AML ML

Standalone PyCharm project for the real, anonymized Elliptic Bitcoin transaction graph. The experiment
compares class-weighted logistic regression, random forest, and XGBoost using published node features and
a strict blockchain-time split.

## Dataset

- [Canonical Elliptic distribution](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
- Real Bitcoin transaction graph labelled as licit, illicit or unknown
- 203,769 transaction nodes over 49 time steps
- Canonical licence: CC BY-NC-ND 4.0; verify terms before redistribution

This baseline trains only on labelled nodes and uses the 165 published node features. It does not use the edge list, so it is a tabular baseline rather than a graph neural network. Unknown labels are excluded rather than treated as legitimate.

## PyCharm setup and run

Open the whole folder and create a Python 3.11 virtual environment:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/download_data.py
python preprocess_data/run.py
python training_model/train.py
python evaluation/evaluate.py
python scripts/export_example.py
python -m fraudbench.inference examples/sample_transaction.json
python -m uvicorn fraudbench.api:app --host 127.0.0.1 --port 8070
```

Raw data and generated processed rows remain ignored by Git.

## Folder guide

| Folder | Contents |
|---|---|
| `dataset` | Source, licence and limitations |
| `preprocess_data` | Label filtering and temporal split |
| `preprocessed_df` | Generated labelled model table |
| `training_model` | Baseline training entrypoint |
| `checkpoint` | Selected `.joblib` model |
| `evaluation` | Evaluation entrypoint |
| `outputs` | Metrics, predictions and charts |
| `utils` | Team helper scripts |

## Evaluation design

- Time steps 1–34 train the models.
- Steps 35–41 select the model and threshold.
- Steps 42–49 are held out for final evaluation.
- `time_step` and transaction IDs are not model inputs.
- PR-AUC is the selection metric due to severe class imbalance.

## Citation

Weber, M. et al., “Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics,” 2019.

## Current trained result

XGBoost was selected with validation PR-AUC **0.9321**. On labelled nodes from later blockchain time steps
42–49 it achieved PR-AUC **0.5533**, ROC-AUC **0.8463**, precision **0.8818**, recall **0.4755**, and F1
**0.6178**. The large reduction from validation performance is reported as temporal distribution shift,
not hidden by random splitting.

![Elliptic model comparison](outputs/model_comparison.png)

![Elliptic confusion matrix](outputs/confusion_matrix.png)
