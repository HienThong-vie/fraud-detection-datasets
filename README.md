# Elliptic Bitcoin AML

[Selected model and operating points](FINAL_MODEL.md) · [Conda workflow](experiment_suite/CONDA_GUIDE.md)

A Python project for fraud-data analysis, supervised classification and exploratory clustering. It includes trained pipelines, a local dashboard, evaluation charts and reproducible training code.

## Features

- Four supervised classifiers: logistic regression, random forest, extra trees, xgboost.
- Separate MiniBatch K-means segmentation with training-only selection and seed-stability checks.
- Time-ordered data splits, validation-calibrated alert thresholds and saved model pipelines.
- FastAPI dashboard for predictions, metrics and downloadable evaluation files.

## Quick start

```bash
git clone --branch elliptic --single-branch https://github.com/KennyDao1324/fraud-detection-datasets.git
cd fraud-detection-datasets
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r experiment_suite/requirements.txt
python experiment_suite/app.py
```

Open http://127.0.0.1:8050/. Saved models work without downloading the raw data. Use the example engineered record shown in the dashboard; raw transaction rows may require preprocessing.

In PyCharm, open the project folder and select `.venv` as the Python interpreter. On Windows, activate the environment with `.venv\Scripts\Activate.ps1`. Training uses Mac/Linux file locks; inference does not require them.

## Project structure

| Path | Contents |
|---|---|
| `experiment_suite/training_model/` | Current preprocessing, classification and clustering code |
| `experiment_suite/pipelines/elliptic/` | Dataset-specific implementation and configuration |
| `experiment_suite/evaluation/elliptic/` | Saved pipelines, metrics, charts and model cards |
| `experiment_suite/dataset/elliptic/` | Local raw input files; excluded from Git |
| `experiment_suite/app.py` | Local API and dashboard server |
| `training_model/train.py` | Classification launcher |
| `training_model/cluster.py` | Clustering launcher |
| `src/`, `scripts/`, `docs/` | Original dataset implementation, utilities and provenance |
| `artifacts/`, `checkpoint/`, `outputs/` | Earlier baseline models and evaluation evidence |

Small `__init__.py`, `.gitkeep` and launcher files support packaging or folder layout; the main implementation lives in the source and training modules.

## Classification results

Experiment: `feature_policy_v4`. Selected model: **xgboost**. The table uses the 1% calibration-FPR operating point; actual later-period FPR can differ.

| Model | PR-AUC | Recall | FPR | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.1579 | 35.8% | 5.775% | 23.1% | 0.2805 | 0.8207 |
| random_forest | 0.5449 | 47.8% | 1.767% | 56.7% | 0.5186 | 0.8555 |
| extra_trees | 0.5366 | 48.0% | 2.146% | 52.0% | 0.4994 | 0.8287 |
| xgboost | 0.5333 | 48.0% | 1.210% | 65.8% | 0.5552 | 0.8032 |

[Model comparison](experiment_suite/evaluation/elliptic/feature_policy_v4/model_comparison.png) · [PR curves](experiment_suite/evaluation/elliptic/feature_policy_v4/precision_recall_curves.png) · [Metrics CSV](experiment_suite/evaluation/elliptic/feature_policy_v4/model_comparison.csv) · [Full results and protocol](experiment_suite/evaluation/elliptic/feature_policy_v4/report.json)

Model settings and thresholds were selected using chronological validation. Test periods have already been inspected, so comparisons are exploratory rather than a fresh holdout. A score supports review and does not establish fraud or authorize an account decision.

[Model card](experiment_suite/evaluation/elliptic/feature_policy_v4/MODEL_CARD.md) · [Threshold-policy comparison](experiment_suite/evaluation/elliptic/feature_policy_v4/threshold_policies.csv)

The dashboard supports five calibration FPR budgets, optional review-capacity limits and an approximate one-sided 95% calibration FPR bound. These are calibration diagnostics, not guarantees of future FPR or alert volume.

The selected Elliptic classifier remains an offline benchmark using supplied neighbour aggregates. It is not validated for use at transaction creation. Incoming-only candidates are reported separately. This dataset concerns Bitcoin AML, not bank-account applications.

## Clustering

MiniBatch K-means selected **k=2**. Training silhouette: **0.379**; mean seed ARI: **0.906**.

Passed the stated size/repeatability floors; this does not establish real-world or fraud relevance.

Preprocessing and k selection use training inputs only. Fraud labels and label-derived features are excluded from the inputs and used only afterward to interpret cluster fraud rates. Clusters are not fraud predictions; silhouette is not classification accuracy.

[Interpretation](experiment_suite/evaluation/elliptic/clustering_v1/README.md) · [Silhouette and stability](experiment_suite/evaluation/elliptic/clustering_v1/silhouette_stability.png) · [Cluster sizes](experiment_suite/evaluation/elliptic/clustering_v1/cluster_sizes.png) · [Feature profiles](experiment_suite/evaluation/elliptic/clustering_v1/feature_profiles.png) · [Fraud-rate interpretation](experiment_suite/evaluation/elliptic/clustering_v1/fraud_rates.png)

## Training

Place the original downloaded files in `experiment_suite/dataset/elliptic/`. Keep the raw files local and follow their source licences. From `experiment_suite/`, prepare the data:

```bash
python -c "import sys; sys.path.insert(0, 'training_model'); from priority_data import prepare; prepare('elliptic')"
python training_model/feature_policy_search.py --dataset elliptic
python training_model/clustering.py --dataset elliptic
```

Completed experiment folders are preserved and may report already complete or reuse cached fits. Use a separate working copy and a new versioned output directory for a new search; do not overwrite the published evidence. Run classification and clustering separately.

## Dataset and licence

- [Canonical Elliptic distribution](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
- Real Bitcoin transaction graph labelled as licit, illicit or unknown
- 203,769 transaction nodes over 49 time steps
- Canonical licence: CC BY-NC-ND 4.0; verify terms before redistribution

This baseline trains only on labelled nodes and uses the 165 published node features. It does not use the edge list, so it is a tabular baseline rather than a graph neural network. Unknown labels are excluded rather than treated as legitimate.

Raw datasets and row-level prediction exports are not redistributed here. Code and dataset licences are separate; check the original source terms before reuse.

## Citation

Weber, M. et al., “Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics,” 2019.
