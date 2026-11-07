# IEEE-CIS Fraud Detection

A Python project for fraud-data analysis, supervised classification and exploratory clustering. It includes trained pipelines, a local dashboard, evaluation charts and reproducible training code.

## Features

- Four supervised classifiers: logistic regression, random forest, extra trees, xgboost.
- Separate MiniBatch K-means segmentation with training-only selection and seed-stability checks.
- Time-ordered data splits, validation-calibrated alert thresholds and saved model pipelines.
- FastAPI dashboard for predictions, metrics and downloadable evaluation files.

## Quick start

```bash
git clone --branch ieee --single-branch https://github.com/KennyDao1324/fraud-detection-datasets.git
cd fraud-detection-datasets/experiment_suite
conda env create -f environment.yml
conda activate fraudsuite
python app.py
```

Open http://127.0.0.1:8050/. Saved models work without downloading the raw data. Use the example engineered record shown in the dashboard; raw transaction rows may require preprocessing.

In PyCharm, open the project folder and select the `fraudsuite` conda environment as interpreter. Training uses Mac/Linux file locks; inference does not require them.

## Project structure

| Path | Contents |
|---|---|
| `experiment_suite/training_model/` | Current preprocessing, classification and clustering code |
| `experiment_suite/pipelines/ieee/` | Dataset-specific implementation and configuration |
| `experiment_suite/evaluation/ieee/` | Saved pipelines, metrics, charts and model cards |
| `experiment_suite/dataset/ieee/` | Local raw input files; excluded from Git |
| `experiment_suite/app.py` | Local API and dashboard server |
| `training_model/train.py` | Classification launcher |
| `training_model/cluster.py` | Clustering launcher |
| `src/`, `scripts/`, `docs/` | Original dataset implementation, utilities and provenance |
| `artifacts/`, `checkpoint/`, `outputs/` | Earlier baseline models and evaluation evidence |

Small `__init__.py`, `.gitkeep` and launcher files support packaging or folder layout; the main implementation lives in the source and training modules.

## Classification results

Experiment: `feature_policy_v4`. Selected model: **xgboost**. The table uses the 5% calibration-FPR headline operating point; actual later-period FPR can differ. 1% is secondary.

| Model | PR-AUC | Recall | FPR | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.1818 | 49.6% | 6.582% | 21.4% | 0.2985 | 0.8218 |
| random_forest | 0.4635 | 57.6% | 5.861% | 26.2% | 0.3599 | 0.8679 |
| extra_trees | 0.3852 | 49.5% | 5.737% | 23.7% | 0.3208 | 0.8417 |
| xgboost | 0.5235 | 65.4% | 5.828% | 28.8% | 0.4002 | 0.9028 |

[Model comparison](experiment_suite/evaluation/ieee/feature_policy_v4/model_comparison.png) · [PR curves](experiment_suite/evaluation/ieee/feature_policy_v4/precision_recall_curves.png) · [Metrics CSV](experiment_suite/evaluation/ieee/feature_policy_v4/model_comparison.csv) · [Full results and protocol](experiment_suite/evaluation/ieee/feature_policy_v4/report.json)

Model settings and thresholds were selected using chronological validation. Test periods have already been inspected, so comparisons are exploratory rather than a fresh holdout. A score supports review and does not establish fraud or authorize an account decision.

[Model card](experiment_suite/evaluation/ieee/feature_policy_v4/MODEL_CARD.md) · [Threshold-policy comparison](experiment_suite/evaluation/ieee/feature_policy_v4/threshold_policies.csv)

The dashboard supports five calibration FPR budgets, optional review-capacity limits and an approximate one-sided 95% calibration FPR bound. These are calibration diagnostics, not guarantees of future FPR or alert volume.

## Clustering

MiniBatch K-means selected **k=3**. Training silhouette: **0.467**; mean seed ARI: **0.868**.

Passed the stated size/repeatability floors; this does not establish real-world or fraud relevance.

Preprocessing and k selection use training inputs only. Fraud labels and label-derived features are excluded from the inputs and used only afterward to interpret cluster fraud rates. Clusters are not fraud predictions; silhouette is not classification accuracy.

[Interpretation](experiment_suite/evaluation/ieee/clustering_v1/README.md) · [Silhouette and stability](experiment_suite/evaluation/ieee/clustering_v1/silhouette_stability.png) · [Cluster sizes](experiment_suite/evaluation/ieee/clustering_v1/cluster_sizes.png) · [Feature profiles](experiment_suite/evaluation/ieee/clustering_v1/feature_profiles.png) · [Fraud-rate interpretation](experiment_suite/evaluation/ieee/clustering_v1/fraud_rates.png)

## Training

Place the original downloaded files in `experiment_suite/dataset/ieee/`. Keep the raw files local and follow their source licences. From `experiment_suite/`, prepare the data:

```bash
python -c "import sys; sys.path.insert(0, 'training_model'); from priority_data import prepare; prepare('ieee')"
python training_model/feature_policy_search.py --dataset ieee
python training_model/clustering.py --dataset ieee
```

Completed experiment folders are preserved and may report already complete or reuse cached fits. Use a separate working copy and a new versioned output directory for a new search; do not overwrite the published evidence. Run classification and clustering separately.

## Dataset and licence

- [IEEE-CIS Fraud Detection competition and dataset](https://www.kaggle.com/competitions/ieee-fraud-detection)

Kaggle requires an account and acceptance of the competition rules before download. Those rules apply to
the data independently of this project's code licence, so the raw CSV files are not redistributed here.

Raw datasets and row-level prediction exports are not redistributed here. Code and dataset licences are separate; check the original source terms before reuse.

Raw datasets and row-level prediction exports are not redistributed here. Code and dataset licences are separate; check the original source terms before reuse.

[Selected-model identity and operating points](FINAL_MODEL.md) · [Conda workflow](experiment_suite/CONDA_GUIDE.md)
