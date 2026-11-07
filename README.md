# ULB Credit-Card Fraud Detection

A Python project for fraud-data analysis, supervised classification and exploratory clustering. It includes trained pipelines, a local dashboard, evaluation charts and reproducible training code.

## Features

- Four supervised classifiers: logistic regression, random forest, extra trees, xgboost.
- Separate MiniBatch K-means segmentation with training-only selection and seed-stability checks.
- Time-ordered data splits, validation-calibrated alert thresholds and saved model pipelines.
- FastAPI dashboard for predictions, metrics and downloadable evaluation files.

## Quick start

```bash
git clone --branch ulb --single-branch https://github.com/KennyDao1324/fraud-detection-datasets.git
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
| `experiment_suite/pipelines/ulb/` | Dataset-specific implementation and configuration |
| `experiment_suite/evaluation/ulb/` | Saved pipelines, metrics, charts and model cards |
| `experiment_suite/dataset/ulb/` | Local raw input files; excluded from Git |
| `experiment_suite/app.py` | Local API and dashboard server |
| `training_model/train.py` | Classification launcher |
| `training_model/cluster.py` | Clustering launcher |
| `src/`, `scripts/`, `docs/` | Original dataset implementation, utilities and provenance |
| `artifacts/`, `checkpoint/`, `outputs/` | Earlier baseline models and evaluation evidence |

Small `__init__.py`, `.gitkeep` and launcher files support packaging or folder layout; the main implementation lives in the source and training modules.

## Classification results

Experiment: `fit_balance_v3`. Selected model: **xgboost**. The table uses the 5% calibration-FPR headline operating point; actual later-period FPR can differ. 1% is secondary.

| Model | PR-AUC | Recall | FPR | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.7579 | 90.9% | 3.607% | 3.2% | 0.0623 | 0.9842 |
| random_forest | 0.7747 | 89.4% | 4.123% | 2.8% | 0.0540 | 0.9795 |
| extra_trees | 0.7871 | 90.9% | 4.530% | 2.6% | 0.0503 | 0.9856 |
| xgboost | 0.7644 | 89.4% | 4.127% | 2.8% | 0.0540 | 0.9796 |

[Model comparison](experiment_suite/evaluation/ulb/fit_balance_v3/model_comparison.png) · [PR curves](experiment_suite/evaluation/ulb/fit_balance_v3/precision_recall_curves.png) · [Metrics CSV](experiment_suite/evaluation/ulb/fit_balance_v3/model_comparison.csv) · [Full results and protocol](experiment_suite/evaluation/ulb/fit_balance_v3/report.json)

Model settings and thresholds were selected using chronological validation. Test periods have already been inspected, so comparisons are exploratory rather than a fresh holdout. A score supports review and does not establish fraud or authorize an account decision.

## Clustering

MiniBatch K-means selected **k=2**. Training silhouette: **0.259**; mean seed ARI: **0.008**.

Weak/unstable geometry: no candidate satisfied both quality floors; do not claim meaningful fraud groups.

Preprocessing and k selection use training inputs only. Fraud labels and label-derived features are excluded from the inputs and used only afterward to interpret cluster fraud rates. Clusters are not fraud predictions; silhouette is not classification accuracy.

[Interpretation](experiment_suite/evaluation/ulb/clustering_v1/README.md) · [Silhouette and stability](experiment_suite/evaluation/ulb/clustering_v1/silhouette_stability.png) · [Cluster sizes](experiment_suite/evaluation/ulb/clustering_v1/cluster_sizes.png) · [Feature profiles](experiment_suite/evaluation/ulb/clustering_v1/feature_profiles.png) · [Fraud-rate interpretation](experiment_suite/evaluation/ulb/clustering_v1/fraud_rates.png)

## Training

Place the original downloaded files in `experiment_suite/dataset/ulb/`. Keep the raw files local and follow their source licences. From `experiment_suite/`, prepare the data:

```bash
python -c "import sys; sys.path.insert(0, 'training_model'); from priority_data import prepare; prepare('ulb')"
python training_model/fit_balance_search.py --dataset ulb
python training_model/clustering.py --dataset ulb
```

Completed experiment folders are preserved and may report already complete or reuse cached fits. Use a separate working copy and a new versioned output directory for a new search; do not overwrite the published evidence. Run classification and clustering separately.

## Dataset and licence

- [Original Kaggle distribution](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 real transactions, including 492 confirmed fraud cases
- `V1`–`V28` are anonymized PCA components; `Time` and `Amount` retain their published meaning

The anonymization prevents business-level interpretation of the PCA components. This project must not invent meanings for them.

Raw datasets and row-level prediction exports are not redistributed here. Code and dataset licences are separate; check the original source terms before reuse.

Raw datasets and row-level prediction exports are not redistributed here. Code and dataset licences are separate; check the original source terms before reuse.

## Citation

Dal Pozzolo, A. et al., “Calibrating Probability with Undersampling for Unbalanced Classification,” IEEE SSCI, 2015.

[Selected-model identity and operating points](FINAL_MODEL.md) · [Conda workflow](experiment_suite/CONDA_GUIDE.md)
