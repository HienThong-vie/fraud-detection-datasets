# PaySim Fraud Detection

[Selected model and operating points](FINAL_MODEL.md) · [Conda workflow](experiment_suite/CONDA_GUIDE.md)

A Python project for fraud-data analysis, supervised classification and exploratory clustering. It includes trained pipelines, a local dashboard, evaluation charts and reproducible training code.

## Features

- Four supervised classifiers: logistic regression, random forest, extra trees, xgboost.
- Separate MiniBatch K-means segmentation with training-only selection and seed-stability checks.
- Time-ordered data splits, validation-calibrated alert thresholds and saved model pipelines.
- FastAPI dashboard for predictions, metrics and downloadable evaluation files.

## Quick start

```bash
git clone --branch paysim --single-branch https://github.com/KennyDao1324/fraud-detection-datasets.git
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
| `experiment_suite/pipelines/paysim/` | Dataset-specific implementation and configuration |
| `experiment_suite/evaluation/paysim/` | Saved pipelines, metrics, charts and model cards |
| `experiment_suite/dataset/paysim/` | Local raw input files; excluded from Git |
| `experiment_suite/app.py` | Local API and dashboard server |
| `training_model/train.py` | Classification launcher |
| `training_model/cluster.py` | Clustering launcher |
| `src/`, `scripts/`, `docs/` | Original dataset implementation, utilities and provenance |
| `artifacts/`, `checkpoint/`, `outputs/` | Earlier baseline models and evaluation evidence |

Small `__init__.py`, `.gitkeep` and launcher files support packaging or folder layout; the main implementation lives in the source and training modules.

## Classification results

Experiment: `feature_policy_v4`. Selected model: **random_forest**. The table uses the 1% calibration-FPR operating point; actual later-period FPR can differ.

| Model | PR-AUC | Recall | FPR | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.2733 | 44.5% | 1.875% | 24.4% | 0.3150 | 0.8935 |
| random_forest | 0.5261 | 46.5% | 0.598% | 51.3% | 0.4878 | 0.9427 |
| extra_trees | 0.5446 | 52.0% | 0.793% | 47.1% | 0.4943 | 0.9461 |
| xgboost | 0.6414 | 57.8% | 0.545% | 59.0% | 0.5842 | 0.9594 |

[Model comparison](experiment_suite/evaluation/paysim/feature_policy_v4/model_comparison.png) · [PR curves](experiment_suite/evaluation/paysim/feature_policy_v4/precision_recall_curves.png) · [Metrics CSV](experiment_suite/evaluation/paysim/feature_policy_v4/model_comparison.csv) · [Full results and protocol](experiment_suite/evaluation/paysim/feature_policy_v4/report.json)

Model settings and thresholds were selected using chronological validation. Test periods have already been inspected, so comparisons are exploratory rather than a fresh holdout. A score supports review and does not establish fraud or authorize an account decision.

[Model card](experiment_suite/evaluation/paysim/feature_policy_v4/MODEL_CARD.md) · [Threshold-policy comparison](experiment_suite/evaluation/paysim/feature_policy_v4/threshold_policies.csv)

The dashboard supports five calibration FPR budgets, optional review-capacity limits and an approximate one-sided 95% calibration FPR bound. These are calibration diagnostics, not guarantees of future FPR or alert volume.

## Clustering

MiniBatch K-means selected **k=8**. Training silhouette: **0.446**; mean seed ARI: **0.864**.

Passed the stated size/repeatability floors; this does not establish real-world or fraud relevance.

Preprocessing and k selection use training inputs only. Fraud labels and label-derived features are excluded from the inputs and used only afterward to interpret cluster fraud rates. Clusters are not fraud predictions; silhouette is not classification accuracy.

[Interpretation](experiment_suite/evaluation/paysim/clustering_v1/README.md) · [Silhouette and stability](experiment_suite/evaluation/paysim/clustering_v1/silhouette_stability.png) · [Cluster sizes](experiment_suite/evaluation/paysim/clustering_v1/cluster_sizes.png) · [Feature profiles](experiment_suite/evaluation/paysim/clustering_v1/feature_profiles.png) · [Fraud-rate interpretation](experiment_suite/evaluation/paysim/clustering_v1/fraud_rates.png)

## Training

Place the original downloaded files in `experiment_suite/dataset/paysim/`. Keep the raw files local and follow their source licences. From `experiment_suite/`, prepare the data:

```bash
python -c "import sys; sys.path.insert(0, 'training_model'); from priority_data import prepare; prepare('paysim')"
python training_model/feature_policy_search.py --dataset paysim
python training_model/clustering.py --dataset paysim
```

Completed experiment folders are preserved and may report already complete or reuse cached fits. Use a separate working copy and a new versioned output directory for a new search; do not overwrite the published evidence. Run classification and clustering separately.

## Dataset and licence

- [PaySim on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)
- Licence: CC BY-SA 4.0
- Target: `isFraud`
- Time field: `step` (one hour per step)

The project does not redistribute the raw CSV. PaySim is synthetic and its results do not establish
performance on real customers or real mobile-money systems.

Raw datasets and row-level prediction exports are not redistributed here. Code and dataset licences are separate; check the original source terms before reuse.

## Citation

E. A. Lopez-Rojas, A. Elmir, and S. Axelsson, “PaySim: A financial mobile money simulator for fraud
detection,” 28th European Modeling and Simulation Symposium, 2016.
