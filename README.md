# Sparkov Fraud Detection

A Python project for fraud-data analysis, supervised classification and exploratory clustering. It includes trained pipelines, a local dashboard, evaluation charts and reproducible training code.

## Features

- Four supervised classifiers: logistic regression, random forest, extra trees, catboost.
- Separate MiniBatch K-means segmentation with training-only selection and seed-stability checks.
- Time-ordered data splits, validation-calibrated alert thresholds and saved model pipelines.
- FastAPI dashboard for predictions, metrics and downloadable evaluation files.

## Quick start

```bash
git clone --branch sparkov --single-branch https://github.com/KennyDao1324/fraud-detection-datasets.git
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
| `experiment_suite/pipelines/sparkov/` | Dataset-specific implementation and configuration |
| `experiment_suite/evaluation/sparkov/` | Saved pipelines, metrics, charts and model cards |
| `experiment_suite/dataset/sparkov/` | Local raw input files; excluded from Git |
| `experiment_suite/app.py` | Local API and dashboard server |
| `training_model/train.py` | Classification launcher |
| `training_model/cluster.py` | Clustering launcher |
| `src/`, `scripts/`, `docs/` | Original dataset implementation, utilities and provenance |
| `artifacts/`, `checkpoint/`, `outputs/` | Earlier baseline models and evaluation evidence |

Small `__init__.py`, `.gitkeep` and launcher files support packaging or folder layout; the main implementation lives in the source and training modules.

## Classification results

Experiment: `fit_balance_v3`. Selected model: **catboost**. The table uses the 5% calibration-FPR headline operating point; actual later-period FPR can differ. 1% is secondary.

| Model | PR-AUC | Recall | FPR | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.1531 | 74.1% | 4.955% | 5.5% | 0.1020 | 0.9140 |
| random_forest | 0.8440 | 97.9% | 5.078% | 7.0% | 0.1299 | 0.9953 |
| extra_trees | 0.8062 | 93.8% | 4.921% | 6.9% | 0.1281 | 0.9827 |
| catboost | 0.8859 | 97.9% | 4.939% | 7.1% | 0.1330 | 0.9939 |

[Model comparison](experiment_suite/evaluation/sparkov/fit_balance_v3/model_comparison.png) · [PR curves](experiment_suite/evaluation/sparkov/fit_balance_v3/precision_recall_curves.png) · [Metrics CSV](experiment_suite/evaluation/sparkov/fit_balance_v3/model_comparison.csv) · [Full results and protocol](experiment_suite/evaluation/sparkov/fit_balance_v3/report.json)

Model settings and thresholds were selected using chronological validation. Test periods have already been inspected, so comparisons are exploratory rather than a fresh holdout. A score supports review and does not establish fraud or authorize an account decision.

## Clustering

MiniBatch K-means selected **k=5**. Training silhouette: **0.141**; mean seed ARI: **0.281**.

Weak/unstable geometry: no candidate satisfied both quality floors; do not claim meaningful fraud groups.

Preprocessing and k selection use training inputs only. Fraud labels and label-derived features are excluded from the inputs and used only afterward to interpret cluster fraud rates. Clusters are not fraud predictions; silhouette is not classification accuracy.

[Interpretation](experiment_suite/evaluation/sparkov/clustering_v1/README.md) · [Silhouette and stability](experiment_suite/evaluation/sparkov/clustering_v1/silhouette_stability.png) · [Cluster sizes](experiment_suite/evaluation/sparkov/clustering_v1/cluster_sizes.png) · [Feature profiles](experiment_suite/evaluation/sparkov/clustering_v1/feature_profiles.png) · [Fraud-rate interpretation](experiment_suite/evaluation/sparkov/clustering_v1/fraud_rates.png)

## Training

Place the original downloaded files in `experiment_suite/dataset/sparkov/`. Keep the raw files local and follow their source licences. From `experiment_suite/`, prepare the data:

```bash
python -c "import sys; sys.path.insert(0, 'training_model'); from priority_data import prepare; prepare('sparkov')"
python training_model/fit_balance_search.py --dataset sparkov
python training_model/clustering.py --dataset sparkov
```

Completed experiment folders are preserved and may report already complete or reuse cached fits. Use a separate working copy and a new versioned output directory for a new search; do not overwrite the published evidence. Run classification and clustering separately.

## Dataset and licence

- [Kaggle source](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- Synthetic data generated with Sparkov; distributor licence: CC0 1.0
- Provided files: `fraudTrain.csv` and `fraudTest.csv`

No raw customer names, addresses, card numbers, transaction IDs or merchant names are used as model features.


Raw datasets and row-level prediction exports are not redistributed here. Code and dataset licences are separate; check the original source terms before reuse.

## Citation

Brandon Harris, *Sparkov Data Generation*, and Kartik Shenoy, *Credit Card Transactions Fraud Detection Dataset*.

[Selected-model identity and operating points](FINAL_MODEL.md) · [Conda workflow](experiment_suite/CONDA_GUIDE.md)

## Prepared data and reproducibility

[CSV export, scoring, refitting and prediction commands](experiment_suite/CONDA_GUIDE.md). Each export has a data dictionary and checksum manifest. Refit outputs are separate from the selected checkpoint.
