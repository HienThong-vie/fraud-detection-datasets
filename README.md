## Two method types: classification + clustering

The four supervised classifiers are unchanged. A separate exploratory MiniBatch K-means study groups similar records; clusters are not fraud predictions.

Selected k: **3**; training silhouette: **0.467**; mean seed ARI: **0.868**.

Passed the stated size/repeatability floors; this does not establish real-world or fraud relevance.

Preprocessing, k selection and repeatability checks use training inputs only. Labels, identifiers, label-derived features, absolute event-time columns and Elliptic neighbour aggregates are excluded. Fraud rates are interpreted afterward, never used to choose clusters. Validation/test records are assigned without refitting. Previously viewed test periods remain exploratory.

- [Interpretation and limitations](experiment_suite/evaluation/ieee/clustering_v1/README.md)
- [Silhouette and seed stability](experiment_suite/evaluation/ieee/clustering_v1/silhouette_stability.png)
- [Cluster sizes](experiment_suite/evaluation/ieee/clustering_v1/cluster_sizes.png)
- [Feature profiles](experiment_suite/evaluation/ieee/clustering_v1/feature_profiles.png)
- [Post-hoc fraud rates](experiment_suite/evaluation/ieee/clustering_v1/fraud_rates.png)
- [Metrics and saved clustering pipeline](experiment_suite/evaluation/ieee/clustering_v1/)

Run `python training_model/cluster.py` from the branch root after supplying the raw data as described below. This entrypoint preserves completed runs. The local dashboard shows a separate clustering section. The main training entrypoint still runs classification only.

Silhouette is not accuracy, and seed repeatability does not establish temporal stability. Weak clustering is a documented finding, not evidence of successful fraud detection.

---

# IEEE: latest supervised experiment

Latest: `feature_policy_v4`. Four supervised families; selected champion: **xgboost**. Training ran locally on the Mac CPU. Selection uses chronological validation, not test scores.

| Model | Test AP* | Recall* | Actual FPR* | Precision* | F1* | ROC-AUC* |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.1818 | 33.2% | 2.621% | 31.3% | 0.3224 | 0.8218 |
| random_forest | 0.4635 | 40.4% | 1.308% | 52.7% | 0.4571 | 0.8679 |
| extra_trees | 0.3852 | 35.6% | 1.290% | 49.9% | 0.4158 | 0.8417 |
| xgboost | 0.5235 | 45.2% | 1.409% | 53.6% | 0.4904 | 0.9028 |

*Previously inspected test periods are exploratory, not a fresh holdout. The displayed operating point uses a 1% calibration FPR budget; future FPR can differ. Scores support review and do not establish fraud. No business deployment threshold has been approved.

- [Model comparison](experiment_suite/evaluation/ieee/feature_policy_v4/model_comparison.png)
- [Precision–recall curves](experiment_suite/evaluation/ieee/feature_policy_v4/precision_recall_curves.png)
- [All metrics](experiment_suite/evaluation/ieee/feature_policy_v4/model_comparison.csv)
- [Protocol and results](experiment_suite/evaluation/ieee/feature_policy_v4/report.json)
- [Saved models and all graphs](experiment_suite/evaluation/ieee/feature_policy_v4/)

- [Threshold policies](experiment_suite/evaluation/ieee/feature_policy_v4/threshold_policies.csv)
- [Feature comparison](experiment_suite/evaluation/ieee/feature_policy_v4/candidate_feature_comparison.png)
- [Temporal checks](experiment_suite/evaluation/ieee/feature_policy_v4/temporal_validation.png)
- [Model card and limitations](experiment_suite/evaluation/ieee/feature_policy_v4/MODEL_CARD.md)

The dashboard offers 0.5%, 1%, 2%, 3% and 5% calibration FPR budgets, an approximate one-sided 95% calibration FPR bound, and review-capacity limits. These constraints apply to calibration data and do not guarantee future compliance.

## Run in PyCharm

Open this branch as a project, create a Python 3.12 virtual environment, then use the terminal:

```bash
python -m pip install -r experiment_suite/requirements.txt
python experiment_suite/app.py
```

Open http://127.0.0.1:8050/. Stop another suite server first if the port is in use. Saved models work without raw data; inputs must contain the engineered columns shown by the dashboard.

## Reproduce training

Put downloaded source files in `experiment_suite/dataset/ieee/`. Do not commit raw data. Run from `experiment_suite/`:

```bash
python -c "import sys; sys.path.insert(0, 'training_model'); from priority_data import prepare; prepare('ieee')"
```

```bash
python training_model/feature_policy_search.py --dataset ieee
```

Completed experiment folders are preserved. A completed run may report already complete or reuse cached fits. For a genuinely new search, use a separate working copy and a new versioned experiment directory; do not overwrite the published evidence. Mac/Linux is required for the training locks (`fcntl`); dashboard inference does not need those locks.

The full implementation lives in `experiment_suite/training_model/`; original project code and documentation remain below. Production baseline artifacts are retained separately.

---

## Earlier project documentation, sources and licences

# IEEE: four supervised-model comparison

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

# IEEE-CIS Fraud Detection ML

This is a standalone machine-learning project for the IEEE-CIS Fraud Detection dataset. It is separate
from the BAF project and has its own environment, data links, models, outputs, tests, API, and Git history.

The pipeline merges transaction and identity records by `TransactionID`, uses a chronological
train/validation/test split, compares SGD logistic classification, histogram gradient boosting, and
XGBoost, and selects the model with the highest validation PR-AUC. The held-out internal test period is
not used for selection.

## Dataset source

- [IEEE-CIS Fraud Detection competition and dataset](https://www.kaggle.com/competitions/ieee-fraud-detection)

Kaggle requires an account and acceptance of the competition rules before download. Those rules apply to
the data independently of this project's code licence, so the raw CSV files are not redistributed here.

## Standard team folder layout

| Folder | Purpose |
| --- | --- |
| `checkpoint/` | Production model and model-card links |
| `dataset/` | Direct access to the five supplied `.csv` files |
| `evaluation/` | Metrics, comparison tables, prediction CSV, and PNG charts |
| `preprocess_data/` | Merge, validation, profiling, and hashing entrypoint |
| `preprocessed_df/` | Merged train-ready `.csv.gz` table with split labels |
| `training_model/` | Main model-training and submission entrypoint |
| `utils/` | Shared paths used by the workflow scripts |

### Why some files are intentionally short

Some files in these folders contain only one or two lines because they support the project structure rather
than implement the machine-learning pipeline:

- `.gitkeep` keeps an otherwise empty output folder visible in Git.
- `__init__.py` marks a directory as a Python package and may only contain a package description or version.
- Folder-level `README.md` files briefly describe what belongs in that folder.
- Small CSV and JSON files are generated summaries or metadata, not Python source code.

The main implementation is under `src/ieee_fraud/`. The scripts in `preprocess_data/`, `training_model/`, and
`evaluation/` are convenient entrypoints that call that shared implementation without duplicating it.

Use the standardized workflow from the project root:

```bash
python preprocess_data/run.py
python training_model/train.py
python evaluation/evaluate.py
```

## Project map

| Location | Purpose |
| --- | --- |
| `src/ieee_fraud/data.py` | CSV loading, identity merge, feature preparation, and temporal split |
| `src/ieee_fraud/training.py` | Model comparison, selection, refitting, and prediction CSV creation |
| `src/ieee_fraud/inference.py` | Local JSON prediction |
| `src/ieee_fraud/api.py` | FastAPI prediction service |
| `src/ieee_fraud/profiling.py` | Dataset statistics and SHA-256 evidence |
| `configs/training.json` | Laptop-safe sample limits and model settings |
| `outputs/test_predictions.csv` | Kaggle-format `TransactionID,isFraud` output |
| `artifacts/model_card.json` | Split, selection, metrics, and configuration record |
| `docs/DATASET.md` | Dataset handling notes |

## Evaluation results

Team members can start from the [evaluation guide](evaluation/README.md) or open the
[complete outputs folder](outputs/). The main report figures are linked below.

- [Confusion matrix](outputs/confusion_matrix.png)
- [Model comparison chart](outputs/model_comparison.png)
- [Prediction distribution](outputs/prediction_distribution.png)
- [Model comparison table](outputs/model_comparison.csv)
- [Complete metrics](outputs/metrics.json)

![IEEE-CIS confusion matrix](outputs/confusion_matrix.png)

![IEEE-CIS model comparison](outputs/model_comparison.png)

![IEEE-CIS prediction distribution](outputs/prediction_distribution.png)

XGBoost was selected by validation PR-AUC (0.5403). On the held-out internal test period it achieved
PR-AUC **0.4795**, ROC-AUC **0.8876**, precision **0.2627**, recall **0.6070**, and F1 **0.3668**.
The validation threshold targets a 5% false-positive rate; the observed test false-positive rate was
6.14% under temporal drift.

## Setup in PyCharm

Open this folder as the project:

```text
/Users/daoduykha/Documents/Playground/ieee-fraud-ml
```

This computer is already configured to use the project-local interpreter:

```text
/Users/daoduykha/Documents/Playground/ieee-fraud-ml/.venv/bin/python
```

For a fresh clone on another computer, create the same environment in PyCharm's terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Select `.venv/bin/python` under **Settings > Project > Python Interpreter**. The included
`environment.yml` remains available for teammates who prefer Conda.

## Connect the supplied data

The linking command avoids making another 1.3 GB copy:

```bash
python scripts/link_data.py "/Users/daoduykha/Downloads/ieee-fraud-detection 2"
```

Confirm the links:

```bash
ls -lh data/raw
```

Raw files are ignored by Git.

## Train and create the output CSV

```bash
python -m ieee_fraud.training
```

The command creates:

- `artifacts/production_model.joblib`
- `artifacts/model_card.json`
- `outputs/model_comparison.csv`
- `outputs/metrics.json`
- `outputs/test_predictions.csv`

The last file is the prediction output for all rows in `test_transaction.csv`, using the same columns as
`sample_submission.csv`.

The default configuration trains on a reproducible class-stratified sample from each chronological period
to fit on a laptop. Change the `max_*_rows` values in `configs/training.json` to `null` to train on all
available labeled rows.

## Profile the dataset

```bash
python -m ieee_fraud.profiling
```

This writes `outputs/dataset_profile.json` with row counts, fraud rate, missingness, file sizes, and hashes.
Hashing all 1.3 GB can take several minutes.

## Run the API

```bash
python -m uvicorn ieee_fraud.api:app --host 127.0.0.1 --port 8020
```

Open <http://127.0.0.1:8020/docs>. The root endpoint is also implemented, so opening
<http://127.0.0.1:8020/> will not return a 404.

Example request body:

```json
{
  "features": {
    "TransactionAmt": 68.5,
    "ProductCD": "W",
    "card1": 13926,
    "card4": "discover",
    "card6": "credit"
  }
}
```

## Command-line prediction

```bash
python -m ieee_fraud.inference examples/sample_transaction.json
```

## Verification

```bash
pytest -q
ruff check .
```

This is an educational decision-support prototype. A high score means human review is recommended; it is
not proof of fraud and must not automatically reject or block a transaction.

