# SPARKOV: latest supervised experiment

Latest: `fit_balance_v3`. Four supervised families; selected champion: **catboost**. Training ran locally on the Mac CPU. Selection uses chronological validation, not test scores.

| Model | Test AP* | Recall* | Actual FPR* | Precision* | F1* | ROC-AUC* |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.1531 | 51.3% | 0.970% | 17.0% | 0.2555 | 0.9140 |
| random_forest | 0.8440 | 95.3% | 1.150% | 24.3% | 0.3874 | 0.9953 |
| extra_trees | 0.8062 | 86.0% | 1.142% | 22.6% | 0.3578 | 0.9827 |
| catboost | 0.8859 | 94.8% | 1.076% | 25.5% | 0.4013 | 0.9939 |

*Previously inspected test periods are exploratory, not a fresh holdout. The displayed operating point uses a 1% calibration FPR budget; future FPR can differ. Scores support review and do not establish fraud. No business deployment threshold has been approved.

- [Model comparison](experiment_suite/evaluation/sparkov/fit_balance_v3/model_comparison.png)
- [Precision–recall curves](experiment_suite/evaluation/sparkov/fit_balance_v3/precision_recall_curves.png)
- [All metrics](experiment_suite/evaluation/sparkov/fit_balance_v3/model_comparison.csv)
- [Protocol and results](experiment_suite/evaluation/sparkov/fit_balance_v3/report.json)
- [Saved models and all graphs](experiment_suite/evaluation/sparkov/fit_balance_v3/)

This branch retains its previously completed experiment; the latest feature-policy retraining targeted BAF, IEEE, PaySim and Elliptic only.

## Run in PyCharm

Open this branch as a project, create a Python 3.12 virtual environment, then use the terminal:

```bash
python -m pip install -r experiment_suite/requirements.txt
python experiment_suite/app.py
```

Open http://127.0.0.1:8050/. Stop another suite server first if the port is in use. Saved models work without raw data; inputs must contain the engineered columns shown by the dashboard.

## Reproduce training

Put downloaded source files in `experiment_suite/dataset/sparkov/`. Do not commit raw data. Run from `experiment_suite/`:

```bash
python -c "import sys; sys.path.insert(0, 'training_model'); from priority_data import prepare; prepare('sparkov')"
```

```bash
python training_model/fit_balance_search.py --dataset sparkov
```

Completed experiment folders are preserved. A completed run may report already complete or reuse cached fits. For a genuinely new search, use a separate working copy and a new versioned experiment directory; do not overwrite the published evidence. Mac/Linux is required for the training locks (`fcntl`); dashboard inference does not need those locks.

The full implementation lives in `experiment_suite/training_model/`; original project code and documentation remain below. Production baseline artifacts are retained separately.

---

## Earlier project documentation, sources and licences

# SPARKOV: four supervised-model comparison

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

# Sparkov Synthetic Credit-Card Fraud ML

Standalone PyCharm project for the Sparkov synthetic transaction dataset. It compares class-weighted
logistic regression, random forest, and CatBoost, selects by validation PR-AUC, and evaluates once on the
later official test period. CatBoost keeps the dataset's categorical fields native instead of one-hot
expanding them.

## Dataset

- [Kaggle source](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- Synthetic data generated with Sparkov; distributor licence: CC0 1.0
- Provided files: `fraudTrain.csv` and `fraudTest.csv`

No raw customer names, addresses, card numbers, transaction IDs or merchant names are used as model features.

## PyCharm setup

Open the whole `sparkov-fraud-ml` folder, create a Python 3.11 virtual environment, then run:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Place the files here:

```text
data/raw/fraudTrain.csv
data/raw/fraudTest.csv
```

Run:

```bash
python preprocess_data/run.py
python training_model/train.py
python evaluation/evaluate.py
python -m fraudbench.inference examples/sample_transaction.json
python -m uvicorn fraudbench.api:app --host 127.0.0.1 --port 8040
```

Open <http://127.0.0.1:8040/docs> for the API.

## Folder guide

| Folder | Contents |
|---|---|
| `dataset` | Provenance, licence and required filenames |
| `preprocess_data` | Data validation and feature-engineering entrypoint |
| `preprocessed_df` | Generated train-ready table |
| `training_model` | Training entrypoint |
| `checkpoint` | `.joblib` model and model card |
| `evaluation` | Evaluation entrypoint |
| `outputs` | Metrics CSV/JSON and PNG charts |
| `utils` | Team helper scripts |

Some files are intentionally short: entrypoints call the reusable implementation under `src/fraudbench`, and `.gitkeep` preserves empty folders.

## Leakage controls

- The original training file is split chronologically for development and validation.
- The later official test file is untouched until final evaluation.
- Sampling happens after the temporal split and preserves the class ratio.
- Preprocessing is fitted only inside each scikit-learn training pipeline.
- Identity fields are excluded to prevent memorisation.

## Citation

Brandon Harris, *Sparkov Data Generation*, and Kartik Shenoy, *Credit Card Transactions Fraud Detection Dataset*.

## Current trained result

CatBoost was selected with validation PR-AUC **0.9130**. On the later 50,000-row test sample it achieved
PR-AUC **0.8643**, ROC-AUC **0.9908**, precision **0.7843**, recall **0.8290**, and F1 **0.8060**. See
[`outputs/metrics.json`](outputs/metrics.json) and [`outputs/model_comparison.csv`](outputs/model_comparison.csv).

![Sparkov model comparison](outputs/model_comparison.png)

![Sparkov confusion matrix](outputs/confusion_matrix.png)
