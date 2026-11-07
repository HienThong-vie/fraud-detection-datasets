# HANDBOOK: four supervised-model comparison

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

# Fraud Detection Handbook Simulated-Data ML

This is a standalone, ML-only project for the Fraud Detection Handbook's simulated card-transaction data.
It creates prior-only customer and terminal behavior features, uses chronological train/validation/test
periods, compares class-weighted logistic regression, random forest, and XGBoost, and selects by validation
PR-AUC before one held-out test evaluation.

## Dataset source

- [Official simulated-data repository](https://github.com/Fraud-Detection-Handbook/simulated-data-raw)
- [Fraud Detection Handbook](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook)
- [Official simulation notebook](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook/blob/main/Chapter_3_GettingStarted/SimulatedDataset.ipynb)

The Handbook code repository uses GPL-3.0. This project does not copy its implementation; it has an original
pipeline for the published output data. The separate raw-data repository does not display an explicit data
licence file, so verify redistribution permission. Raw files are kept local and ignored by Git.

## Open in PyCharm on this Mac

Open this folder as a separate PyCharm project:

```text
/Users/daoduykha/Documents/Playground/fraud-handbook-ml
```

In PyCharm, choose **Settings → Project → Python Interpreter → Add Interpreter → Virtualenv** and create
`.venv` inside this project. Then run in the PyCharm terminal:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Download the dataset

The downloader is pinned to the official data commit recorded in `configs/training.json`:

```bash
python scripts/download_handbook.py
```

This downloads daily files from 2018-04-01 through 2018-09-30 into `data/raw/`. For a quick download test:

```bash
python scripts/download_handbook.py --limit-days 3
```

Security note: these files use Python pickle format. Pickles can execute code when opened. Only use the files
downloaded from the pinned official source; never place an unknown `.pkl` file in `data/raw/`.

## Run the complete workflow

Run from the project root:

```bash
python preprocess_data/run.py
python training_model/train.py
python evaluation/evaluate.py
```

Run a local prediction after training:

```bash
python -m handbook_fraud.inference examples/sample_transaction.json
```

Run the optional local API:

```bash
python -m uvicorn handbook_fraud.api:app --host 127.0.0.1 --port 8040
```

Open <http://127.0.0.1:8040/docs>.

## Team folder layout

| Folder | Purpose |
| --- | --- |
| `checkpoint/` | Fitted `.joblib` pipeline and model card |
| `dataset/` | Official source links, licence notes, and data safety warning |
| `evaluation/` | Evaluation entrypoint and links to outputs |
| `preprocess_data/` | Daily-file loading, feature engineering, splitting, and profiling |
| `preprocessed_df/` | Reproducible train-ready sampled table |
| `training_model/` | Model-training entrypoint |
| `utils/` | Shared project path helpers |
| `src/handbook_fraud/` | Main implementation |
| `outputs/` | Metrics, predictions, and PNG report charts |

Some files are intentionally short. `.gitkeep` preserves an empty folder in Git, `__init__.py` marks a
Python package, and the workflow entrypoints call the shared implementation under `src/`.

## Features and leakage controls

Model inputs are:

- transaction amount, hour, and day of week
- number of earlier transactions for the customer
- customer's mean amount over earlier transactions
- time since the customer's previous transaction
- number of earlier transactions at the terminal
- time since the terminal's previous transaction

Controls:

- Every behavioral feature is calculated from earlier transactions only.
- `TX_FRAUD_SCENARIO` is excluded because it directly describes how fraud was injected.
- `CUSTOMER_ID` and `TERMINAL_ID` are excluded to avoid memorising synthetic entities.
- Splits are chronological by simulated day.
- Sampling occurs after splitting, preserves class ratios, and has Mac-safe limits in
  `configs/training.json`.
- The test period is not used for model selection or threshold selection.

## Outputs

After training, [open the evaluation guide](evaluation/README.md) or the `outputs/` folder. Expected files:

- `dataset_profile.json`: file hashes, row counts, time range, and fraud rate
- `model_comparison.csv`: validation metrics for all candidate models
- `metrics.json`: selected model and held-out test metrics
- `test_predictions.csv`: held-out labels and fraud scores
- `model_comparison.png`
- `precision_recall_curves.png`
- `confusion_matrix.png`
- `prediction_distribution.png`

### Current trained result

Random forest was selected by validation PR-AUC. The held-out 50,000-row test sample produced PR-AUC
0.2511, ROC-AUC 0.6316, precision 0.9697, and recall 0.2143. See [all evaluation files](outputs/) and the
[evaluation guide with chart previews](evaluation/README.md).
XGBoost was evaluated as a third candidate, but its validation PR-AUC (0.1904) did not exceed the random
forest result (0.2508), so it was not selected.

![Handbook model comparison](outputs/model_comparison.png)

![Handbook confusion matrix](outputs/confusion_matrix.png)

## Tests

```bash
pytest -q
ruff check .
```

The tests generate a small transaction history and do not require the downloaded dataset.
