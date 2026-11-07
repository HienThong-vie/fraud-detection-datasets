# Data leakage checklist

This checklist records the controls implemented by the training pipeline. The machine-readable copy is
generated at `outputs/leakage_checklist.json` during training.

| Check | Status | Evidence |
| --- | :---: | --- |
| Development and test months are chronological and disjoint | Pass | Months 0–4 train, month 5 validation, months 6–7 test in `configs/training.json` |
| Test rows are excluded from preprocessing fit | Pass | Each sklearn pipeline is fitted only with `X_train` in `training.py` |
| Test rows are excluded from model fitting | Pass | Estimators receive only training rows |
| Test rows are excluded from threshold selection | Pass | Threshold is selected using month-5 validation scores |
| Test metrics are excluded from winner selection | Pass | Winner is selected only by validation PR-AUC |
| Target and month are excluded from model inputs | Pass | `make_temporal_data` removes both columns |
| Protected diagnostic fields are excluded from model inputs | Pass | Age, income, and employment status are removed before fitting |
| Synthetic separability fields are excluded from model inputs | Pass | `x1` and `x2` are profiled but excluded from Variants III and V models |
| Missing-value processing is fitted only on training rows | Pass | Imputers are inside the sklearn pipeline |
| Missing-code conversion is deterministic and documented | Pass | All negative `intended_balcon_amount` values and exact `-1` codes in the other documented fields are converted before splitting |
| Imbalance treatment is limited to model fitting | Pass | Class weights are estimator parameters; no pre-split resampling occurs |
| Sampling is reproducible | Pass | Sampling is stratified by month/label with seed 42 |
| Duplicate counts are recorded | Pass | Each generated dataset profile contains `duplicate_rows` |
| Raw and processed datasets are excluded from Git | Pass | `.gitignore` excludes both directories |

If resampling is introduced later, it must be placed inside an imbalanced-learn pipeline and applied only
to training folds. It must never run before the temporal split.
