# Environment and prepared-data workflow

Run in `experiment_suite/` from PyCharm Terminal or a shell:

```bash
conda env create -f environment.yml
conda activate fraudsuite
python -m pytest tests/test_prepared_release.py -q
python app.py
```

In a second terminal with the same environment:

```bash
python training_model/export_final_model_data.py
python training_model/reproduce_from_prepared.py score
python training_model/reproduce_from_prepared.py refit
python training_model/reproduce_from_prepared.py predict --record examples/baf_raw_record.json
```

The raw-input export is `final_model_data/Base_v4_final_model_data.csv`: 650,000 rows, 32 columns, 141,799,764 bytes. Its 26 model inputs rebuild the 60 engineered features exactly; protected fields are audit-only.

Export requires the original locally installed dataset or prepared cache. Once exported, score and refit read the CSV and adjacent metadata only; keep those files together. CSV files and row-level predictions are not redistributed in Git. Request them from the dataset owner and verify the manifest checksum. Never unpickle an untrusted checkpoint.

Refit uses the selected hyperparameters and train rows only, with thresholds calibrated on the calibration period. It writes a separate checkpoint under `checkpoint/baf/prepared_refit/` and does not replace the frozen champion. Existing exports/refits are preserved; use a new output directory to repeat them. Generic prediction expects prepared features, not arbitrary raw transactions.

The API/dashboard default to this branch's selected model and the 5% calibration-FPR budget; 1% remains secondary. Actual later FPR can differ. Test periods have been inspected previously and are not a fresh holdout.

Python/model-library versions are pinned. Conda installs Python and pip; pip installs the locked libraries. Mac/Linux training uses file locks. Four supervised model families and clustering remain comparison evidence.
