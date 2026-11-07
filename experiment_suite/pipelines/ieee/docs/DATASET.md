# Dataset record

This project uses the IEEE-CIS Fraud Detection competition files supplied by the team on 5 November
2026. The local dataset folder contains:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

The raw files total approximately 1.3 GB and are not copied into Git. Run `python scripts/link_data.py`
as described in the README to create local symbolic links. The test identity file uses hyphenated `id-*`
column names while the training identity file uses underscored `id_*` names; the loader normalizes both
to the underscored form before merging.

The data was distributed for a Kaggle competition. Check the competition rules before redistributing it
or using it outside the permitted educational context.

