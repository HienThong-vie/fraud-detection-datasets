# Fraud Detection Handbook simulated-data record

- Data repository: https://github.com/Fraud-Detection-Handbook/simulated-data-raw
- Handbook: https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook
- Pinned data commit: `6e67dbd0a3bfe0d7ec33abc4bce5f37cd4ff0d6a`
- Format: one trusted pickle file per simulated day
- Target: `TX_FRAUD`
- Time fields: `TX_DATETIME`, `TX_TIME_SECONDS`, and `TX_TIME_DAYS`

The Handbook implementation repository uses GPL-3.0. This project does not copy its implementation; it
contains an original scikit-learn pipeline that consumes the published output data. The separate raw-data
repository does not display an explicit licence file, so verify permission before redistribution. Raw files
remain local and are ignored by Git.

`TX_FRAUD_SCENARIO` and raw entity IDs are excluded from model inputs. Customer and terminal behavior
features are computed using earlier transactions only, before chronological train/validation/test periods.
