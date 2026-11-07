# Training run log

- Dataset: Sparkov synthetic credit-card transactions
- Split: chronological development/validation plus later official test file
- Sample rows: 200,000 train; 50,000 validation; 50,000 test
- Selected model: random forest
- Test PR-AUC: 0.8369
- Test precision / recall / F1: 0.8563 / 0.7409 / 0.7944
- Command: `python training_model/train.py`
