# Training run log

- Dataset: real anonymized Elliptic Bitcoin transaction graph
- Split: steps 1–34 train; 35–41 validation; 42–49 test
- Labelled rows: 29,894 train; 7,829 validation; 8,841 test
- Selected model: random forest
- Test PR-AUC: 0.5467
- Test precision / recall / F1: 0.9183 / 0.4681 / 0.6201
- Temporal shift note: validation PR-AUC was 0.9289; the later test result is retained unchanged.
- Command: `python training_model/train.py`
