# AP, recall and false-positive-rate experiment (v2)

Run in PyCharm with the suite `.venv/bin/python`, or from its terminal:

```bash
python utils/run_priority_search.py
python utils/run_priority_search.py --datasets handbook baf
```

Runs sequentially on the local Mac CPU, with four threads per estimator. Completed candidates
are cached so interrupted training can resume. Completed reports are skipped. Do not change
configuration or input data and reuse an old candidate cache; use a new experiment version.
Smoke tests use separate directories and never appear as full training results.

On this 32 GB / 10-core Mac, a second reverse-order queue can be run with
`python utils/run_priority_search.py --datasets ulb banksim sparkov paysim ieee elliptic --status-name supplemental`.
Use at most these two queues (eight estimator threads total). Per-dataset file locks prevent
duplicate training/output writes; completed reports are skipped. Supplemental status is separate.
The `fcntl` lock is macOS/Linux-specific; Windows needs an equivalent lock before running concurrent queues.

## Four improvements

1. Calibrate thresholds for FPR budgets of 0.5%, 1% and 5%, selecting the lowest cutoff that
   respects each calibration budget, including tied scores. This maximizes available calibration
   recall within the budget. Report actual selection/test FPR; budgets are not guarantees.
2. Select by worst chronological selection-block utility: 65% average precision + 35% average
   recall across budgets, penalizing FPR excess and the positive train/selection AP gap.
   ROC-AUC and accuracy are diagnostic only. Default operating budget is 1%, a demonstration
   choice rather than a claim about a bank's acceptable customer friction.
3. Rebuild from raw data with time metadata. Larger training caps: BAF 500k, IEEE 300k,
   PaySim 300k, Handbook 300k, Sparkov 400k, BankSim 300k; ULB/Elliptic use all training rows.
   Strongly regularized candidates compete with v1 settings retrained under the same new
   protocol. Elliptic additionally examines recent training history (steps >=20) within each
   family's candidates. No test labels select configurations or thresholds.
4. Keep exactly four supervised families per dataset: logistic regression, random forest,
   extra trees and CatBoost (BAF/Sparkov/BankSim) or XGBoost (others). Three candidates per
   family means 12 fits/dataset, 96 planned full fits. Graphs compare all four and include
   recall/actual test FPR at all budgets.

## Chronology and features

Training precedes calibration, which precedes model selection and then test. Validation is
split at event-time group boundaries, never by random interleaving. Selection uses two
chronological blocks where both contain both labels, otherwise one combined chronological
selection period. BAF only supplies month-level time, so train months 0–3, calibrate month 4,
select month 5, test months 6–7. This differs from v1; do not attribute changes solely to models.

Handbook adds trailing 1/7/30-day customer and terminal counts, means and amount-to-history
ratios. All exclude current/simultaneous transactions. Terminal fraud rates use a strict
seven-day outcome delay. This is an explicit **online replay**: an outcome from a past
validation/test transaction may inform a later transaction only after its delay. A deployment
without this feedback stream must omit those features and retrain. Customer/terminal IDs,
fraud scenarios and labels themselves are not model inputs. Model examples contain engineered
features, not raw transactions; real scoring requires maintaining the corresponding history.

Feature methodology follows the official [Fraud Detection Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/BaselineFeatureTransformation.html).
The local implementation uses NumPy search boundaries and prior-event cumulative sums.

## Evidence and limitations

Per dataset: `evaluation/<id>/priority_search_v2/` contains `report.json`, `protocol.json`,
`selection.json`, all trial records, `operating_points.csv`, `model_comparison.csv`, predictions,
four fitted `.joblib` pipelines, `champion.joblib` and seven PNG charts. Full training logs and
runner status are in `evaluation/priority_search_v2/`. New prepared tables are separate under
`preprocessed_df/<id>/priority_v2/`; v1 and production data/checkpoints remain unchanged.
Consolidated results are in `evaluation/priority_search_v2/RESULTS.md`, `WINNERS.csv`,
`ALL_OPERATING_POINTS.csv` and `V1_V2_COMPARISON.csv`. Each completed full run reloads all
four family checkpoints and independently reproduces test scores and calibrated thresholds.
Forest fitting uses four threads; forest prediction uses a fixed single-thread reduction order.
New scores are stored and served at 12-decimal precision to make exact threshold ties reproducible.
The first v2 runs received this numerical correction without refitting, changing selected
configurations, or choosing against test results. Their original reports/selections remain in
`report_before_stable_scoring.json` and `selection_before_stable_scoring.json`. Original candidate
selection utilities remain the audit record; corrected calibration/validation/test metrics are
reported separately. Fresh runs and resumed candidate assessment use stable scoring throughout.

The dashboard at http://127.0.0.1:8050 has a **Refresh training results** button, a validation
FPR-budget selector, and a separate **AP / recall / FPR experiment** prediction model. The
new experiment defaults to a 1% validation budget. Production is still available separately.

CLI example after Handbook completes:

```bash
python predict.py --dataset handbook --variant priority --fpr-budget 0.01 --input evaluation/handbook/priority_search_v2/example.json
python utils/verify_priority.py
```

The test periods have already been inspected; results remain exploratory. Use a genuinely
new future period for final confirmation. AP depends on prevalence and must not be compared
naively across datasets. More training does not guarantee improvement. A new validation
winner can be worse on the already-inspected test; retain this evidence rather than test-tuning.
No automatic production replacement, Git push, external compute or LLM training occurs.
