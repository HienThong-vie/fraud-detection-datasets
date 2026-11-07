"""Process, re-train and predict using only the prepared final-model CSV (no raw Kaggle files needed).

Prepared-data workflow for the frozen BAF model (feature_policy_v4, CatBoost):

* ``score``  (default): rebuild the 60 engineered features from the CSV, score every split with the saved
  champion, recalibrate the review thresholds on the calibration month, and print the test results at the
  5% (headline) and 1% FPR budgets plus the age-group false-positive rates. Expected test output:
  PR-AUC 0.2127; at 5%: TP 359, FP 1,767 (recall 51.1%, FPR 3.58%); at 1%: TP 153, FP 305.
* ``refit``: train a fresh CatBoost with the champion's frozen settings on the ``train`` rows only, set its
  threshold on ``calibration`` and report ``selection`` and ``test`` metrics (same steps as the original run).
* ``predict --record FILE``: score one application (JSON object or list) and return score, threshold and
  review label at the chosen budget. Example: ``examples/baf_raw_record.json``.

Usage (from ``experiment_suite/``)::

    python training_model/reproduce_from_prepared.py score
    python training_model/reproduce_from_prepared.py refit
    python training_model/reproduce_from_prepared.py predict --record examples/baf_raw_record.json

A score supports analyst review only; it never approves, rejects or blocks an application.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipelines' / 'baf' / 'src'))
sys.path.insert(0, str(ROOT / 'training_model'))

from priority_search import predict_scores  # noqa: E402  (scores rounded exactly as in the original run)
from threshold_policy import calibrate  # noqa: E402     (same threshold rule as the original run)
from upgrade_support import baf_features  # noqa: E402    (builds the 60 candidate features)

CHAMPION = ROOT / 'evaluation' / 'baf' / 'feature_policy_v4' / 'champion.joblib'
DATA = ROOT / 'final_model_data' / 'Base_v4_final_model_data.csv'
DICTIONARY = ROOT / 'final_model_data' / 'Base_v4_final_model_data_dictionary.csv'
BUDGETS = ('0.05', '0.01')          # headline budget first


def load_prepared(path):
    """Verify the CSV checksum and schema before restoring its full-precision numeric values."""
    path = Path(path)
    dictionary = pd.read_csv(path.with_name(path.stem+'_dictionary.csv'))
    manifest = json.loads(path.with_name(path.stem+'_manifest.json').read_text())
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != manifest['csv_sha256']:
            raise ValueError('Prepared CSV checksum differs from its manifest')
    types = dictionary.set_index('column')['type'].to_dict()
    frame = pd.read_csv(path, dtype=types, float_precision='round_trip', keep_default_na=False, na_values=[''])
    if list(frame.columns) != dictionary.column.tolist() or len(frame) != manifest['rows']:
        raise ValueError('Prepared data rows or columns differ from the manifest')
    inputs = dictionary[dictionary.model_input == 'yes'].column.tolist()
    if set(inputs) & {'customer_age','income','employment_status','fraud_bool','month','_split'}:
        raise ValueError('Protected or target/time fields must not be model inputs')
    if set(frame._split.unique()) != set(manifest['splits']):
        raise ValueError('Unexpected prepared partition')
    for name, info in manifest['splits'].items():
        part = frame[frame._split == name]
        if len(part) != info['rows'] or int(part.fraud_bool.sum()) != info['fraud']:
            raise ValueError('Prepared split counts differ: '+name)
    if frame.intended_balcon_amount.lt(0).any():
        raise ValueError('Negative intended transfer amounts must be missing in prepared data')
    return frame, inputs


def features(frame, inputs):
    """Further processing: missing-value flags, log transforms, ratios and interactions (60 columns)."""
    return baf_features(frame[inputs])


def confusion(y, scores, threshold):
    """Confusion counts and the report's metrics for one threshold (flag when score >= threshold)."""
    flag = scores >= threshold
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    return dict(threshold=float(threshold), tp=tp, fp=fp, fn=fn, tn=tn, recall=recall,
                fpr=fp / (fp + tn), precision=precision, accuracy=(tp+tn)/len(y),
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0)


def evaluate(model, frame, inputs, columns, label):
    """Calibrate thresholds on month 4 and report selection and test results for one fitted model."""
    x = features(frame, inputs)
    scores = {s: predict_scores(model, x[frame._split == s][columns]) for s in ('calibration', 'selection', 'test')}
    targets = {s: frame[frame._split == s].fraud_bool.to_numpy() for s in scores}
    policies = calibrate(pd.Series(targets['calibration']), scores['calibration'])
    out = dict(model=label, policy_thresholds=policies)
    for split in ('selection', 'test'):
        y, s = targets[split], scores[split]
        out[split] = dict(pr_auc=float(average_precision_score(y, s)), roc_auc=float(roc_auc_score(y, s)),
                          **{f'budget_{b}': confusion(y, s, policies[f'point|{b}|none']['threshold']) for b in BUDGETS})
    # Age-group false-positive rates on test (age is joined only here; it is never a model input).
    test = frame[frame._split == 'test']
    legit, older = targets['test'] == 0, test.customer_age.to_numpy() >= 50
    out['test_fpr_by_age'] = {}
    for b in BUDGETS:
        flag = scores['test'] >= policies[f'point|{b}|none']['threshold']
        a, n1 = int((flag & legit & older).sum()), int((legit & older).sum())
        c, n2 = int((flag & legit & ~older).sum()), int((legit & ~older).sum())
        out['test_fpr_by_age'][f'budget_{b}'] = dict(fp_50plus=a, legit_50plus=n1, fp_under50=c, legit_under50=n2,
                                                     ratio=round((a / n1) / (c / n2), 3))
    return out


def refit(bundle, frame, inputs):
    """Train a new pipeline with the champion's frozen settings (CatBoost depth 4, 619 rounds) on train only."""
    model = clone(bundle['model'])                      # same steps and hyper-parameters, unfitted
    train = frame[frame._split == 'train']
    x = features(train, inputs)[bundle['feature_columns']]
    transform, classifier = model.steps[0][1], model.steps[-1][1]
    xt = transform.fit_transform(x, train.fraud_bool)   # training-only feature screen and encoding
    cats = [c for c in xt if not pd.api.types.is_numeric_dtype(xt[c])]
    classifier.fit(xt, train.fraud_bool, sample_weight=np.ones(len(train)), cat_features=cats)
    return model


def predict(bundle, record_path, budget):
    """Score one or more raw application records (26 inputs; missing values as -1 codes or null)."""
    from ai4cyber.data import replace_missing_sentinels
    records = json.loads(Path(record_path).read_text())
    frame = pd.DataFrame(records if isinstance(records, list) else [records])
    dictionary = pd.read_csv(DICTIONARY)
    inputs = dictionary[dictionary.model_input == 'yes'].column.tolist()
    missing = set(inputs) - set(frame.columns)
    if missing:
        raise ValueError('Record is missing required inputs: '+', '.join(sorted(missing)))
    for column, kind in dictionary.set_index('column').loc[inputs, 'type'].items():
        if kind not in ('object', 'str', 'string'):     # numbers stay numeric even when a value is null
            frame[column] = pd.to_numeric(frame[column], errors='coerce').astype('float64')
    cleaned = replace_missing_sentinels(frame[inputs])
    scores = predict_scores(bundle['model'], baf_features(cleaned[inputs])[bundle['feature_columns']])
    threshold = bundle['thresholds'][budget]
    return [dict(model_score=round(float(s), 6), threshold=round(threshold, 6), fpr_budget=budget,
                 review_label='review' if s >= threshold else 'no review',
                 notice='Decision support only: an analyst decides; the score is not proof of fraud.') for s in scores]


def main():
    parser = argparse.ArgumentParser(description='Reproduce the final BAF model from the prepared CSV.')
    parser.add_argument('mode', nargs='?', default='score', choices=['score', 'refit', 'predict'])
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--record', type=Path, help='JSON record(s) for predict mode')
    parser.add_argument('--model', type=Path, default=CHAMPION, help='Trusted saved bundle to score')
    parser.add_argument('--output', type=Path, default=ROOT/'checkpoint/baf/prepared_refit',
                        help='New directory for refit bundle and metrics; never overwrites a frozen experiment')
    parser.add_argument('--result', type=Path, help='Write aggregate metrics to a new JSON file')
    parser.add_argument('--budget', default='0.05', choices=['0.005', '0.01', '0.02', '0.03', '0.05'])
    args = parser.parse_args()
    if args.result and args.result.exists():
        raise FileExistsError('Preserving existing metrics; choose a new --result file')
    bundle = joblib.load(args.model)
    if args.mode == 'predict':
        if args.record is None:
            parser.error('predict requires --record')
        print(json.dumps(predict(bundle, args.record, args.budget), indent=2))
        return
    frame, inputs = load_prepared(args.data)
    if args.mode == 'refit' and args.output.exists():
        raise FileExistsError('Preserving existing refit; choose a new --output directory')
    model = bundle['model'] if args.mode == 'score' else refit(bundle, frame, inputs)
    label = 'saved champion' if args.mode == 'score' else 'refitted with champion settings'
    result = evaluate(model, frame, inputs, bundle['feature_columns'], label)
    if args.result:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2))
    if args.mode == 'refit':
        args.output.mkdir(parents=True, exist_ok=False)
        saved = dict(bundle, model=model, policy_thresholds=result['policy_thresholds'],
                     thresholds={str(b):result['policy_thresholds'][f'point|{b}|none']['threshold']
                                 for b in (.005,.01,.02,.03,.05)}, refitted_from_csv=True)
        saved['threshold'] = saved['thresholds']['0.05']
        joblib.dump(saved, args.output/'refit.joblib')
        (args.output/'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
