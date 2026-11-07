"""Reload the four BAF v4 bundles and independently reproduce policies and metrics from the CSV."""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'training_model'))
from reproduce_from_prepared import DATA, load_prepared, features
from priority_search import predict_scores
from threshold_policy import calibrate
from supervised_search import measure


def verify(path):
    """Assert all 200 calibration policies and every recorded test metric before returning a report."""
    frame, inputs = load_prepared(path)
    folder = ROOT/'evaluation/baf/feature_policy_v4'
    report = json.loads((folder/'report.json').read_text())
    x = features(frame, inputs)
    test = frame._split == 'test'
    calibration = frame._split == 'calibration'
    periods_match = all(len(frame[frame._split==split])==info['rows'] and
                        int(frame.loc[frame._split==split,'fraud_bool'].sum())==info['fraud']
                        for split, info in report['preparation']['periods'].items())
    assert periods_match
    out = dict(status='passed', periods_match=True, data_source='prepared CSV only', families={})
    for row in report['models']:
        bundle = joblib.load(folder/(row['model']+'.joblib'))
        columns = bundle['feature_columns']
        scores = predict_scores(bundle['model'], x.loc[test, columns])
        cal_scores = predict_scores(bundle['model'], x.loc[calibration, columns])
        policies = calibrate(frame.loc[calibration,'fraud_bool'], cal_scores)
        policies_match = policies == row['policies'] == bundle['policy_thresholds']
        assert policies_match, row['model']
        worst = 0.
        for key, policy in policies.items():
            metrics = measure(frame.loc[test,'fraud_bool'], scores, policy['threshold'])
            for name, value in metrics.items():
                difference = abs(float(value)-row['test_policies'][key][name])
                worst = max(worst, difference)
                assert difference <= 1e-12, (row['model'],key,name,difference)
        diagnostics = measure(frame.loc[test,'fraud_bool'], scores, bundle['thresholds']['0.05'])
        out['families'][row['model']] = dict(test_pr_auc=diagnostics['pr_auc'],
            reported_test_pr_auc=row['test']['pr_auc'],test_roc_auc=diagnostics['roc_auc'],
            policies_recalibrated_identically=policies_match, policies_checked=len(policies),
            largest_metric_difference=worst)
    champion = joblib.load(folder/'champion.joblib')
    scores = predict_scores(champion['model'], x.loc[test,champion['feature_columns']])
    legit = frame.loc[test,'fraud_bool'].to_numpy() == 0
    older = frame.loc[test,'customer_age'].to_numpy() >= 50
    out['fairness'] = {}
    for budget in ('0.05','0.01'):
        flags = scores >= champion['thresholds'][budget]
        out['fairness'][budget] = dict(fp_50plus=int((flags&legit&older).sum()),
            legit_50plus=int((legit&older).sum()),fp_under50=int((flags&legit&~older).sum()),
            legit_under50=int((legit&~older).sum()))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=ROOT/'evaluation/baf/independent_check_v4_result.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserving existing verification; choose a new --output file')
    result = verify(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
