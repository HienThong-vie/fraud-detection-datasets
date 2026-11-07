"""One worker per dataset avoids collisions between legacy fraudbench packages."""
from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    dataset, variant = sys.argv[1:3]
    budget = float(sys.argv[3]) if len(sys.argv)>3 else .01
    policy = sys.argv[4] if len(sys.argv)>4 else 'point'
    capacity = float(sys.argv[5]) if len(sys.argv)>5 and sys.argv[5] != 'none' else None
    if variant != 'features' and (policy != 'point' or capacity is not None):
        raise ValueError('Review-capacity and confidence policies require the features variant')
    if variant == 'features':
        if budget not in (.005,.01,.02,.03,.05) or policy not in ('point','ucb95') or capacity not in (None,.005,.01,.02,.05):
            raise ValueError('Unsupported feature-policy scenario')
        from experiments import features_ready
        if not features_ready(dataset):
            raise ValueError('Feature experiment has not completed independent verification')
        sys.path.insert(0, str(ROOT/'training_model'))
    elif budget not in (.005,.01,.05):
        raise ValueError('FPR budget must be 0.005, 0.01 or 0.05')
    location = ROOT/'checkpoint'/dataset
    if variant == 'fitting':
        from experiments import fitting_ready
        if not fitting_ready(dataset):
            raise ValueError('Fitting experiment has not completed independent verification')
    experiment = 'feature_policy_v4' if variant == 'features' else 'fit_balance_v3' if variant == 'fitting' else 'priority_search_v2'
    path = (ROOT/'evaluation'/dataset/experiment/'champion.joblib' if variant in ('priority', 'fitting', 'features')
            else ROOT/'evaluation'/dataset/'supervised_search_v1/champion.joblib' if variant == 'supervised'
            else location/('production_model.joblib' if variant == 'production' else 'balanced_experiment/model.joblib'))
    bundle = joblib.load(path)
    records = json.load(sys.stdin)
    model = bundle.get('model',bundle.get('pipeline'))
    columns = bundle['feature_columns']
    threshold = float(bundle['thresholds'][str(budget)] if variant in ('priority', 'fitting') else bundle['threshold'])
    chosen_policy = None
    if variant == 'features':
        key = f'{policy}|{float(budget)}|{float(capacity) if capacity is not None else "none"}'
        chosen_policy = bundle['policy_thresholds'][key]
        if not chosen_policy['calibration_feasible']:
            raise ValueError('Calibration sample cannot support this confidence/capacity scenario')
        threshold = float(chosen_policy['threshold'])
    if records is None:
        response = {'features':list(columns),'model':bundle.get('selected_model',bundle.get('winner',bundle.get('model_name'))),'threshold':threshold}
        if variant in ('priority', 'fitting', 'features'):
            response['example'] = json.loads((ROOT/'evaluation'/dataset/experiment/'example.json').read_text())
            response['validation_fpr_budget'] = budget
            response['experiment'] = experiment
        if chosen_policy:
            response['threshold_policy'] = chosen_policy
            response['history_assumptions'] = bundle['history_assumptions']
        print(json.dumps(response))
        return
    frame = pd.DataFrame(records)
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError('Missing engineered model features: '+', '.join(missing))
    frame = frame[columns]
    if dataset == 'baf':
        from ai4cyber.data import replace_missing_sentinels
        frame = replace_missing_sentinels(frame)
    scores = model.predict_proba(frame)[:,1]
    if 'prediction_decimals' in bundle:
        scores = np.round(scores.astype('float64'), bundle['prediction_decimals'])
    if not np.isfinite(scores).all():
        raise ValueError('Model returned non-finite scores')
    print(json.dumps([dict(score=float(s),threshold=threshold,flagged=bool(s>=threshold),dataset=dataset,variant=variant,
                          model=bundle.get('selected_model',bundle.get('winner',bundle.get('model_name'))),
                          **(dict(threshold_policy=policy,validation_fpr_budget=budget,review_capacity=capacity)
                             if chosen_policy else {})) for s in scores]))

if __name__ == '__main__':
    main()
