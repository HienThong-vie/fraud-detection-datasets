"""Reload each fitted family and independently reproduce its saved test scores."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def verify(dataset, experiment='priority_search_v2'):
    sys.path.insert(0, str(ROOT / 'pipelines' / dataset / 'src'))
    sys.path.insert(0, str(ROOT / 'training_model'))
    folder = ROOT / 'evaluation' / dataset / experiment
    report = json.loads((folder / 'report.json').read_text())
    prepared = pd.read_pickle(ROOT / 'preprocessed_df' / dataset / 'priority_v2/data.pkl')
    test = prepared[prepared._split == 'test']
    calibration = prepared[prepared._split == 'calibration']
    saved = pd.read_csv(folder / 'test_predictions.csv')
    assert np.array_equal(test.target, saved.target)
    from priority_search import fpr_threshold
    from supervised_search import measure
    for row in report['models']:
        family = row['model']
        bundle = joblib.load(folder / (family + '.joblib'))
        columns = bundle['feature_columns']
        scores = bundle['model'].predict_proba(test[columns])[:, 1]
        if 'prediction_decimals' in bundle:
            scores = np.round(scores.astype('float64'), bundle['prediction_decimals'])
        # Pandas emits float32 scores at float32 precision; reload uses float64.
        tolerance = 2 * np.finfo(scores.dtype).eps
        np.testing.assert_allclose(scores, saved[family], rtol=tolerance, atol=tolerance)
        cal = bundle['model'].predict_proba(calibration[columns])[:, 1]
        if 'prediction_decimals' in bundle:
            cal = np.round(cal.astype('float64'), bundle['prediction_decimals'])
        for budget, threshold in row['thresholds'].items():
            # Parallel tree reductions can differ by a few floating-point ULPs on reload.
            assert np.isclose(threshold, fpr_threshold(calibration.target, cal, float(budget)),
                              rtol=1e-12, atol=1e-12)
            metrics = measure(test.target, scores, threshold)
            for key, value in row['test_operating_points'][budget].items():
                assert np.isclose(value, metrics[key], rtol=1e-12, atol=1e-12), (dataset, family, key)
        print('VERIFIED', dataset, family, flush=True)
    if experiment == 'fit_balance_v3':
        from fit_balance_search import admissible
        assert report['data_sha256'] == hashlib.sha256(
            pd.util.hash_pandas_object(prepared, index=True).values.tobytes()).hexdigest()
        frozen = json.loads((folder / 'selection.json').read_text())
        assert frozen['winner'] == report['winner']
        assert len(frozen['models']) == len(report['models']) == 4
        assert all('test' not in row and 'test' not in row['diagnostics'] for row in frozen['models'])
        for row in frozen['models']:
            assert admissible(row, frozen['incumbent_comparison'][row['model']])
        champion = next(r for r in frozen['models'] if r['model'] == frozen['winner'])
        old = json.loads((ROOT / 'evaluation' / dataset / 'priority_search_v2/report.json').read_text())
        assert admissible(champion, frozen['incumbent_comparison'][old['winner']])
        for row in report['models']:
            for budget, metrics in row['calibration_operating_points'].items():
                assert metrics['fpr'] <= float(budget) + 1e-12
        bundle = joblib.load(folder / 'champion.joblib')
        assert bundle['selected_model'] == report['winner']
        champion_scores = np.round(np.asarray(bundle['model'].predict_proba(
            test[bundle['feature_columns']])[:, 1], dtype='float64'), 12)
        np.testing.assert_allclose(champion_scores, saved[report['winner']], rtol=1e-12, atol=1e-12)
        assert bundle['thresholds'] == champion['thresholds']
        (folder / 'verification.json').write_text(json.dumps(dict(
            status='passed', dataset=dataset, families=4, frozen_validation_only_selection=True,
            calibration_budgets_verified=True, checkpoint_scores_reproduced=True,
            prepared_data_hash_verified=True, champion_checkpoint_verified=True), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--experiment', choices=['priority_search_v2', 'fit_balance_v3'], default='priority_search_v2')
    args = parser.parse_args()
    if args.dataset:
        verify(args.dataset, args.experiment)
    else:
        registry = json.loads((ROOT / 'configs/datasets.json').read_text())
        for dataset in registry:
            if (ROOT / 'evaluation' / dataset / args.experiment / 'report.json').exists():
                subprocess.run([sys.executable, __file__, '--dataset', dataset, '--experiment', args.experiment],
                               env=dict(os.environ, MPLCONFIGDIR=str(ROOT / '.cache/matplotlib')), check=True)
