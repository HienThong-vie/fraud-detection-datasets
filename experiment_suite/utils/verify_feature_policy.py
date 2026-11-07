"""Independently reload v4 models and reproduce all 50 calibration/test policy points."""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def verify(dataset):
    sys.path.insert(0, str(ROOT / 'pipelines' / dataset / 'src'))
    sys.path.insert(0, str(ROOT / 'training_model'))
    from upgrade_data import fingerprint
    from feature_policy_search import passes
    from priority_search import predict_scores
    from threshold_policy import calibrate
    from supervised_search import measure
    prepared = ROOT / 'preprocessed_df' / dataset / 'feature_policy_v4'
    frame = pd.read_pickle(prepared / 'data.pkl')
    natural = pd.read_pickle(prepared / 'natural_train.pkl')
    metadata = json.loads((prepared / 'metadata.json').read_text())
    assert fingerprint(frame) == metadata['data_sha256']
    assert fingerprint(natural) == metadata['natural_train_sha256']
    previous = pd.read_pickle(ROOT / 'preprocessed_df' / dataset / 'priority_v2/data.pkl')
    for name in ('calibration', 'selection', 'test'):
        pd.testing.assert_frame_equal(frame[frame._split == name][previous.columns].reset_index(drop=True),
                                      previous[previous._split == name].reset_index(drop=True), check_dtype=False)
    if dataset == 'paysim':
        assert int(frame[frame._split == 'train'].target.sum()) == metadata['sampling']['available_training_fraud']
    folder = ROOT / 'evaluation' / dataset / 'feature_policy_v4'
    report = json.loads((folder / 'report.json').read_text())
    assert not report['smoke']
    frozen = json.loads((folder / 'selection.json').read_text())
    assert len(report['models']) == len(frozen['models']) == 4
    assert report['winner'] == frozen['winner']
    for row in frozen['models']:
        assert 'test' not in row and 'test_policies' not in row
        assert 'test' not in row['diagnostics']
        assert passes(row, frozen['incumbent_comparison'][row['model']])
        for fold in row['forward_validation']:
            if 'skipped' not in fold:
                assert fold['train_last'] < fold['calibration_first'] <= fold['calibration_last'] < fold['evaluation_first']
    test = frame[frame._split == 'test']
    calibration = frame[frame._split == 'calibration']
    saved = pd.read_csv(folder / 'test_predictions.csv')
    assert np.array_equal(saved.target, test.target)
    for row in report['models']:
        bundle = joblib.load(folder / (row['model'] + '.joblib'))
        values = predict_scores(bundle['model'], test[bundle['feature_columns']])
        np.testing.assert_allclose(values, saved[row['model']], rtol=1e-12, atol=1e-12)
        cal_values = predict_scores(bundle['model'], calibration[bundle['feature_columns']])
        assert calibrate(calibration.target, cal_values) == row['policies'] == bundle['policy_thresholds']
        for key, policy in row['policies'].items():
            metrics = measure(test.target, values, policy['threshold'])
            metrics['alert_rate'] = float(np.mean(values >= policy['threshold']))
            for name, value in metrics.items():
                assert np.isclose(value, row['test_policies'][key][name], rtol=1e-12, atol=1e-12)
        print('VERIFIED', dataset, row['model'], '50 policies', flush=True)
    champion = joblib.load(folder / 'champion.joblib')
    assert champion['selected_model'] == report['winner']
    np.testing.assert_allclose(predict_scores(champion['model'], test[champion['feature_columns']]),
                               saved[report['winner']], rtol=1e-12, atol=1e-12)
    (folder / 'verification.json').write_text(json.dumps(dict(status='passed', dataset=dataset,
        verified_families=4, verified_policies=200, frozen_evaluation_rows=True,
        test_free_selection=True, prepared_hashes_verified=True, champion_checkpoint_verified=True), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['baf', 'ieee', 'paysim', 'elliptic'])
    args = parser.parse_args()
    verify(args.dataset)
