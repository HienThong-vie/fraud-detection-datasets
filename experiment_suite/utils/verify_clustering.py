"""Reproduce cluster assignments and post-hoc tables from the saved pipeline."""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training_model'))
from clustering import choose_candidate, feature_columns, interpret
from priority_data import prepare


def verify(dataset):
    folder = ROOT / 'evaluation' / dataset / 'clustering_v1'
    report = json.loads((folder / 'report.json').read_text())
    bundle = joblib.load(folder / 'clustering.joblib')
    frame, metadata = prepare(dataset)
    columns = feature_columns(dataset, metadata['feature_columns'])
    assert columns == bundle['feature_columns'] == report['feature_columns']
    assert not report['labels_used_for_fit_or_selection'] and not report['validation_test_refit']
    candidates = pd.read_csv(folder / 'k_comparison.csv')
    selected, relaxed = choose_candidate(candidates.to_dict('records'))
    assert selected['k'] == bundle['selected_k'] == report['selected_k']
    assert relaxed == report['fit_info']['quality_constraints_relaxed']
    assignments = pd.Series(index=frame.index, dtype='int32')
    with threadpool_limits(limits=4):
        for start in range(0, len(frame), 25000):
            part = frame.iloc[start:start + 25000]
            assignments.loc[part.index] = bundle['pipeline'].predict(part[columns])
    actual = interpret(frame, assignments.astype('int32'))
    recorded = pd.read_csv(folder / 'cluster_metrics.csv')
    pd.testing.assert_frame_equal(actual, recorded, check_dtype=False, atol=1e-10)
    rng = np.random.default_rng(42)
    train = frame[frame._split == 'train']
    indices = np.sort(rng.choice(len(train), min(len(train), 80000), replace=False))
    numeric = bundle['pipeline'].named_steps['preprocess'].transformers_[0][2]
    imputer = bundle['pipeline'].named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute']
    raw = train.iloc[indices][numeric]
    expected = raw.median().fillna(0).to_numpy()
    np.testing.assert_allclose(imputer.statistics_, expected, rtol=1e-7, atol=1e-8)
    for filename in ('silhouette_stability.png', 'cluster_sizes.png', 'feature_profiles.png', 'fraud_rates.png', 'README.md'):
        assert (folder / filename).stat().st_size > 100
    marker = dict(status='passed', dataset=dataset, selected_k=bundle['selected_k'],
                  saved_assignments_reproduced=True, interpretation_tables_reproduced=True,
                  training_only_numeric_imputer_verified=True, k_selection_reproduced=True,
                  classifier_replacement=False)
    (folder / 'verification.json').write_text(json.dumps(marker, indent=2))
    print('VERIFIED CLUSTERING', dataset, bundle['selected_k'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    verify(parser.parse_args().dataset)
