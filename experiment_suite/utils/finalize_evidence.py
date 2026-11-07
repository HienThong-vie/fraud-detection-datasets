"""Freeze validation-selected champions and export reproducible BAF evidence."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training_model'))
VERSIONS = dict(baf='feature_policy_v4', ieee='feature_policy_v4', paysim='feature_policy_v4',
                elliptic='feature_policy_v4', sparkov='fit_balance_v3', ulb='fit_balance_v3',
                handbook='priority_search_v2', banksim='priority_search_v2')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def freeze():
    registry = dict(selection='Existing chronological-validation winners only; no test-based re-selection.',
                    headline_budget=.05, secondary_budget=.01, policy='point', review_capacity=None,
                    test_interpretation='Previously inspected test periods; exploratory, not fresh holdouts.', datasets={})
    for dataset, version in VERSIONS.items():
        directory = ROOT / 'evaluation' / dataset / version
        report = json.loads((directory / 'report.json').read_text())
        champion = next(r for r in report['models'] if r['model'] == report['winner'])
        model = directory / 'champion.joblib'
        points = {}
        for budget in (.05, .01):
            key = str(budget)
            metrics = champion['test_policies'][f'point|{key}|none'] if version == 'feature_policy_v4' else champion['test_operating_points'][key]
            points[key] = dict(threshold=champion['thresholds'][key], test_metrics=metrics)
        registry['datasets'][dataset] = dict(model=report['winner'], experiment=version, variant=champion['variant'],
            checkpoint=str(model.relative_to(ROOT)), checkpoint_sha256=sha(model), operating_points=points,
            validation_ap=champion['validation']['pr_auc'],
            selection_score=champion.get('fit_balance_score', champion.get('selection_utility')),
            score_name='fit_balance_score' if 'fit_balance_score' in champion else 'selection_utility',
            offline_only=dataset == 'elliptic')
    path = ROOT / 'configs/final_models.json'
    if path.exists():
        assert json.loads(path.read_text()) == registry, 'Frozen registry changed; create a new version explicitly'
    else:
        path.write_text(json.dumps(registry, indent=2))
    return registry


def baf_evidence():
    sys.path.insert(0, str(ROOT / 'pipelines/baf/src'))
    from ai4cyber.config import TrainingConfig
    from ai4cyber.data import load_baf_csv, make_temporal_data
    cfg = TrainingConfig(**json.loads((ROOT / 'pipelines/baf/configs/training.json').read_text()))
    cfg = replace(cfg, train_months=[0, 1, 2, 3], validation_months=[4, 5], max_train_rows=500000, max_validation_rows=None)
    raw = load_baf_csv(ROOT / 'dataset/baf/Base.csv', cfg)
    test = make_temporal_data(raw, cfg).test.sort_values('month', kind='stable').reset_index(drop=True)
    frame = pd.read_pickle(ROOT / 'preprocessed_df/baf/feature_policy_v4/data.pkl')
    test_frame = frame[frame._split == 'test'].reset_index(drop=True)
    assert np.array_equal(test.fraud_bool, test_frame.target)
    np.testing.assert_allclose(test.name_email_similarity, test_frame.name_email_similarity)
    output = ROOT / 'evaluation/baf/final_evidence'
    output.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(ROOT / 'evaluation/baf/feature_policy_v4/test_predictions.csv')
    assert np.array_equal(scores.target, test.fraud_bool)
    registry = freeze()
    rows = []
    for budget in (.05, .01):
        threshold = registry['datasets']['baf']['operating_points'][str(budget)]['threshold']
        flags = scores.catboost >= threshold
        for label, mask in [('age>=50', test.customer_age >= 50), ('age<50', test.customer_age < 50)]:
            legitimate = mask & (test.fraud_bool == 0)
            fp = int((flags & legitimate).sum())
            rows.append(dict(fpr_budget=budget, threshold=threshold, age_group=label,
                             legitimate_rows=int(legitimate.sum()), false_positives=fp,
                             fpr=fp / int(legitimate.sum())))
    pd.DataFrame(rows).to_csv(output / 'fairness_age.csv', index=False)
    (output / 'fairness_age.json').write_text(json.dumps(dict(groups=rows, test_rows=len(test),
        grouping='customer_age >=50 versus <50; source age bands', alignment_verified=True,
        checkpoint_sha256=registry['datasets']['baf']['checkpoint_sha256'],
        interpretation='Observed group disparity; excluded age is not proof of fairness. No causal claim.'), indent=2))
    export = ROOT / 'exports/baf_final_v4'
    export.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((ROOT / 'preprocessed_df/baf/feature_policy_v4/metadata.json').read_text())
    bundle = joblib.load(ROOT / registry['datasets']['baf']['checkpoint'])
    retained = bundle['model'].steps[0][1].named_steps['screen'].columns_
    columns = metadata['feature_columns']
    data = frame[columns].copy()
    data['fraud_bool'] = frame.target
    data['month'] = frame._event_time.astype('int8')
    data['_split'] = frame._split
    destination = export / 'Base_final_model_data.csv'
    old_manifest = export / 'Base_final_model_data_manifest.json'
    rewrite = not destination.exists()
    if destination.exists() and old_manifest.exists():
        old = json.loads(old_manifest.read_text())
        if old['float_format'] != '%.17g':
            assert sha(destination) == old['sha256'], 'Export was edited; preserve it before regenerating'
            rewrite = True
    if rewrite:
        data.to_csv(destination, index=False, float_format='%.17g', chunksize=10000)
    dictionary = pd.read_csv(ROOT.parent / 'ai4cyber-fraud/dataset/Base_final_model_data_dictionary.csv')
    meanings = dict(zip(dictionary.column, dictionary.meaning))
    definitions = dict(fraud_bool='Observed fraud label; never a model input.', month='Original event month; split/audit only.',
        _split='Temporal role: train, calibration, selection or test.', missing_numeric_count='Count of missing original numeric inputs.',
        device_emails_foreign='Device distinct-email count multiplied by foreign-request flag.',
        phone_both_invalid='1 when both original phone-validity flags are zero.',
        payment_device='Original payment type joined with device OS.')
    definitions.update(meanings)
    dictionary_rows = []
    for column in data:
        meaning = definitions.get(column)
        if meaning is None:
            if column.endswith('_missing'):
                meaning = 'Missingness indicator for ' + column.removesuffix('_missing') + '; computed after missing-code cleaning.'
            elif column.endswith('_log1p'):
                meaning = 'log(1 + max(0, ' + column.removesuffix('_log1p') + ')).'
            elif '_to_' in column:
                a, b = column.split('_to_', 1)
                meaning = f'{a} divided by max(1, {b}), clipped to [0,10000].'
            else:
                raise ValueError('Missing dictionary meaning: ' + column)
        dictionary_rows.append(dict(column=column, type=str(data[column].dtype), meaning=meaning,
            pipeline_input='yes' if column in columns else 'no',
            model_input='yes' if column in retained else 'no',
            note='Screened out before CatBoost (constant/redundant training feature).' if column in columns and column not in retained else ''))
    pd.DataFrame(dictionary_rows).to_csv(export / 'Base_final_model_data_dictionary.csv', index=False)
    actual_rows = sum(len(chunk) for chunk in pd.read_csv(destination, chunksize=50000, usecols=['fraud_bool']))
    assert actual_rows == len(data)
    check = pd.read_csv(destination, nrows=128, float_precision='round_trip')
    np.testing.assert_allclose(check[columns].select_dtypes(include=np.number), data.head(128)[columns].select_dtypes(include=np.number), rtol=1e-10, atol=1e-9, equal_nan=True)
    manifest = dict(rows=len(data), columns=len(data.columns), candidate_inputs=len(columns), retained_inputs=len(retained),
        split_rows=data.groupby('_split').size().to_dict(), bytes=destination.stat().st_size,
        decimal_MB=destination.stat().st_size / 1e6, binary_MiB=destination.stat().st_size / 2**20,
        sha256=sha(destination), float_format='%.17g', null_representation='empty CSV field',
        sample_weight='Omitted: BAF weights are all one.', licence='CC BY-NC-SA 4.0',
        source='https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022',
        checkpoint_sha256=registry['datasets']['baf']['checkpoint_sha256'],
        upload_limit='Not supplied; compare this size and the final ZIP size with the actual portal limit.')
    (export / 'Base_final_model_data_manifest.json').write_text(json.dumps(manifest, indent=2))
    (output / 'data_export_manifest.json').write_text(json.dumps(manifest, indent=2))
    (output / 'DATA_EXPORT.md').write_text(f"# BAF processed-data export\n\n{len(data):,} rows, {len(data.columns)} columns, {manifest['decimal_MB']:.2f} MB ({manifest['binary_MiB']:.2f} MiB), plain CSV.\n\nThe CSV stays local under `exports/baf_final_v4/`; only the dictionary and manifest are published. It contains 60 pipeline inputs, 44 retained CatBoost inputs, label, month and split. Raw/derived BAF data remain subject to CC BY-NC-SA 4.0.\n")
    print('BAF EVIDENCE', manifest, flush=True)


if __name__ == '__main__':
    freeze()
    baf_evidence()
