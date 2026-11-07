"""Export the exact rows used by the final BAF model (experiment feature_policy_v4) as one CSV.

The exported dataset contains only the rows used by the final model. This script rebuilds
the four frozen partitions from the raw Kaggle ``Base.csv`` with the same preparation code that trained the
model (``priority_data.prepare`` and ``upgrade_data.prepare``), then writes:

* ``Base_v4_final_model_data.csv``: one row per application, a ``_split`` column (train / calibration /
  selection / test), the 26 cleaned model inputs (BAF missing-value codes already converted to empty cells),
  the label ``fraud_bool``, the ``month`` and the three protected fields. The protected fields are kept only for
  the age-group fairness check; they are never model inputs.
* ``Base_v4_final_model_data_dictionary.csv``: meaning, dtype and model-input flag for every column.
* ``Base_v4_final_model_data_manifest.json``: row and fraud counts per split, SHA-256 of the CSV, seeds.

Before writing, the script checks that (1) the row and fraud counts equal ``report.json`` and (2) the 60
engineered features rebuilt from the CSV columns equal the frozen v4 training frame exactly.
The 34 engineered features are not stored: ``reproduce_from_prepared.py`` rebuilds them with ``baf_features``.

Usage (from ``experiment_suite/``, with the raw ``Base.csv`` in ``dataset/baf/``)::

    python training_model/export_final_model_data.py

"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipelines' / 'baf' / 'src'))
sys.path.insert(0, str(ROOT / 'training_model'))

from priority_data import bound, prepare as prepare_partitions  # noqa: E402  (same sampling as training)
import upgrade_data  # noqa: E402  (v4 feature frame used to train and test the final model)
from upgrade_support import baf_features  # noqa: E402

PROTECTED = ['customer_age', 'income', 'employment_status']
EXPERIMENT = ROOT / 'evaluation' / 'baf' / 'feature_policy_v4'
OLD_DICTIONARY = ROOT / 'final_model_data' / 'Base_v4_final_model_data_dictionary.csv'
SPLIT_MEANING = 'Frozen v4 partition: train (months 0-3), calibration (month 4), selection (month 5), test (months 6-7).'


def raw_partitions(raw_csv):
    """Rebuild each partition with every raw column, in the same row order as the frozen v4 frame.

    The order matters: ``priority_data.prepare`` sorts every partition by month (stable sort), so applying
    the same sampling and the same sort here lets the protected fields be attached row by row.
    """
    from ai4cyber.config import TrainingConfig
    from ai4cyber.data import load_baf_csv, make_temporal_data, replace_missing_sentinels
    config = json.loads((ROOT / 'pipelines' / 'baf' / 'configs' / 'training.json').read_text())
    # Same overrides as priority_data.prepare('baf'): train months 0-3 capped at 500,000 rows,
    # months 4 and 5 kept whole here and sampled below with seeds 46 and 47.
    config.update(train_months=[0, 1, 2, 3], validation_months=[4, 5], max_train_rows=500000,
                  max_validation_rows=None)
    cfg = TrainingConfig(**config)
    splits = make_temporal_data(load_baf_csv(raw_csv, cfg), cfg)
    parts = {'train': splits.train, 'test': splits.test}
    for month, name in [(4, 'calibration'), (5, 'selection')]:
        parts[name] = bound(splits.validation[splits.validation.month == month], 'fraud_bool', 50000, 42 + month)
    columns = splits.feature_columns
    out = {}
    for name, data in parts.items():
        data = data.sort_values('month', kind='stable').reset_index(drop=True)
        cleaned = replace_missing_sentinels(data[columns])          # -1 codes and negative balance -> missing
        out[name] = pd.concat([cleaned, data[['fraud_bool', 'month', *PROTECTED]]], axis=1)
    return out, columns


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--raw', type=Path, default=ROOT / 'dataset' / 'baf' / 'Base.csv')
    parser.add_argument('--out', type=Path, default=ROOT / 'final_model_data')
    args = parser.parse_args()
    if (args.out / 'Base_v4_final_model_data.csv').exists():
        raise FileExistsError('Preserving existing export; choose a new --out directory.')

    # 1. Frozen v4 frame exactly as used for training, calibration, selection and test.
    prepare_partitions('baf')
    frozen, _, meta = upgrade_data.prepare('baf')
    report = json.loads((EXPERIMENT / 'report.json').read_text())
    assert meta['periods'] == report['preparation']['periods'], 'partitions differ from the published experiment'

    # 2. Raw rows of the same partitions, with month and protected fields attached.
    parts, inputs = raw_partitions(args.raw)
    order = ['train', 'calibration', 'selection', 'test']
    export = pd.concat([parts[name].assign(_split=name) for name in order], ignore_index=True)
    export = export[['_split', *inputs, 'fraud_bool', 'month', *PROTECTED]]

    # 3. Check: labels, counts and the 60 engineered features must match the frozen frame row by row.
    for name in order:
        mine, ref = export[export._split == name].reset_index(drop=True), frozen[frozen._split == name].reset_index(drop=True)
        assert len(mine) == report['preparation']['periods'][name]['rows']
        assert int(mine.fraud_bool.sum()) == report['preparation']['periods'][name]['fraud']
        assert np.array_equal(mine.fraud_bool.to_numpy(), ref.target.to_numpy())
        rebuilt = baf_features(mine[inputs])
        pd.testing.assert_frame_equal(rebuilt[meta['feature_columns']], ref[meta['feature_columns']], check_dtype=False, check_column_type=False, check_exact=True)

    # 4. Write the CSV, then read it back and repeat the feature check (proves the CSV round-trips exactly).
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / 'Base_v4_final_model_data.csv'
    export.to_csv(csv_path, index=False, float_format='%.17g')
    dtypes = {c: (str(t) if pd.api.types.is_numeric_dtype(t) else 'object') for c, t in export.dtypes.items()}
    back = pd.read_csv(csv_path, dtype=dtypes, float_precision='round_trip', keep_default_na=False, na_values=[''])
    pd.testing.assert_frame_equal(export, back, check_exact=True, check_dtype=False)
    for name in order:
        restored = back[back._split == name].reset_index(drop=True)
        reference = frozen[frozen._split == name].reset_index(drop=True)
        pd.testing.assert_frame_equal(baf_features(restored[inputs])[meta['feature_columns']],
                                      reference[meta['feature_columns']], check_dtype=False,
                                      check_column_type=False, check_exact=True)

    # 5. Data dictionary and manifest.
    old = pd.read_csv(OLD_DICTIONARY).set_index('column')['meaning'] if OLD_DICTIONARY.exists() else pd.Series(dtype=str)
    rows = []
    for column in export.columns:
        meaning = SPLIT_MEANING if column == '_split' else old.get(column, '')
        if column in PROTECTED:
            if 'fairness check only' not in meaning:
                meaning += ' Protected: fairness check only, never a model input.'
        if not str(meaning).strip():
            raise ValueError('Missing data dictionary meaning for ' + column)
        rows.append(dict(column=column, type=dtypes[column], meaning=meaning.strip(),
                         model_input='yes' if column in inputs else 'no'))
    pd.DataFrame(rows).to_csv(args.out / 'Base_v4_final_model_data_dictionary.csv', index=False)
    manifest = dict(
        source='Kaggle sgpjesus/bank-account-fraud-dataset-neurips-2022, version 2, Base.csv (CC BY-NC-SA 4.0)',
        experiment='feature_policy_v4', champion='catboost features_setting0_windowNone',
        rows=len(export), columns=len(export.columns),
        splits={n: dict(rows=int((export._split == n).sum()), fraud=int(export[export._split == n].fraud_bool.sum()),
                        months=sorted(int(m) for m in export[export._split == n].month.unique())) for n in order},
        sampling_seeds=dict(train=42, calibration=46, selection=47, test=44),
        engineered_features='60 candidate features rebuilt by upgrade_support.baf_features; the champion keeps 44',
        csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        csv_bytes=csv_path.stat().st_size,
        raw_source_sha256=hashlib.file_digest(args.raw.open('rb'), 'sha256').hexdigest(),
        original_input_columns=inputs,
        engineered_feature_columns=meta['feature_columns'],
        csv_dtypes=dtypes,
        frozen_data_sha256=meta['data_sha256'],
        float_format='%.17g, round_trip reader',
        checks='counts equal report.json; engineered features equal the frozen v4 frame for every split; CSV round-trip verified')
    (args.out / 'Base_v4_final_model_data_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
