"""Versioned raw-data preparation with event-time metadata kept out of model inputs."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CAPS = dict(baf=500000, ieee=300000, paysim=300000, handbook=300000,
            sparkov=400000, banksim=300000, ulb=None, elliptic=None)


def project_directory(dataset):
    return ROOT / 'pipelines' / dataset if (ROOT / 'pipelines').exists() else ROOT


def evaluation_directory(dataset):
    return ROOT / 'evaluation' / dataset if (ROOT / 'pipelines').exists() else ROOT / 'outputs'


def bound(frame, target, maximum, seed):
    if maximum is None or len(frame) <= maximum:
        return frame.copy()
    chosen, _ = train_test_split(np.arange(len(frame)), train_size=maximum,
                                 stratify=frame[target], random_state=seed)
    return frame.iloc[np.sort(chosen)].copy()


def rolling_history(raw):
    """Only strictly earlier transactions; fraud outcomes become known after seven days."""
    features = pd.DataFrame(index=raw.index)
    seconds = raw.TX_TIME_SECONDS.to_numpy(dtype=float)
    if np.any(seconds[1:] < seconds[:-1]):
        raise ValueError('Sort transaction history by event time before rolling feature preparation')
    amounts = raw.TX_AMOUNT.to_numpy(dtype=float)
    labels = raw.TX_FRAUD.to_numpy(dtype=float)
    for entity, prefix in [('CUSTOMER_ID', 'customer'), ('TERMINAL_ID', 'terminal')]:
        counts = {w: np.zeros(len(raw)) for w in (1, 7, 30)}
        means = {w: np.zeros(len(raw)) for w in (1, 7, 30)}
        risks = {w: np.zeros(len(raw)) for w in (1, 7, 30)}
        for indices in raw.groupby(entity, sort=False).indices.values():
            t = seconds[indices]
            right = np.searchsorted(t, t, side='left')
            amount_sum = np.r_[0., amounts[indices].cumsum()]
            fraud_sum = np.r_[0., labels[indices].cumsum()]
            known_right = np.searchsorted(t, t - 7 * 86400, side='left')
            for window in counts:
                left = np.searchsorted(t, t - window * 86400, side='left')
                n = right - left
                counts[window][indices] = n
                means[window][indices] = (amount_sum[right] - amount_sum[left]) / np.maximum(n, 1)
                if prefix == 'terminal':
                    known_left = np.searchsorted(t, t - (7 + window) * 86400, side='left')
                    risks[window][indices] = ((fraud_sum[known_right] - fraud_sum[known_left])
                                              / np.maximum(known_right - known_left, 1))
        for window in counts:
            features[f'{prefix}_count_{window}d'] = counts[window].astype('float32')
            features[f'{prefix}_mean_amount_{window}d'] = means[window].astype('float32')
            if prefix == 'customer':
                features[f'amount_to_customer_mean_{window}d'] = (
                    amounts / np.maximum(means[window], 1)).astype('float32')
            else:
                features[f'terminal_known_fraud_rate_{window}d_delay7d'] = risks[window].astype('float32')
    return features


def prepare(dataset, smoke=False):
    directory = project_directory(dataset)
    sys.path.insert(0, str(directory / 'src'))
    if (ROOT / 'pipelines').exists():
        raw_root = ROOT / 'dataset' / dataset
    else:
        linked_raw = ROOT / '.local_datasets' / dataset / 'raw'
        raw_root = linked_raw if linked_raw.exists() else ROOT / 'data/raw'
    config = json.loads((directory / 'configs/training.json').read_text())
    original_cap = config['max_train_rows']
    config['max_train_rows'] = CAPS[dataset]
    config['max_validation_rows'] = config.get('max_validation_rows')
    config['max_test_rows'] = config.get('max_test_rows')
    prepared_root = ROOT / 'preprocessed_df' / dataset if (ROOT / 'pipelines').exists() else ROOT / 'preprocessed_df'
    out = prepared_root / ('priority_smoke' if smoke else 'priority_v2')
    out.mkdir(parents=True, exist_ok=True)
    destination = out / 'data.pkl'
    if destination.exists():
        return pd.read_pickle(destination), json.loads((out / 'metadata.json').read_text())

    def assemble(x, y, times):
        part = x.copy().reset_index(drop=True)
        part['target'] = np.asarray(y, dtype='int8')
        part['_event_time'] = np.asarray(times)
        return part.sort_values('_event_time', kind='stable').reset_index(drop=True)

    parts = {}
    notes = []
    if dataset == 'baf':
        from ai4cyber.config import TrainingConfig
        from ai4cyber.data import load_baf_csv, make_temporal_data, replace_missing_sentinels
        # Month is the finest available timestamp: never fabricate within-month chronology.
        values = dict(config, train_months=[0, 1, 2, 3], validation_months=[4, 5],
                      max_validation_rows=None)
        cfg = TrainingConfig(**values)
        raw = load_baf_csv(raw_root / 'Base.csv', cfg)
        splits = make_temporal_data(raw, cfg)
        columns = splits.feature_columns
        for name in ('train', 'test'):
            data = getattr(splits, name)
            parts[name] = assemble(replace_missing_sentinels(data[columns]), data.fraud_bool, data.month)
        for month, name in [(4, 'calibration'), (5, 'selection')]:
            data = bound(splits.validation[splits.validation.month == month], 'fraud_bool', 50000, 42 + month)
            parts[name] = assemble(replace_missing_sentinels(data[columns]), data.fraud_bool, data.month)
        notes.append('BAF train months 0-3, calibration month 4, selection month 5, test months 6-7. No finer timestamps available. This changes the training boundary from v1.')
    elif dataset == 'ieee':
        from ieee_fraud.config import TrainingConfig
        from ieee_fraud.data import load_train_data, sampled_splits, model_frame
        cfg = TrainingConfig(**config)
        raw = load_train_data(raw_root)
        split = sampled_splits(raw, cfg)
        for name in ('train', 'validation', 'test'):
            data = getattr(split, name)
            parts[name] = assemble(model_frame(data), data.isFraud, data.TransactionDT)
    elif dataset == 'handbook':
        from handbook_fraud.data import load_raw, engineer_prior_features, chronological_masks
        raw, features = engineer_prior_features(load_raw(raw_root))
        features = pd.concat([features, rolling_history(raw)], axis=1)
        masks = chronological_masks(raw, config['train_fraction'], config['validation_fraction'])
        for offset, (name, mask) in enumerate(masks.items()):
            data = assemble(features.loc[mask], raw.loc[mask, 'TX_FRAUD'], raw.loc[mask, 'TX_TIME_SECONDS'])
            parts[name] = bound(data, 'target', config[f'max_{name}_rows'], 42 + offset)
        notes.append('Handbook rolling 1/7/30-day history excludes current and simultaneous transactions. Terminal fraud labels are used only after >7 days in an explicit online-replay scenario; past validation/test outcomes can enter later features after this delay. Not a frozen no-feedback deployment.')
    elif dataset == 'paysim':
        from paysim_fraud.data import load_raw, chronological_split, limit_rows, make_features
        raw = load_raw(raw_root / Path(config['dataset_path']).name)
        splits = chronological_split(raw, config['train_fraction'], config['validation_fraction'])
        for offset, (name, data) in enumerate(splits.items()):
            data = limit_rows(data, config[f'max_{name}_rows'], 42 + offset)
            parts[name] = assemble(make_features(data), data.isFraud, data.step)
        notes.append('PaySim retains pre-transaction features only; post-transaction balances and isFlaggedFraud excluded.')
    else:
        from fraudbench import data as module
        if dataset == 'sparkov':
            development = module.load_file(raw_root / Path(config['train_path']).name)
            test_raw = module.load_file(raw_root / Path(config['test_path']).name)
            cut = int(len(development) * (1 - config['validation_fraction']))
            splits = dict(train=development.iloc[:cut], validation=development.iloc[cut:], test=test_raw)
            clock = 'unix_time'
        elif dataset == 'elliptic':
            raw = module.load_file(raw_root / Path(config['features_path']).name,
                                   raw_root / Path(config['classes_path']).name)
            splits = module.temporal_split(raw, config['train_last_step'], config['validation_last_step'])
            clock = 'time_step'
        else:
            raw = module.load_file(raw_root / Path(config['dataset_path']).name)
            splits = module.chronological_split(raw, config['train_fraction'], config['validation_fraction'])
            clock = 'Time' if dataset == 'ulb' else 'step'
        for offset, (name, data) in enumerate(splits.items()):
            data = module.limit_rows(data, config[f'max_{name}_rows'], 42 + offset)
            parts[name] = assemble(module.make_features(data), data[module.TARGET], data[clock])
        if dataset == 'elliptic':
            notes.append('Elliptic includes a recent-history training candidate (steps >=20) to examine temporal stability; no future labels or graph outcomes added.')

    if 'validation' in parts:
        validation = parts.pop('validation').sort_values('_event_time', kind='stable')
        times = np.sort(validation._event_time.unique())
        cut = times[max(1, int(len(times) * .4)) - 1]
        parts['calibration'] = validation[validation._event_time <= cut].copy()
        parts['selection'] = validation[validation._event_time > cut].copy()
    if smoke:
        parts = {name: bound(part, 'target', min(len(part), 1800), 900 + i)
                 for i, (name, part) in enumerate(parts.items())}
    for name, part in parts.items():
        if part.target.nunique() < 2:
            raise ValueError(f'{dataset} {name} has fewer than two classes; enlarge period, never randomly mix time.')
    assert parts['train']._event_time.max() <= parts['calibration']._event_time.min()
    assert parts['calibration']._event_time.max() < parts['selection']._event_time.min()
    assert parts['selection']._event_time.max() <= parts['test']._event_time.min()
    frame = pd.concat([part.assign(_split=name) for name, part in parts.items()], ignore_index=True)
    columns = [c for c in frame if c not in ('target', '_split', '_event_time')]
    metadata = dict(dataset=dataset, feature_columns=columns, old_training_cap=original_cap,
                    requested_training_cap=CAPS[dataset], notes=notes,
                    periods={name: dict(rows=len(part), fraud=int(part.target.sum()),
                                       first=float(part._event_time.min()), last=float(part._event_time.max()))
                             for name, part in parts.items()},
                    preparation='Original raw split boundaries and test sampling seeds preserved except BAF calibration/train change; no test oversampling.')
    frame.to_pickle(destination)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    return frame, metadata
