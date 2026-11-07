"""Versioned feature enrichment with frozen v2 evaluation rows and training-only sampling."""
import gc
import hashlib
import json
import sys
from collections import deque

import numpy as np
import pandas as pd

from priority_data import ROOT, project_directory
from upgrade_support import before_bucket_history, baf_features

TARGETS = ('baf', 'ieee', 'paysim', 'elliptic')


def fingerprint(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def incoming_graph_features(nodes, edges):
    """Incoming spend edges are observable at transaction creation; outgoing future edges are excluded."""
    ids = pd.Index(nodes.tx_id)
    source = ids.get_indexer(edges.iloc[:, 0])
    dest = ids.get_indexer(edges.iloc[:, 1])
    available = (source >= 0) & (dest >= 0)
    source, dest = source[available], dest[available]
    clock = nodes.time_step.to_numpy()
    causal = clock[source] <= clock[dest]
    source, dest = source[causal], dest[causal]
    n = len(nodes)
    indegree = np.bincount(dest, minlength=n)
    remaining = indegree.copy()
    depth = np.zeros(n, dtype='int32')
    order = np.argsort(source, kind='stable')
    neighbors = dest[order]
    boundaries = np.r_[0, np.cumsum(np.bincount(source, minlength=n))]
    queue = deque(np.flatnonzero(remaining == 0))
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for child in neighbors[boundaries[node]:boundaries[node + 1]]:
            depth[child] = max(depth[child], depth[node] + 1)
            remaining[child] -= 1
            if remaining[child] == 0:
                queue.append(child)
    if visited != n:
        raise ValueError('Graph is not a DAG; incoming causal-depth features cannot be justified')
    out = pd.DataFrame(index=nodes.index)
    out['incoming_parent_count'] = indegree.astype('float32')
    out['incoming_ancestor_depth'] = depth.astype('float32')
    for label, values in [('indegree', indegree), ('depth', depth), ('age_steps', clock[dest] - clock[source])]:
        weights = values if label == 'age_steps' else values[source]
        out['parent_mean_' + label] = (np.bincount(dest, weights=weights, minlength=n)
                                        / np.maximum(indegree, 1)).astype('float32')
    for i in range(1, 17):
        values = nodes[f'feature_{i}'].to_numpy()
        out[f'parent_local_mean_{i}'] = (np.bincount(dest, weights=values[source], minlength=n)
                                        / np.maximum(indegree, 1)).astype('float32')
    return out


def _align_parts(parts, old):
    for split in ('train', 'validation', 'test'):
        parts[split] = parts[split].sort_values('_event_time', kind='stable').reset_index(drop=True)
    validation = parts.pop('validation')
    times = np.sort(validation._event_time.unique())
    cut = times[max(1, int(len(times) * .4)) - 1]
    parts['calibration'] = validation[validation._event_time <= cut].copy()
    parts['selection'] = validation[validation._event_time > cut].copy()
    for name in ('calibration', 'selection', 'test'):
        reference = old[old._split == name].reset_index(drop=True)
        actual = parts[name].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[reference.columns.drop('_split')],
                                      reference.drop(columns='_split'), check_dtype=False)
    return parts


def prepare(dataset):
    directory = ROOT / 'preprocessed_df' / dataset / 'feature_policy_v4'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'metadata.json').exists():
        return pd.read_pickle(directory / 'data.pkl'), pd.read_pickle(directory / 'natural_train.pkl'), json.loads((directory / 'metadata.json').read_text())
    sys.path.insert(0, str(project_directory(dataset) / 'src'))
    old_root = ROOT / 'preprocessed_df' / dataset / 'priority_v2'
    old = pd.read_pickle(old_root / 'data.pkl')
    old_meta = json.loads((old_root / 'metadata.json').read_text())
    old_columns = old_meta['feature_columns']
    raw_root = ROOT / 'dataset' / dataset
    notes, sampling = [], {}
    if dataset == 'baf':
        enriched = baf_features(old[old_columns])
        frame = pd.concat([enriched, old[['target', '_event_time', '_split']]], axis=1)
        natural = frame[frame._split == 'train'].copy()
        notes.append('Missingness flags, nonnegative log transforms, velocity ratios and device/payment interactions. No protected columns reintroduced; month-level timing only.')
    elif dataset in ('ieee', 'paysim'):
        config = json.loads((project_directory(dataset) / 'configs/training.json').read_text())
        config['max_train_rows'] = 300000
        if dataset == 'ieee':
            from ieee_fraud.config import TrainingConfig
            from ieee_fraud.data import load_train_data, sampled_splits, model_frame
            raw = load_train_data(raw_root)
            vcols = [f'V{i}' for i in range(1, 340)]
            extra = pd.read_csv(raw_root / 'train_transaction.csv', usecols=['TransactionID', *vcols],
                                dtype={c: 'float32' for c in vcols})
            raw = raw.merge(extra, on='TransactionID', how='left', validate='one_to_one')
            del extra
            raw = raw.sort_values(['TransactionDT', 'TransactionID'], kind='stable').reset_index(drop=True)
            entity = raw[['card1', 'card2', 'card3', 'card5']].astype('string').fillna('?').agg('|'.join, axis=1)
            history = before_bucket_history(entity, raw.TransactionDT, raw.TransactionAmt, (86400, 604800), 'card_group')
            address = raw[['card1', 'addr1']].astype('string').fillna('?').agg('|'.join, axis=1)
            history = pd.concat([history, before_bucket_history(address, raw.TransactionDT, raw.TransactionAmt,
                                                                (86400,), 'card_address_group')], axis=1)
            enriched = pd.concat([model_frame(raw), raw[vcols], history], axis=1)
            raw['_row_id'] = np.arange(len(raw))
            splits = sampled_splits(raw, TrainingConfig(**config))
            source_parts = {name: getattr(splits, name) for name in ('train', 'validation', 'test')}
            target, clock = 'isFraud', 'TransactionDT'
            notes.append('All 339 V-columns read. Missing/constant/highly correlated V-column screening fits inside each training pipeline, including forward-validation fits. Card-group proxies are not asserted to be unique customer IDs. Strictly prior 1-day/7-day activity excludes simultaneous timestamps and all outcome labels.')
        else:
            from paysim_fraud.data import chronological_split, limit_rows, make_features
            path = next(raw_root.glob('*.csv'))
            raw = pd.read_csv(path, usecols=['step', 'type', 'amount', 'nameOrig', 'nameDest', 'isFraud'],
                              dtype=dict(step='int16', type='category', amount='float32', isFraud='int8'))
            raw['_row_id'] = np.arange(len(raw))
            history = before_bucket_history(raw.nameOrig, raw.step, raw.amount, (24, 168), 'sender')
            history = pd.concat([history, before_bucket_history(raw.nameDest, raw.step, raw.amount,
                                                                (24, 168), 'recipient')], axis=1)
            enriched = pd.concat([make_features(raw), history], axis=1)
            enriched['amount_log1p'] = np.log1p(raw.amount.clip(lower=0))
            splits = chronological_split(raw, config['train_fraction'], config['validation_fraction'])
            source_parts = {name: limit_rows(part, config[f'max_{name}_rows'], 42 + i)
                            for i, (name, part) in enumerate(splits.items())}
            target, clock = 'isFraud', 'step'
            full_train = splits['train']
            positive = full_train[full_train.isFraud == 1]
            negatives = full_train[full_train.isFraud == 0]
            picked = negatives.sample(n=min(len(negatives), 300000 - len(positive)), random_state=42)
            retained = pd.concat([positive, picked]).sort_values('step', kind='stable')
            negative_factor = len(negatives) / len(picked)
            sampling = dict(available_training_rows=len(full_train), available_training_fraud=len(positive),
                            retained_training_rows=len(retained), retained_training_fraud=len(positive),
                            negative_inclusion_probability=len(picked) / len(negatives),
                            positive_inclusion_probability=1., negative_inverse_probability_weight=negative_factor)
            notes.append('All fraud cases from the original training period retained; only training negatives are sampled. Inverse inclusion weights restore the original training class distribution for population-weighted fits/diagnostics. Calibration, selection, test and forward-evaluation rows remain natural-prevalence samples. No balance columns or isFlaggedFraud used. Sender/recipient 24h/168h history uses only earlier step buckets; requires an online raw-event history, never outcome feedback.')
        def assembled(data):
            indices = data._row_id.to_numpy()
            part = enriched.iloc[indices].reset_index(drop=True).copy()
            part['target'] = data[target].to_numpy(dtype='int8')
            part['_event_time'] = data[clock].to_numpy()
            return part
        parts = _align_parts({name: assembled(part) for name, part in source_parts.items()}, old)
        natural = parts['train'].assign(_split='train').copy()
        pd.testing.assert_frame_equal(natural[old.columns].reset_index(drop=True),
                                      old[old._split == 'train'].reset_index(drop=True), check_dtype=False)
        if dataset == 'paysim':
            parts['train'] = assembled(retained)
            parts['train']['_sample_weight'] = np.where(parts['train'].target == 1, 1., negative_factor)
        frame = pd.concat([part.assign(_split=name) for name, part in parts.items()], ignore_index=True)
        del raw, enriched, history, parts, source_parts, splits
        gc.collect()
    else:
        nodes = pd.read_csv(raw_root / 'elliptic_txs_features.csv', header=None,
                            names=['tx_id', 'time_step', *[f'feature_{i}' for i in range(1, 166)]])
        edges = pd.read_csv(raw_root / 'elliptic_txs_edgelist.csv')
        graph = incoming_graph_features(nodes, edges)
        graph['tx_id'] = nodes.tx_id
        from fraudbench.data import load_file, temporal_split
        labelled = load_file(raw_root / 'elliptic_txs_features.csv', raw_root / 'elliptic_txs_classes.csv')
        labelled = labelled.merge(graph, on='tx_id', how='left', validate='one_to_one')
        config = json.loads((project_directory(dataset) / 'configs/training.json').read_text())
        splits = temporal_split(labelled, config['train_last_step'], config['validation_last_step'])
        columns = [*old_columns, *graph.columns.drop('tx_id')]
        parts = {name: part[columns].assign(target=part.is_illicit.to_numpy(),
                                           _event_time=part.time_step.to_numpy()).reset_index(drop=True)
                 for name, part in splits.items()}
        parts = _align_parts(parts, old)
        frame = pd.concat([part.assign(_split=name) for name, part in parts.items()], ignore_index=True)
        natural = parts['train'].assign(_split='train').copy()
        notes.append('New Elliptic models exclude supplied feature_94..165 (72 forward/backward neighborhood aggregates), keeping 93 local numeric features. Incoming parent counts/depth/parent-local summaries use the directed spend DAG, no labels or outgoing future children. Within-step parents are assumed observable when their spending child is created; original local features lack finer timestamps and their online availability cannot be independently verified. Unknown-class nodes provide structure only, never negative targets. Full/14-step/7-step recent training windows are compared without changing evaluation boundaries.')
    frame['_sample_weight'] = frame.get('_sample_weight', pd.Series(1., index=frame.index)).fillna(1.)
    natural['_sample_weight'] = 1.
    columns = [c for c in frame if c not in ('target', '_event_time', '_split', '_sample_weight')]
    for name in ('calibration', 'selection', 'test'):
        reference = old[old._split == name].reset_index(drop=True)
        actual = frame[frame._split == name][old.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, reference, check_dtype=False)
    meta = dict(dataset=dataset, feature_columns=columns, notes=notes, sampling=sampling,
                old_preparation=old_meta, data_sha256=fingerprint(frame), natural_train_sha256=fingerprint(natural),
                frozen_evaluation_rows=True, periods={name: dict(rows=len(part), fraud=int(part.target.sum()),
                    first=float(part._event_time.min()), last=float(part._event_time.max()))
                    for name, part in frame.groupby('_split', sort=False)})
    frame.to_pickle(directory / 'data.pkl')
    natural.to_pickle(directory / 'natural_train.pkl')
    (directory / 'metadata.json').write_text(json.dumps(meta, indent=2))
    return frame, natural, meta
