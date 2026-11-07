"""Extended BAF diagnostics and Elliptic clustering including unknown-label nodes."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from priority_data import ROOT, prepare
from clustering import fit_search, feature_columns, graphs, wilson_interval

VERSION = 'clustering_extended_v2'


def load_data(dataset):
    if dataset == 'baf':
        frame, metadata = prepare(dataset)
        return frame, feature_columns(dataset, metadata['feature_columns'])
    directory = ROOT / 'dataset/elliptic'
    names = ['tx_id', 'time_step', *[f'feature_{i}' for i in range(1, 166)]]
    frame = pd.read_csv(directory / 'elliptic_txs_features.csv', header=None, names=names,
                        usecols=range(95))
    classes = pd.read_csv(directory / 'elliptic_txs_classes.csv', dtype={'class': str}).set_index('txId')['class']
    assert frame.tx_id.is_unique and classes.index.is_unique
    frame['source_class'] = frame.tx_id.map(classes)
    assert frame.source_class.notna().all()
    frame['target'] = frame.source_class.map({'1': 1., '2': 0.})
    frame['_event_time'] = frame.time_step
    frame['_split'] = np.select([frame.time_step <= 34, frame.time_step <= 41], ['train', 'validation'], default='test')
    return frame, [f'feature_{i}' for i in range(1, 94)]


def select(rows):
    table = pd.DataFrame(rows)
    x = (table.k - table.k.min()) / (table.k.max() - table.k.min())
    y = (table.inertia - table.inertia.min()) / max(float(table.inertia.max() - table.inertia.min()), 1e-12)
    elbow = int(table.iloc[np.argmax(1 - x - y)].k)
    eligible = table[(table.min_cluster_share >= .01) & (table.stability_ari_mean >= .6)].copy()
    relaxed = eligible.empty
    if relaxed:
        eligible = table.copy()
    eligible['selection_rank'] = (.5 * eligible.silhouette.rank(ascending=False, method='min')
                                  + .3 * eligible.davies_bouldin.rank(method='min')
                                  + .2 * (eligible.k - elbow).abs().rank(method='min'))
    winner = int(eligible.sort_values(['selection_rank', 'k']).iloc[0].k)
    return winner, elbow, relaxed, eligible[['k', 'selection_rank']].to_dict('records')


def group_metrics(frame, assignments, k):
    records = []
    masks = [('all', np.ones(len(frame), dtype=bool))]
    masks += [(split, frame._split.to_numpy() == split) for split in frame._split.unique()]
    for split, split_mask in masks:
        for cluster in range(k):
            subset = frame.loc[split_mask & (assignments == cluster)]
            known = subset.target.dropna()
            illicit = int(known.sum())
            low, high = wilson_interval(illicit, len(known)) if len(known) else (None, None)
            records.append(dict(k=k, split=split, cluster=cluster, rows=len(subset),
                share=len(subset) / int(split_mask.sum()), unknown_rows=int(subset.target.isna().sum()),
                unknown_share=float(subset.target.isna().mean()) if len(subset) else None,
                labelled_rows=len(known), fraud_cases=illicit,
                fraud_rate=illicit / len(known) if len(known) else None,
                fraud_rate_low=low, fraud_rate_high=high))
    return pd.DataFrame(records)


def run(dataset):
    folder = ROOT / 'evaluation' / dataset / VERSION
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'verification.json').exists():
        print('ALREADY VERIFIED', dataset, flush=True)
        return
    frame, columns = load_data(dataset)
    train = frame[frame._split == 'train']
    protocol = dict(dataset=dataset, method='MiniBatchKMeans', method_type='clustering', feature_columns=columns,
        total_rows=len(frame), training_rows=len(train), train_last=float(train._event_time.max()),
        split_rows=frame.groupby('_split').size().to_dict(),
        label_counts=dict(unknown=int(frame.target.isna().sum()), illicit=int(frame.target.eq(1).sum()), licit=int(frame.target.eq(0).sum())),
        labels_used_for_fit_or_selection=False, validation_test_refit=False,
        selection='Training only: minimum1% cluster size and mean seed ARI>=0.6; if none pass, explicit relaxed-quality fallback. Minimize .5 silhouette descending rank + .3 Davies-Bouldin ascending rank + .2 distance-to-inertia-elbow rank. Ties prefer smaller k. Elbow maximizes distance from normalized inertia chord. Weights are engineering heuristics, not proven optimal.',
        representation='Training-only median/mode imputation, percentile clipping, scaling and encoding; 20D train-fitted SVD where needed. Silhouette/DB use fixed uniform 2000-row training sample; inertia uses training-fit sample (maximum80000).',
        limitations='Descriptive clusters, not fraud predictions. Unknown nodes are never treated as licit. Illicit rates divide by labelled rows only. Initialisation stability is not temporal stability. Test periods previously viewed. Anonymous local-feature availability cannot all be verified. Geometry depends on preprocessing and SVD.')
    (folder / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    k_values = range(2, 7) if dataset == 'baf' else range(2, 9)
    with threadpool_limits(limits=4):
        _, candidates, _, info, models = fit_search(train[columns], k_values=k_values, return_candidates=True)
        k, elbow, relaxed, ranks = select(candidates)
        selected = next(row for row in candidates if row['k'] == k)
        info['quality_constraints_relaxed'] = relaxed
        (folder / 'selection.json').write_text(json.dumps(dict(k=k, elbow_k=elbow, ranks=ranks,
            labels_used=False, test_used=False, quality_constraints_relaxed=relaxed), indent=2))
        tables = []
        for candidate_k, model in models.items():
            values = np.concatenate([model.predict(frame.iloc[start:start + 25000][columns]) for start in range(0, len(frame), 25000)])
            tables.append(group_metrics(frame, values, candidate_k))
            joblib.dump(dict(pipeline=model, feature_columns=columns, selected_k=candidate_k, dataset=dataset),
                        folder / f'k{candidate_k}.joblib', compress=3)
        model = models[k]
        joblib.dump(dict(pipeline=model, feature_columns=columns, selected_k=k, dataset=dataset), folder / 'clustering.joblib', compress=3)
        all_groups = pd.concat(tables, ignore_index=True)
        groups = all_groups[all_groups.k == k].reset_index(drop=True)
        assignments = model.predict(train[columns])
        pre = model.named_steps['preprocess']
        numeric = pre.transformers_[0][2]
        encoded = pre.named_transformers_['numeric'].transform(train[numeric])
        profiles = pd.DataFrame(encoded, columns=numeric).groupby(assignments).mean()
        profiles = profiles[profiles.std().sort_values(ascending=False).head(15).index]
        all_groups.to_csv(folder / 'cluster_metrics_all_k.csv', index=False)
        groups.to_csv(folder / 'cluster_metrics.csv', index=False)
        pd.DataFrame(candidates).to_csv(folder / 'k_comparison.csv', index=False)
        profiles.to_csv(folder / 'numeric_profiles.csv', index_label='cluster')
        pd.DataFrame(columns=['cluster', 'feature', 'value', 'share']).to_csv(folder / 'categorical_profiles.csv', index=False)
        graphs(folder, pd.DataFrame(candidates), groups, profiles)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for axis, metric in zip(axes, ('silhouette', 'inertia', 'davies_bouldin')):
            axis.plot([r['k'] for r in candidates], [r[metric] for r in candidates], 'o-')
            axis.axvline(k, linestyle='--', color='grey')
            axis.set(xlabel='k', title=metric + (' (higher better)' if metric == 'silhouette' else ' (lower better)'))
        fig.tight_layout(); fig.savefig(folder / 'selection_diagnostics.png', dpi=160); plt.close(fig)
        restored = joblib.load(folder / 'clustering.joblib')['pipeline']
        np.testing.assert_array_equal(restored.predict(frame[columns].head(100)), model.predict(frame[columns].head(100)))
    quality = ('Weak or unstable geometry; no candidate met both size and repeatability floors.' if relaxed else
               'Passed stated training size/repeatability floors; this does not establish fraud relevance.')
    report = dict(**protocol, selected_k=k, selected=selected, fit_info=info, quality_assessment=quality)
    (folder / 'report.json').write_text(json.dumps(report, indent=2))
    verify(dataset)
    (folder / 'README.md').write_text(f"# {dataset.upper()} extended clustering\n\nSelected k={k}; silhouette {selected['silhouette']:.4f}, Davies–Bouldin {selected['davies_bouldin']:.4f}. {quality}\n\n{protocol['selection']}\n\n{protocol['representation']}\n\n[Selection diagnostics](selection_diagnostics.png) · [All k metrics](k_comparison.csv) · [All cluster counts and rates](cluster_metrics_all_k.csv) · [Feature profiles](feature_profiles.png)\n\nIllicit/fraud rates use labelled rows only; unknown records are not negatives. Training, validation and test assignments are reported separately.\n\n{protocol['limitations']}\n\nRun `python training_model/extended_clustering.py --dataset {dataset}` from the suite root.\n")
    print('COMPLETE', dataset, 'rows', len(frame), 'k', k, flush=True)


def verify(dataset):
    folder = ROOT / 'evaluation' / dataset / VERSION
    frame, columns = load_data(dataset)
    report = json.loads((folder / 'report.json').read_text())
    expected = pd.read_csv(folder / 'cluster_metrics_all_k.csv')
    actual = []
    with threadpool_limits(limits=4):
        for path in sorted(folder.glob('k[2-8].joblib')):
            bundle = joblib.load(path)
            values = np.concatenate([bundle['pipeline'].predict(frame.iloc[i:i + 25000][columns]) for i in range(0, len(frame), 25000)])
            actual.append(group_metrics(frame, values, bundle['selected_k']))
    pd.testing.assert_frame_equal(pd.concat(actual, ignore_index=True), expected, check_dtype=False, atol=1e-10)
    chosen, _, _, _ = select(pd.read_csv(folder / 'k_comparison.csv').to_dict('records'))
    assert chosen == report['selected_k']
    assert len(frame) == report['total_rows']
    if dataset == 'elliptic':
        assert len(frame) == 203769 and int(frame.target.isna().sum()) == 157205 and len(columns) == 93
    (folder / 'verification.json').write_text(json.dumps(dict(status='passed', dataset=dataset,
        all_k_assignments_reproduced=True, counts_reproduced=True, unknown_not_negative=True, selected_k=chosen), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['baf', 'elliptic'])
    args = parser.parse_args()
    from extended_clustering import run as execute
    execute(args.dataset)
