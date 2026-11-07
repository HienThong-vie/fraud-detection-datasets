"""Training-only exploratory segmentation, with labels reserved for interpretation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.cache/matplotlib'))
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from priority_data import ROOT, CAPS, prepare

VERSION = 'clustering_v1'


class QuantileClipper(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        x = np.asarray(x, dtype=float)
        self.lower_ = np.nanquantile(x, .01, axis=0)
        self.upper_ = np.nanquantile(x, .99, axis=0)
        return self

    def transform(self, x):
        return np.clip(np.asarray(x, dtype=float), self.lower_, self.upper_)


class ClusterRepresentation(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return np.asarray(x.toarray() if hasattr(x, 'toarray') else x, dtype=np.float32)


def feature_columns(dataset, columns):
    excluded = {'target', 'fraud_bool', 'isFraud', 'is_fraud', 'Class', 'class',
                'TransactionID', 'tx_id', 'nameOrig', 'nameDest', 'CUSTOMER_ID', 'TERMINAL_ID',
                'step', 'day', 'Time', 'TransactionDay', 'time_step',
                'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2'}
    return [c for c in columns if c not in excluded and not c.startswith('_')
            and 'fraud' not in c.lower() and not c.lower().endswith('_id')
            and not (dataset == 'elliptic' and c.startswith('feature_') and int(c.split('_')[1]) >= 94)]


def preprocessing(x):
    numeric = list(x.select_dtypes(include=np.number).columns)
    categorical = [c for c in x if c not in numeric]
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                              ('clip', QuantileClipper()), ('scale', StandardScaler())]), numeric),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                                  ('encode', OneHotEncoder(handle_unknown='ignore', min_frequency=20))]), categorical)
    ], sparse_threshold=1.)


def choose_candidate(rows):
    eligible = [r for r in rows if r['min_cluster_share'] >= .01 and r['stability_ari_mean'] >= .6
                and r['silhouette'] is not None]
    fallback = not eligible
    pool = eligible or [r for r in rows if r['silhouette'] is not None]
    if not pool:
        raise ValueError('No candidate has two nonempty clusters on the silhouette sample')
    return max(pool, key=lambda r: (r['silhouette'], -r['k'])), fallback


def fit_search(x, maximum=80000, probe_rows=2000):
    rng = np.random.default_rng(42)
    indices = np.sort(rng.choice(len(x), min(len(x), maximum), replace=False))
    fit = x.iloc[indices]
    preprocessor = preprocessing(fit)
    encoded = preprocessor.fit_transform(fit)
    reduction = None
    if encoded.shape[1] > 20 and len(fit) > 21:
        reduction = TruncatedSVD(n_components=20, n_iter=7, random_state=42)
        z = reduction.fit_transform(encoded)
    else:
        z = encoded.toarray() if hasattr(encoded, 'toarray') else np.asarray(encoded)
    z = np.asarray(z, dtype=np.float32)
    probe = np.sort(rng.choice(len(z), min(probe_rows, len(z)), replace=False))
    results, models = [], {}
    for k in range(2, 9):
        seed_labels = []
        reference = None
        for seed in (42, 52, 62):
            model = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3,
                                    batch_size=2048, max_iter=100, reassignment_ratio=.01)
            model.fit(z)
            seed_labels.append(model.predict(z[probe]))
            if reference is None:
                reference = model
        labels = reference.predict(z)
        counts = np.bincount(labels, minlength=k)
        values = np.unique(seed_labels[0])
        silhouette = float(silhouette_score(z[probe], seed_labels[0])) if 1 < len(values) < len(probe) else None
        stability = [float(adjusted_rand_score(seed_labels[i], seed_labels[j]))
                     for i, j in ((0, 1), (0, 2), (1, 2))]
        row = dict(k=k, silhouette=silhouette, min_cluster_share=float(counts.min() / counts.sum()),
                   stability_ari_mean=float(np.mean(stability)), stability_ari_min=min(stability),
                   inertia_per_row=float(-reference.score(z) / len(z)), cluster_sizes=counts.tolist())
        results.append(row)
        models[k] = reference
        print('K', k, 'silhouette', silhouette, 'stability', row['stability_ari_mean'], flush=True)
    selected, fallback = choose_candidate(results)
    steps = [('preprocess', preprocessor)]
    if reduction is not None:
        steps.append(('svd', reduction))
    steps.append(('representation', ClusterRepresentation()))
    steps.append(('kmeans', models[selected['k']]))
    return Pipeline(steps), results, selected, dict(training_fit_rows=len(fit),
        training_fit_index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        silhouette_sample_rows=len(probe), encoded_features=encoded.shape[1], representation_features=z.shape[1],
        svd_explained_variance=float(reduction.explained_variance_ratio_.sum()) if reduction is not None else None,
        quality_constraints_relaxed=fallback)


def wilson_interval(positive, total):
    p, z = positive / total, 1.96
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0., centre - radius), min(1., centre + radius)


def interpret(frame, assignments):
    rows = []
    for split in ('train', 'calibration', 'selection', 'test'):
        part = frame[frame._split == split]
        labels = assignments.loc[part.index]
        for cluster in sorted(assignments.unique()):
            target = part.loc[labels == cluster, 'target']
            n, positives = len(target), int(target.sum())
            lo, hi = wilson_interval(positives, n) if n else (None, None)
            rows.append(dict(split=split, cluster=int(cluster), rows=n, share=n / max(len(part), 1),
                             fraud_cases=positives, fraud_rate=positives / n if n else None,
                             fraud_rate_low=lo, fraud_rate_high=hi))
    return pd.DataFrame(rows)


def graphs(out, candidates, groups, profiles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(candidates.k, candidates.silhouette, 'o-')
    axes[0].set(xlabel='k', ylabel='Training silhouette (fixed uniform sample)', ylim=(-1, 1))
    axes[1].plot(candidates.k, candidates.stability_ari_mean, 'o-', label='Mean ARI across three seeds')
    axes[1].axhline(.6, linestyle='--', color='grey', label='Selection eligibility floor')
    axes[1].set(xlabel='k', ylabel='Training assignment stability', ylim=(-.05, 1.05))
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / 'silhouette_stability.png', dpi=160); plt.close(fig)
    for metric, filename, ylabel in [('share', 'cluster_sizes.png', 'Share of records'),
                                      ('fraud_rate', 'fraud_rates.png', 'Observed fraud rate (labels used only after fitting)')]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for axis, split in zip(axes, ('train', 'test')):
            part = groups[groups.split == split]
            axis.bar(part.cluster.astype(str), part[metric], color='#2563eb')
            if metric == 'fraud_rate':
                rate = part[metric].to_numpy(dtype=float)
                axis.errorbar(np.arange(len(part)), rate,
                    yerr=np.maximum(0, np.array([rate - part.fraud_rate_low, part.fraud_rate_high - rate], dtype=float)),
                    fmt='none', color='black', capsize=3)
            axis.set(xlabel='Cluster (arbitrary ID)', ylabel=ylabel, title=split + (' — exploratory' if split == 'test' else ''))
        fig.tight_layout(); fig.savefig(out / filename, dpi=160); plt.close(fig)
    fig, axis = plt.subplots(figsize=(12, max(3, .55 * len(profiles))))
    image = axis.imshow(profiles.to_numpy(), cmap='coolwarm', vmin=-2, vmax=2, aspect='auto')
    axis.set_xticks(range(len(profiles.columns)), profiles.columns, rotation=70, ha='right', fontsize=8)
    axis.set_yticks(range(len(profiles)), [f'Cluster {c}' for c in profiles.index])
    axis.set_title('Training numeric feature profiles: clipped, standardized means')
    fig.colorbar(image, ax=axis, label='Training standard deviations (colour clipped ±2)')
    fig.tight_layout(); fig.savefig(out / 'feature_profiles.png', dpi=160); plt.close(fig)


def run(dataset):
    out = ROOT / 'evaluation' / dataset / VERSION
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'report.json').exists():
        print('ALREADY COMPLETE', dataset, flush=True)
        return
    frame, metadata = prepare(dataset)
    columns = feature_columns(dataset, metadata['feature_columns'])
    train = frame[frame._split == 'train']
    if not columns or len(train) < 16:
        raise ValueError('Insufficient training features or records')
    protocol = dict(dataset=dataset, method='MiniBatchKMeans', method_type='clustering',
        training_device='local Mac CPU', feature_columns=columns,
        excluded_columns=[c for c in metadata['feature_columns'] if c not in columns],
        training_sample='Uniform, fixed seed, maximum 80000 rows from the existing prepared training split; no label stratification in this experiment. Existing source tables may have earlier prevalence-preserving sampling.',
        selection='Highest training silhouette among k=2..8 with minimum 1% training-fit cluster share and mean seed ARI >=0.6. If none eligible, highest valid silhouette with an explicit quality warning. Ties favour smaller k. The selected model uses seed42; seed52/62 are stability checks, not label-based selection.',
        preprocessing='Fit median/mode imputation, numeric 1st/99th percentile clipping, StandardScaler and categorical one-hot encoding on the training fit sample only. Train-fitted 20-component TruncatedSVD if encoded dimensions exceed20. Euclidean K-means in that representation.',
        seeds=[42, 52, 62], thread_limit=4, labels_used_for_fit_or_selection=False,
        validation_test_refit=False, labels_used_after_freeze='Per-cluster fraud-rate interpretation only; not classification metrics or cluster names.',
        limitations='Exploratory segmentation, not a fraud classifier. Silhouette is an internal geometry score, not accuracy. Mixed categorical/numeric distances, scaling, sampling and compression affect geometry. Repeated seeds measure initialization sensitivity, not temporal stability. Previously viewed test periods are not a fresh holdout. Elliptic uses only labelled-subset records and local anonymous features; availability cannot all be independently verified. Demographic features, where present, may drive groups; no adverse decisions should be made from clusters. Fraud-rate Wilson intervals are descriptive and assume independent cases; temporal dependence can make them optimistic.')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    with threadpool_limits(limits=4):
        pipeline, rows, selected, fit_info = fit_search(train[columns])
        bundle = dict(pipeline=pipeline, feature_columns=columns, dataset=dataset, selected_k=selected['k'])
        joblib.dump(bundle, out / 'clustering.joblib', compress=3)
        (out / 'selection.json').write_text(json.dumps(dict(selected=selected, fit_info=fit_info,
                                                          labels_used=False, test_used=False), indent=2))
        assignments = pd.Series(index=frame.index, dtype='int32')
        for start in range(0, len(frame), 25000):
            part = frame.iloc[start:start + 25000]
            assignments.loc[part.index] = pipeline.predict(part[columns])
        assignments = assignments.astype('int32')
        groups = interpret(frame, assignments)
        prep = pipeline.named_steps['preprocess']
        numeric = prep.transformers_[0][2]
        transformed = prep.named_transformers_['numeric'].transform(train[numeric])
        profile = pd.DataFrame(transformed, columns=numeric, index=train.index).groupby(assignments.loc[train.index]).mean()
        important = profile.std(axis=0).sort_values(ascending=False).head(15).index
        profiles = profile[important]
        categories = []
        for cluster in sorted(assignments.unique()):
            members = train.loc[assignments.loc[train.index] == cluster]
            for column in prep.transformers_[1][2]:
                frequencies = members[column].fillna('(missing)').value_counts(normalize=True)
                for value, share in frequencies.head(3).items():
                    categories.append(dict(cluster=int(cluster), feature=column, value=str(value), share=float(share)))
        pd.DataFrame(rows).to_csv(out / 'k_comparison.csv', index=False)
        groups.to_csv(out / 'cluster_metrics.csv', index=False)
        profiles.to_csv(out / 'numeric_profiles.csv', index_label='cluster')
        pd.DataFrame(categories, columns=['cluster', 'feature', 'value', 'share']).to_csv(out / 'categorical_profiles.csv', index=False)
        graphs(out, pd.DataFrame(rows), groups, profiles)
        restored = joblib.load(out / 'clustering.joblib')['pipeline']
        check = frame[columns].head(128)
        assert np.array_equal(restored.predict(check), pipeline.predict(check))
    quality = ('Weak/unstable geometry: no candidate satisfied both quality floors; do not claim meaningful fraud groups.'
               if fit_info['quality_constraints_relaxed'] else
               'Passed the stated size/repeatability floors; this does not establish real-world or fraud relevance.')
    report = dict(**protocol, selected_k=selected['k'], selected=selected, fit_info=fit_info,
                  quality_assessment=quality, split_rows=frame.groupby('_split').size().to_dict(),
                  saved_pipeline_reproduced=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    write_card(out, report, groups, profiles)
    print('COMPLETE', dataset, 'k', selected['k'], quality, flush=True)


def write_card(out, report, groups, profiles):
    lines = [f"# {report['dataset'].upper()}: exploratory clustering", '',
             f"MiniBatch K-means selected **k={report['selected_k']}** using training-only geometry. "
             f"Training silhouette: {report['selected']['silhouette']:.3f}; mean seed ARI: {report['selected']['stability_ari_mean']:.3f}.", '',
             report['quality_assessment'], '',
             'This is the second method type alongside the four supervised classifiers. It is not a fifth '
             'fraud classifier, and cluster IDs do not mean fraud/legitimate.', '',
             '| Cluster | Train share | Train fraud rate | Test share* | Test fraud rate* |',
             '|---|---:|---:|---:|---:|']
    for cluster in profiles.index:
        a = groups[(groups.split == 'train') & (groups.cluster == cluster)].iloc[0]
        b = groups[(groups.split == 'test') & (groups.cluster == cluster)].iloc[0]
        lines.append(f'| {cluster} | {a.share:.1%} | {a.fraud_rate:.2%} | {b.share:.1%} | {b.fraud_rate:.2%} |')
    lines += ['', '*Previously viewed test periods; interpretation only. Neither fraud rates nor test data '
              'selected k, preprocessing or seed. Differences are associations, not causal evidence.', '',
              '## Training feature interpretation', '']
    for cluster, values in profiles.iterrows():
        top = values.abs().sort_values(ascending=False).head(3).index
        lines.append(f"- Cluster {cluster}: " + ', '.join(f'{name} {values[name]:+.2f} training SD' for name in top) + '.')
    lines += ['', 'Profiles describe training numeric means relative to the training-fit scale. '
              'Inspect categorical profiles as well; numeric summaries alone do not fully explain mixed-feature distances.', '',
              '## Reproduce', '', 'Run from the suite root:', '', '```bash',
              f"python training_model/clustering.py --dataset {report['dataset']}", '```', '',
              'Completed folders are preserved. Use a new versioned directory for a new experiment.', '',
              '## Evidence', '', '[Selection and seed stability](silhouette_stability.png) · '
              '[Cluster sizes](cluster_sizes.png) · [Numeric profiles](feature_profiles.png) · '
              '[Fraud-rate interpretation](fraud_rates.png) · [Metrics](cluster_metrics.csv) · '
              '[Full protocol](report.json)', '', '## Limitations', '', report['limitations'], '']
    (out / 'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(CAPS))
    args = parser.parse_args()
    from clustering import run as run_clustering
    run_clustering(args.dataset)
