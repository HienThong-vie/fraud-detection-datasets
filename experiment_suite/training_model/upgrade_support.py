"""Portable, training-fitted feature screening for the v4 pipelines."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureScreen(TransformerMixin, BaseEstimator):
    def __init__(self, local_only=False, missing_limit=.98, correlation_limit=.995, sample_rows=20000, drop_columns=()):
        self.local_only = local_only
        self.missing_limit = missing_limit
        self.correlation_limit = correlation_limit
        self.sample_rows = sample_rows
        self.drop_columns = drop_columns

    def fit(self, X, y=None):
        columns = list(X.columns)
        dropped = list(self.drop_columns)
        if self.local_only:
            dropped += [c for c in columns if c.startswith('feature_') and int(c.split('_')[1]) >= 94]
        vcols = [c for c in columns if c.startswith('V') and c[1:].isdigit()]
        dropped += [c for c in vcols if X[c].isna().mean() > self.missing_limit or X[c].nunique() <= 1]
        useful = [c for c in vcols if c not in dropped]
        if useful:
            sample = X[useful].sample(n=min(len(X), self.sample_rows), random_state=42)
            corr = sample.corr().abs()
            kept = []
            for c in useful:
                if any(corr.loc[c, other] >= self.correlation_limit for other in kept):
                    dropped.append(c)
                else:
                    kept.append(c)
        dropped += [c for c in columns if X[c].nunique(dropna=False) <= 1]
        self.columns_ = [c for c in columns if c not in dropped]
        self.dropped_ = sorted(set(dropped))
        self.n_features_in_ = len(columns)
        return self

    def transform(self, X):
        return X[self.columns_].copy()


class NativeFrame(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        self.categorical_ = [c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
        return self

    def transform(self, X):
        result = X[self.columns_].copy()
        for c in self.categorical_:
            result[c] = result[c].astype('string').fillna('__MISSING__').astype(str)
        return result


def before_bucket_history(entity, clock, amounts, windows, prefix):
    """Aggregate strictly before the time bucket, including all simultaneous events together."""
    codes, _ = pd.factorize(pd.Series(entity, copy=False), sort=False)
    source = pd.DataFrame(dict(entity=codes.astype('int32'), time=np.asarray(clock, dtype='int64'),
                               amount=np.asarray(amounts, dtype='float64')))
    agg = source.groupby(['entity', 'time'], sort=True).amount.agg(['size', 'sum']).reset_index()
    agg['count_end'] = agg.groupby('entity', sort=False)['size'].cumsum()
    agg['amount_end'] = agg.groupby('entity', sort=False)['sum'].cumsum()
    agg['prior_count'] = agg.count_end - agg['size']
    agg['prior_amount'] = agg.amount_end - agg['sum']
    prior_time = agg.groupby('entity', sort=False).time.shift()
    agg[prefix + '_prior_gap'] = (agg.time - prior_time).fillna(-1).astype('float32')
    features = [prefix + '_prior_gap']
    ordered = agg.sort_values('time', kind='stable')
    for window in windows:
        query = ordered[['entity', 'time']].copy()
        query['time'] -= window
        past = pd.merge_asof(query, ordered[['entity', 'time', 'count_end', 'amount_end']],
                             on='time', by='entity', direction='backward', allow_exact_matches=False)
        n = ordered.prior_count.to_numpy() - past.count_end.fillna(0).to_numpy()
        total = ordered.prior_amount.to_numpy() - past.amount_end.fillna(0).to_numpy()
        count_name, mean_name = f'{prefix}_count_{window}', f'{prefix}_mean_amount_{window}'
        agg.loc[ordered.index, count_name] = n.astype('float32')
        agg.loc[ordered.index, mean_name] = (total / np.maximum(n, 1)).astype('float32')
        features.extend([count_name, mean_name])
    keys = pd.MultiIndex.from_arrays([codes, np.asarray(clock, dtype='int64')])
    result = agg.set_index(['entity', 'time'])[features].reindex(keys).reset_index(drop=True)
    for window in windows:
        mean = result[f'{prefix}_mean_amount_{window}'].to_numpy()
        result[f'{prefix}_amount_to_mean_{window}'] = np.clip(
            np.asarray(amounts) / np.maximum(mean, 1), 0, 100000).astype('float32')
    return result.astype('float32')


def baf_features(frame):
    frame = frame.copy()
    for column in frame.columns:
        if column not in {'payment_type', 'housing_status', 'source', 'device_os'}:
            frame[column] = pd.to_numeric(frame[column], errors='raise')
    result = frame.copy()
    numeric = frame.select_dtypes(include='number').columns
    for c in numeric:
        result[c + '_missing'] = frame[c].isna().astype('int8')
    result['missing_numeric_count'] = frame[numeric].isna().sum(axis=1).astype('int16')
    for c in ['velocity_6h', 'velocity_24h', 'velocity_4w', 'zip_count_4w',
              'bank_branch_count_8w', 'device_distinct_emails_8w']:
        result[c + '_log1p'] = np.log1p(frame[c].clip(lower=0))
    for short, long in [('velocity_6h', 'velocity_24h'), ('velocity_24h', 'velocity_4w')]:
        result[short + '_to_' + long] = (frame[short] / frame[long].clip(lower=1)).clip(0, 10000)
    result['device_emails_foreign'] = frame.device_distinct_emails_8w * frame.foreign_request
    result['phone_both_invalid'] = ((frame.phone_home_valid == 0) & (frame.phone_mobile_valid == 0)).astype('int8')
    result['payment_device'] = frame.payment_type.astype(str) + '|' + frame.device_os.astype(str)
    return result
