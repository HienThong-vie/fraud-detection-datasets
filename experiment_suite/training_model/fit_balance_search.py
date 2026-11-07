"""Validation-only capacity search, retaining v2 as an explicit incumbent."""
import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import log_loss

from priority_data import ROOT, prepare, project_directory, evaluation_directory
from priority_search import (BUDGETS, assess_trial, budget_plot, predict_scores,
                             selection_blocks)
from supervised_search import measure, plots

VERSION = 'fit_balance_v3'
TARGETS = ('baf', 'ieee', 'paysim', 'sparkov', 'ulb', 'elliptic')


def diagnostics(y, scores):
    prevalence = float(np.mean(y))
    metrics = measure(y, scores, .5)
    return dict(rows=len(y), positives=int(np.sum(y)), prevalence=prevalence,
                ap=metrics['pr_auc'], roc_auc=metrics['roc_auc'],
                log_loss=float(log_loss(y, np.clip(scores, 1e-12, 1 - 1e-12), labels=[0, 1])),
                ap_above_random=(metrics['pr_auc'] - prevalence) / (1 - prevalence))


def add_diagnostics(record, model, train, calibration, selection, columns):
    record['diagnostics'] = {name: diagnostics(part.target, predict_scores(model, part[columns]))
                             for name, part in [('train', train), ('calibration', calibration),
                                                ('validation', selection)]}
    train_d, valid_d = [record['diagnostics'][name] for name in ('train', 'validation')]
    ap_gap = max(0., train_d['ap_above_random'] - valid_d['ap_above_random'])
    auc_gap = max(0., train_d['roc_auc'] - valid_d['roc_auc'])
    weakest_ap = min(row['ap'] for row in record['validation_blocks'])
    instability = max(0., valid_d['ap'] - weakest_ap)
    record['fit_balance_score'] = record['selection_utility'] - .25 * ap_gap - .15 * auc_gap - .10 * instability
    record['gap_diagnostics'] = dict(positive_ap_skill_gap=ap_gap, positive_roc_auc_gap=auc_gap,
                                     validation_block_ap_drop=instability)
    record['fitting_assessment'] = (
        'large training/validation gap; overfitting or temporal shift remains'
        if ap_gap > .10 or auc_gap > .08 else
        'no large observed training/validation gap; this does not prove optimal fit')
    return record


def admissible(record, incumbent):
    """Performance floors prevent selecting an uninformative but flat learning gap."""
    return (record['validation']['pr_auc'] >= .95 * incumbent['validation']['pr_auc']
            and record['validation']['recall'] >= .85 * incumbent['validation']['recall']
            and record['selection_utility'] >= incumbent['selection_utility'] - .03)


def choose(records, incumbent):
    eligible = [r for r in records if admissible(r, incumbent)]
    return max([incumbent, *eligible], key=lambda r: r['fit_balance_score'])


def inner_split(train):
    times = np.sort(train._event_time.unique())
    if len(times) < 3:
        raise ValueError('At least three training time groups are needed for early stopping')
    boundary = times[max(1, int(.85 * len(times))) - 1]
    earlier, later = train[train._event_time <= boundary], train[train._event_time > boundary]
    if earlier.target.nunique() != 2 or later.target.nunique() != 2:
        raise ValueError('Both classes are required in each inner training period')
    return earlier, later


def options(family, incumbent, smoke=False):
    for index in range(3 if family in ('catboost', 'xgboost') else 2):
        model = clone(incumbent)
        estimator = model.steps[-1][1]
        if family == 'logistic_regression':
            params = dict(C=(.001, 1.)[index], class_weight=('balanced', None)[index], max_iter=2000)
            label = f'C{params["C"]}_{params["class_weight"]}'
        elif family in ('random_forest', 'extra_trees'):
            depth, leaf = [(5, 100), (12, 40)][index]
            params = dict(n_estimators=180, max_depth=depth, min_samples_leaf=leaf,
                          max_features=.5, max_samples=.8 if family == 'random_forest' else None,
                          n_jobs=4, class_weight='balanced')
            label = f'd{depth}_leaf{leaf}'
        elif family == 'xgboost':
            depth, child, penalty = [(2, 100, 100), (3, 60, 60), (5, 40, 40)][index]
            params = dict(n_estimators=900, max_depth=depth, min_child_weight=child,
                          learning_rate=.03, reg_lambda=penalty, reg_alpha=5.,
                          subsample=.7, colsample_bytree=.7, n_jobs=4,
                          early_stopping_rounds=60, eval_metric='aucpr')
            label = f'd{depth}_child{child}_earlystop'
        else:
            params = dict(iterations=1000, depth=[3, 4, 5][index], learning_rate=.03,
                          l2_leaf_reg=[60, 40, 30][index], random_strength=2.,
                          auto_class_weights=['None', 'SqrtBalanced', 'SqrtBalanced'][index],
                          thread_count=4, verbose=False, allow_writing_files=False)
            label = f'd{params["depth"]}_l2{params["l2_leaf_reg"]}_earlystop'
        if smoke:
            if 'n_estimators' in params:
                params['n_estimators'] = 5
            if 'iterations' in params:
                params['iterations'] = 5
        estimator.set_params(**params)
        yield label, model


def fit_candidate(model, family, train, columns, categorical):
    info = dict(training_rows=len(train), inner_training_only=True)
    if family not in ('catboost', 'xgboost'):
        model.fit(train[columns], train.target)
        return model, info
    earlier, later = inner_split(train)
    probe = clone(model)
    preprocessing = probe.steps[0][1]
    x = preprocessing.fit_transform(earlier[columns], earlier.target)
    stop_x = preprocessing.transform(later[columns])
    estimator = probe.steps[-1][1]
    kwargs = dict(eval_set=[(stop_x, later.target)], verbose=False)
    if family == 'catboost':
        kwargs.update(cat_features=categorical, early_stopping_rounds=60)
    estimator.fit(x, earlier.target, **kwargs)
    best_rounds = (estimator.get_best_iteration() if family == 'catboost' else estimator.best_iteration) + 1
    final_estimator = model.steps[-1][1]
    if family == 'catboost':
        final_estimator.set_params(iterations=max(1, best_rounds))
        model.fit(train[columns], train.target, **{model.steps[-1][0] + '__cat_features': categorical})
    else:
        final_estimator.set_params(n_estimators=max(1, best_rounds), early_stopping_rounds=None)
        model.fit(train[columns], train.target)
    info.update(best_rounds=int(best_rounds), inner_train_rows=len(earlier),
                inner_stop_rows=len(later), inner_boundary=float(earlier._event_time.max()),
                inner_stop_first=float(later._event_time.min()))
    return model, info


def diagnostic_plot(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(rows))
    for axis, key, label in zip(axes, ('ap', 'roc_auc'), ('Average precision', 'ROC-AUC')):
        for offset, split in [(-1, 'train'), (0, 'validation'), (1, 'test')]:
            axis.bar(x + offset * .25, [r['diagnostics'][split][key] for r in rows], .25, label=split)
        axis.set_xticks(x, [r['model'] for r in rows], rotation=20)
        axis.set_ylim(0, 1.05)
        axis.set_title(label + ': temporal generalisation')
        axis.legend()
    figure.tight_layout()
    figure.savefig(out / 'fit_diagnostics.png', dpi=160)
    plt.close(figure)


def run(dataset, smoke=False):
    import fcntl
    out = evaluation_directory(dataset) / ('fit_balance_smoke' if smoke else VERSION)
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (out / 'report.json').exists():
            print('ALREADY COMPLETE', dataset, flush=True)
            return
        _run(dataset, smoke, out)


def _run(dataset, smoke, out):
    sys.path.insert(0, str(project_directory(dataset) / 'src'))
    frame, preparation = prepare(dataset, smoke)
    columns = preparation['feature_columns']
    train, calibration, selection, test = [frame[frame._split == name] for name in
                                          ('train', 'calibration', 'selection', 'test')]
    blocks = selection_blocks(selection)
    categorical = [c for c in columns if not pd.api.types.is_numeric_dtype(train[c])]
    previous = evaluation_directory(dataset) / 'priority_search_v2'
    baseline = json.loads((previous / 'report.json').read_text())
    protocol = dict(dataset=dataset, experiment=VERSION, training_device='local Mac CPU',
                    platform=platform.platform(), threads_per_estimator=4, supervised_only=True,
                    smoke=smoke, seed=42, budgets=list(BUDGETS), default_budget=.01,
                    stable_scoring=True, prediction_decimals=12, forest_prediction_threads=1,
                    preparation=preparation,
                    data_sha256=hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
                    selection_rule='v2 weakest temporal-block utility minus 0.25 positive train/selection AP-above-random gap, 0.15 positive ROC-AUC gap and 0.10 validation block AP drop. Per-family and champion floors: at least 95% incumbent validation AP, 85% incumbent recall at 1% calibration budget, and utility no more than 0.03 below incumbent. Incumbents remain eligible. No test-based selection.',
                    limitations=baseline['limitations'] + ' Fit-gap diagnostics are warnings, not guarantees of no over/underfitting. AP-above-random rescales the random baseline but does not fully correct prevalence shift. Reused validation can itself be overfit; use a new future holdout for confirmation. Learning capacity is screened, not mathematically proven optimal.')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    cache = out / 'candidates'
    cache.mkdir(exist_ok=True)
    trials, rows, models, incumbents = [], [], {}, {}
    for old in baseline['models']:
        family = old['model']
        bundle = joblib.load(previous / (family + '.joblib'))
        model = bundle['model']
        baseline_train = train
        if 'recent_steps20plus' in old['variant']:
            baseline_train = train[train._event_time >= 20]
        record = assess_trial(model, family, 'v2_incumbent_' + old['variant'], baseline_train,
                              calibration, selection, blocks, columns, 0.)
        record['reused_checkpoint'] = True
        record = add_diagnostics(record, model, baseline_train, calibration, selection, columns)
        incumbents[family] = copy.deepcopy(record)
        trials.append(record)
        family_records, family_models = [record], {record['variant']: model}
        for label, candidate in options(family, model, smoke):
            saved = cache / f'{family}_{label}.joblib'
            if saved.exists():
                cached = joblib.load(saved)
                candidate, result = cached['model'], cached['record']
            else:
                print('FIT', dataset, family, label, len(train), flush=True)
                started = time.monotonic()
                candidate, fit_info = fit_candidate(candidate, family, train, columns, categorical)
                result = assess_trial(candidate, family, label, train, calibration, selection,
                                      blocks, columns, time.monotonic() - started)
                result['fit_info'] = fit_info
                result = add_diagnostics(result, candidate, train, calibration, selection, columns)
                joblib.dump(dict(model=candidate, record=result), saved, compress=3)
            result['passes_incumbent_floors'] = admissible(result, record)
            trials.append(result)
            family_records.append(result)
            family_models[label] = candidate
            (out / 'trials.json').write_text(json.dumps(trials, indent=2))
            print(dataset, family, label, 'validationAP', round(result['validation']['pr_auc'], 4),
                  'gap', round(result['positive_pr_auc_gap'], 4), 'eligible', result['passes_incumbent_floors'], flush=True)
        best = choose(family_records, record)
        rows.append(copy.deepcopy(best))
        models[family] = family_models[best['variant']]
        print('SELECTED FAMILY', dataset, family, best['variant'], flush=True)
    winner_record = choose(rows, incumbents[baseline['winner']])
    winner = winner_record['model']
    if winner_record['variant'] != next(row['variant'] for row in rows if row['model'] == winner):
        rows = [copy.deepcopy(winner_record) if row['model'] == winner else row for row in rows]
        models[winner] = joblib.load(previous / (winner + '.joblib'))['model']
    (out / 'selection.json').write_text(json.dumps(dict(winner=winner, models=rows,
                                                     incumbent_comparison=incumbents), indent=2))
    scores, cal_scores, valid_scores, operating = {}, {}, {}, []
    for row in rows:
        family, model = row['model'], models[row['model']]
        scores[family] = predict_scores(model, test[columns])
        cal_scores[family] = predict_scores(model, calibration[columns])
        valid_scores[family] = predict_scores(model, selection[columns])
        row['test_operating_points'] = {str(b): measure(test.target, scores[family], row['thresholds'][str(b)]) for b in BUDGETS}
        row['test'] = row['test_operating_points']['0.01']
        row['diagnostics']['test'] = diagnostics(test.target, scores[family])
        for budget in BUDGETS:
            operating.append(dict(model=family, validation_fpr_budget=budget,
                                  threshold=row['thresholds'][str(budget)], **row['test_operating_points'][str(budget)]))
        joblib.dump(dict(model=model, threshold=row['threshold'], thresholds=row['thresholds'],
                         feature_columns=columns, selected_model=family, dataset=dataset,
                         variant=row['variant'], prediction_decimals=12,
                         history_assumptions=preparation['notes']), out / (family + '.joblib'), compress=3)
    joblib.dump(joblib.load(out / (winner + '.joblib')), out / 'champion.joblib', compress=3)
    pd.DataFrame(operating).to_csv(out / 'operating_points.csv', index=False)
    pd.DataFrame([dict(model=r['model'], variant=r['variant'], threshold=r['threshold'],
                       fit_balance_score=r['fit_balance_score'],
                       **{f'{s}_{k}': v for s in ('train', 'validation', 'test') for k, v in r[s].items()})
                  for r in rows]).to_csv(out / 'model_comparison.csv', index=False)
    pd.DataFrame(dict(target=test.target.to_numpy(), **scores)).to_csv(out / 'test_predictions.csv', index=False)
    pd.DataFrame(dict(target=calibration.target.to_numpy(), **cal_scores)).to_csv(out / 'calibration_predictions.csv', index=False)
    plots(out, rows, test.target, scores, valid_scores, calibration.target, cal_scores)
    budget_plot(out, rows)
    diagnostic_plot(out, rows)
    (out / 'example.json').write_text(test[columns].head(1).replace({np.nan: None}).to_json(orient='records', indent=2))
    report = dict(**protocol, winner=winner, total_candidates=len(trials),
                  new_candidate_fits=sum(not r.get('reused_checkpoint', False) for r in trials), models=rows)
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print('COMPLETE', dataset, winner, next(r['variant'] for r in rows if r['model'] == winner), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=TARGETS, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.dataset, args.smoke)
