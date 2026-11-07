"""Experiment feature_policy_v4: four supervised model families, engineered features and threshold policies.

Steps (BAF): train on months 0-3, calibrate review thresholds on month 4 for FPR budgets of 0.5-5%, choose the
model on month 5 with the selection score, freeze the choice, and then score test months 6-7.
Selection score = weakest-block utility (0.65 x AP + 0.35 x mean recall over the budgets - 0.05 x relative FPR
overshoot) - 0.15 x training-validation gap in AP above random - 0.10 x ROC-AUC gap. Test data never selects."""
import argparse
import copy
import json
import os
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from priority_data import ROOT, project_directory, evaluation_directory, bound
from priority_search import predict_scores, selection_blocks, priority_utility, budget_plot
from supervised_search import measure, plots
from fit_balance_search import inner_split, diagnostic_plot
from threshold_policy import BUDGETS, calibrate, policy_key
from upgrade_data import TARGETS, prepare, fingerprint
from upgrade_support import FeatureScreen, NativeFrame

VERSION = 'feature_policy_v4'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))


def weighted_measure(y, scores, threshold, weight=None):
    """Confusion-matrix metrics at one threshold, optionally weighted (PaySim training rows carry
    inverse-sampling weights)."""
    if weight is None:
        return measure(y, scores, threshold)
    tn, fp, fn, tp = confusion_matrix(y, scores >= threshold, labels=[0, 1], sample_weight=weight).ravel()
    precision, recall = tp / max(tp + fp, 1e-12), tp / max(tp + fn, 1e-12)
    return dict(precision=float(precision), recall=float(recall), f1=float(2 * precision * recall / max(precision + recall, 1e-12)),
                pr_auc=float(average_precision_score(y, scores, sample_weight=weight)),
                roc_auc=float(roc_auc_score(y, scores, sample_weight=weight)), fpr=float(fp / max(tn + fp, 1e-12)),
                accuracy=float((tn + tp) / (tn + tp + fn + fp)), tn=float(tn), fp=float(fp), fn=float(fn), tp=float(tp))


def diagnostic(y, scores, weight=None):
    """Threshold-free diagnostics for one split: prevalence, AP, ROC-AUC, AP above the random baseline and
    log loss."""
    metrics = weighted_measure(y, scores, .5, weight)
    prevalence = float(np.average(y, weights=weight))
    return dict(rows=len(y), positives=int(np.sum(y)), prevalence=prevalence, ap=metrics['pr_auc'],
                roc_auc=metrics['roc_auc'], ap_above_random=(metrics['pr_auc'] - prevalence) / (1 - prevalence),
                log_loss=float(log_loss(y, np.clip(scores, 1e-12, 1 - 1e-12), sample_weight=weight, labels=[0, 1])))


def preprocess(dataset, index, native=False):
    """Preprocessing for one candidate setting: a training-fitted FeatureScreen, then native columns for
    CatBoost or imputation, scaling and one-hot encoding for the other families."""
    drop = ('TransactionDay', 'step', 'day') if index else ()
    screen = FeatureScreen(local_only=dataset == 'elliptic', missing_limit=.98 if index == 0 else .995,
                           correlation_limit=.995 if index == 0 else .999, drop_columns=drop)
    if native:
        return Pipeline([('screen', screen), ('encode', NativeFrame())])
    transform = ColumnTransformer([
        ('numeric', Pipeline([('missing', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                              ('scale', StandardScaler())]), make_column_selector(dtype_include=np.number)),
        ('categorical', Pipeline([('missing', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                                  ('encode', OneHotEncoder(handle_unknown='ignore', min_frequency=20))]),
         make_column_selector(dtype_exclude=np.number))], sparse_threshold=1.)
    return Pipeline([('screen', screen), ('encode', transform)])


def candidate_models(dataset, family, train, smoke=False):
    """Yield the new candidates of one family: two regularised settings (a third recent-window one on
    Elliptic), with class weights for imbalance."""
    count = 3 if dataset == 'elliptic' else 2
    prevalence = float(np.average(train.target, weights=train._sample_weight))
    for index in range(count):
        setting = min(index, 1)
        pre = preprocess(dataset, setting, native=family == 'catboost')
        if family == 'logistic_regression':
            estimator = LogisticRegression(C=[.01, .1][setting], max_iter=1000,
                                           class_weight=None if dataset == 'paysim' else 'balanced', random_state=42)
        elif family in ('random_forest', 'extra_trees'):
            kind = RandomForestClassifier if family == 'random_forest' else ExtraTreesClassifier
            estimator = kind(n_estimators=140 if not smoke else 5, max_depth=[8, 12][setting],
                             min_samples_leaf=[60, 30][setting], max_features='sqrt', n_jobs=4,
                             class_weight=None if dataset == 'paysim' else 'balanced', random_state=42)
        elif family == 'catboost':
            from catboost import CatBoostClassifier
            estimator = CatBoostClassifier(iterations=700 if not smoke else 5, depth=[4, 6][setting],
                l2_leaf_reg=[40, 30][setting], learning_rate=.035, auto_class_weights='SqrtBalanced',
                loss_function='Logloss', random_seed=42, thread_count=4, verbose=False, allow_writing_files=False)
        else:
            from xgboost import XGBClassifier
            estimator = XGBClassifier(n_estimators=800 if not smoke else 5, max_depth=[3, 5][setting],
                min_child_weight=([5, 10] if dataset == 'paysim' else [20, 30])[setting],
                reg_lambda=[30, 40][setting], reg_alpha=2., learning_rate=.035, subsample=.8,
                colsample_bytree=.8, scale_pos_weight=((1 - prevalence) / prevalence) ** (.5 if setting else 0),
                tree_method='hist', objective='binary:logistic', eval_metric='aucpr', n_jobs=4, random_state=42)
        window = (None, 14, 7)[index] if dataset == 'elliptic' else None
        yield f'features_setting{setting}_window{window}', Pipeline([('preprocessor', pre), ('classifier', estimator)]), window


def fit_model(model, family, train, columns):
    """Fit one candidate on training rows only; boosted trees pick their number of rounds by early stopping
    on the latest training period, then refit on all training rows."""
    weights = train._sample_weight.to_numpy(dtype=float)
    weights = weights / weights.mean()
    info = dict(rows=len(train), fraud=int(train.target.sum()), best_rounds=None)
    if family in ('random_forest', 'extra_trees'):
        model.steps[-1][1].set_params(n_jobs=4)
    if family in ('xgboost', 'catboost') and train._event_time.nunique() >= 3:
        earlier, stop = inner_split(train)
        probe = clone(model)
        transform, classifier = probe.steps[0][1], probe.steps[-1][1]
        x = transform.fit_transform(earlier[columns], earlier.target)
        stop_x = transform.transform(stop[columns])
        earlier_weight = earlier._sample_weight.to_numpy(dtype=float)
        earlier_weight = earlier_weight / earlier_weight.mean()
        stop_weight = stop._sample_weight.to_numpy(dtype=float)
        stop_weight = stop_weight / stop_weight.mean()
        if family == 'xgboost':
            classifier.set_params(early_stopping_rounds=50)
            classifier.fit(x, earlier.target, sample_weight=earlier_weight,
                           eval_set=[(stop_x, stop.target)], sample_weight_eval_set=[stop_weight], verbose=False)
            rounds = classifier.best_iteration + 1
            model.steps[-1][1].set_params(n_estimators=rounds, early_stopping_rounds=None)
        else:
            from catboost import Pool
            cats = [c for c in x if not pd.api.types.is_numeric_dtype(x[c])]
            classifier.fit(Pool(x, earlier.target, cat_features=cats, weight=earlier_weight),
                           eval_set=Pool(stop_x, stop.target, cat_features=cats, weight=stop_weight),
                           early_stopping_rounds=50, verbose=False)
            rounds = classifier.get_best_iteration() + 1
            model.steps[-1][1].set_params(iterations=rounds)
        info.update(best_rounds=int(rounds), inner_train_last=float(earlier._event_time.max()),
                    inner_stop_first=float(stop._event_time.min()))
    elif family in ('xgboost', 'catboost'):
        classifier = model.steps[-1][1]
        if family == 'xgboost':
            classifier.set_params(n_estimators=min(300, classifier.n_estimators), early_stopping_rounds=None)
        else:
            classifier.set_params(iterations=min(300, classifier.get_params().get('iterations', 300)))
        info['early_stopping_skipped'] = 'Fewer than three training time groups; fixed <=300 rounds, no random split'
    transform, classifier = model.steps[0][1], model.steps[-1][1]
    x = transform.fit_transform(train[columns], train.target)
    kwargs = dict(sample_weight=weights)
    if family == 'catboost':
        kwargs['cat_features'] = [c for c in x if not pd.api.types.is_numeric_dtype(x[c])]
    classifier.fit(x, train.target, **kwargs)
    if family in ('random_forest', 'extra_trees'):
        classifier.set_params(n_jobs=1)
    if hasattr(transform, 'named_steps') and 'screen' in transform.named_steps:
        screen = transform.named_steps['screen']
        info['retained_feature_columns'] = screen.columns_
        info['dropped_feature_columns'] = screen.dropped_
    return model, info


def assess(model, family, variant, train, calibration, selection, columns):
    """Score a fitted candidate without test data: thresholds from the calibration month, metrics and the
    selection score from the selection month."""
    if family in ('random_forest', 'extra_trees'):
        model.steps[-1][1].set_params(n_jobs=1)
    cal = predict_scores(model, calibration[columns])
    policies = calibrate(calibration.target, cal)
    thresholds = {str(b): policies[policy_key('point', b)]['threshold'] for b in BUDGETS}
    scores = predict_scores(model, selection[columns])
    train_scores = predict_scores(model, train[columns])
    train_w = train._sample_weight.to_numpy(dtype=float)
    training = weighted_measure(train.target, train_scores, thresholds['0.01'], train_w)
    validation = measure(selection.target, scores, thresholds['0.01'])
    blocks = []
    for part in selection_blocks(selection):
        s = predict_scores(model, part[columns])
        metrics = {str(b): measure(part.target, s, thresholds[str(b)]) for b in BUDGETS}
        blocks.append(dict(ap=metrics['0.01']['pr_auc'], operating=metrics,
                           utility=priority_utility(metrics['0.01']['pr_auc'], metrics)))
    diag = dict(train=diagnostic(train.target, train_scores, train_w),
                calibration=diagnostic(calibration.target, cal), validation=diagnostic(selection.target, scores))
    gap = max(0., diag['train']['ap_above_random'] - diag['validation']['ap_above_random'])
    auc_gap = max(0., training['roc_auc'] - validation['roc_auc'])
    utility = min(b['utility'] for b in blocks)
    return dict(model=family, variant=variant, threshold=thresholds['0.01'], thresholds=thresholds,
                policies=policies, train=training, validation=validation, diagnostics=diag,
                positive_pr_auc_gap=max(0., training['pr_auc'] - validation['pr_auc']),
                selection_utility=utility, fit_balance_score=utility - .15 * gap - .10 * auc_gap,
                validation_blocks=blocks, training_rows=len(train),
                validation_operating_points={str(b): measure(selection.target, scores, thresholds[str(b)]) for b in BUDGETS},
                fitting_assessment=('large training/validation gap; investigate overfitting or temporal shift'
                                    if gap > .10 or auc_gap > .08 else
                                    'no large observed training/validation gap; optimal fit is not proven'))


def passes(record, incumbent):
    """Floors a new candidate must meet against the previous version (>= 95% of its validation AP, >= 85% of
    its 1%-budget recall, utility within 0.03)."""
    return (record['validation']['pr_auc'] >= .95 * incumbent['validation']['pr_auc']
            and record['validation']['recall'] >= .85 * incumbent['validation']['recall']
            and record['selection_utility'] >= incumbent['selection_utility'] - .03)


def windowed(train, window):
    """Keep only the most recent time steps of training data (None keeps all); used for the Elliptic
    recent-window candidate."""
    return train if window is None else train[train._event_time >= train._event_time.max() - window + 1]


def forward_refits(model, family, window, train, natural, columns, smoke=False):
    """Diagnostic only: two refits inside the training period (fit, calibrate, evaluate on later blocks) to
    check stability over time; never used to select."""
    times = np.sort(natural._event_time.unique())
    if len(times) < 4:
        return [dict(skipped='Fewer than four training time groups')]
    if len(times) == 4:
        boundaries = [(times[0], times[1], times[2]), (times[1], times[2], times[3])]
    else:
        boundaries = [(times[int(len(times) * a) - 1], times[int(len(times) * b) - 1], times[int(len(times) * c) - 1])
                      for a, b, c in [(.6, .7, .8), (.8, .9, 1.)]]
    results = []
    for index, (last_train, last_cal, last_eval) in enumerate(boundaries):
        fit = windowed(train[train._event_time <= last_train], window).copy()
        cal = natural[(natural._event_time > last_train) & (natural._event_time <= last_cal)]
        evaluate = natural[(natural._event_time > last_cal) & (natural._event_time <= last_eval)]
        if any(part.target.nunique() != 2 for part in (fit, cal, evaluate)):
            results.append(dict(fold=index, skipped='Insufficient class support; no random time mixing'))
            continue
        limit = 2000 if smoke else 100000
        if len(fit) > limit:
            positives, negatives = fit[fit.target == 1], fit[fit.target == 0]
            if len(positives) < limit // 2:
                chosen = negatives.sample(n=min(len(negatives), limit - len(positives)), random_state=70 + index)
                chosen = chosen.copy()
                chosen['_sample_weight'] *= len(negatives) / len(chosen)
                fit = pd.concat([positives, chosen]).sort_values('_event_time', kind='stable')
            else:
                fit = bound(fit, 'target', limit, 70 + index)
        cal, evaluate = [bound(p, 'target', 50000 if not smoke else 2000, 80 + index) for p in (cal, evaluate)]
        clone_model, info = fit_model(clone(model), family, fit, columns)
        record = assess(clone_model, family, 'forward_refit', fit, cal, evaluate, columns)
        results.append(dict(fold=index, train_last=float(fit._event_time.max()),
            calibration_first=float(cal._event_time.min()), calibration_last=float(cal._event_time.max()),
            evaluation_first=float(evaluate._event_time.min()), evaluation_last=float(evaluate._event_time.max()),
            train_rows=len(fit), calibration_rows=len(cal), calibration_fraud=int(cal.target.sum()),
            evaluation_rows=len(evaluate), evaluation_fraud=int(evaluate.target.sum()),
            validation=record['validation'], operating=record['validation_operating_points'], fit_info=info))
    return results


def run(dataset, smoke=False):
    """Run the experiment once per dataset; a file lock and an existing report stop published results from
    being overwritten."""
    import fcntl
    folder = evaluation_directory(dataset) / ('feature_policy_smoke' if smoke else VERSION)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder / 'report.json').exists():
            print('ALREADY COMPLETE', dataset, flush=True)
            return
        _run(dataset, smoke, folder)


def _run(dataset, smoke, folder):
    """The experiment: re-score v3 incumbents, fit and assess new candidates, choose the champion on
    validation, freeze it, then score the test months and save all outputs."""
    sys.path.insert(0, str(project_directory(dataset) / 'src'))
    frame, natural, metadata = prepare(dataset)
    columns = metadata['feature_columns']
    train, cal, selection, test = [frame[frame._split == s].copy() for s in ('train', 'calibration', 'selection', 'test')]
    if smoke:
        train, cal, selection, test, natural = [bound(p, 'target', 2000, 101) for p in (train, cal, selection, test, natural)]
    protocol = dict(dataset=dataset, experiment=VERSION, training_device='local Mac CPU', platform=platform.platform(),
        threads_per_estimator=4, seed=42, smoke=smoke, supervised_only=True, preparation=metadata,
        budgets=list(BUDGETS), default_budget=.01, data_sha256=metadata['data_sha256'],
        selection_rule='Reassess v3 incumbents and new feature candidates on identical calibration/selection rows. Maximize weakest chronological selection-block utility (65% AP + 35% mean recall at five FPR budgets minus FPR-excess penalty), minus 0.15 positive training/validation AP-above-random gap and 0.10 ROC-AUC gap. New candidates must retain >=95% incumbent validation AP, >=85% recall at 1% budget, and utility within 0.03. No test selection. Two forward training-period refits per selected family are diagnostic, not a fresh outer holdout or a champion selection criterion.',
        limitations='Previously viewed test periods and reused validation are exploratory. No guarantee of optimal fitting, future FPR, precision or review capacity. Wilson bounds are approximate one-sided calibration diagnostics, not future-period guarantees under drift or dependence. No deployment threshold is recommended without business constraints. PaySim training metrics are inverse-inclusion-weighted; unweighted enriched-training prevalence differs. Supplied anonymous feature availability cannot all be independently verified.',
        prediction_decimals=12, protected_datasets=['banksim', 'handbook', 'sparkov', 'ulb'])
    (folder / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    previous = evaluation_directory(dataset) / 'fit_balance_v3'
    prior_report = json.loads((previous / 'report.json').read_text())
    cache = folder / 'candidates'
    cache.mkdir(exist_ok=True)
    rows, trials, chosen, incumbent_records = [], [], {}, {}
    for old in prior_report['models']:
        family = old['model']
        original = joblib.load(previous / (family + '.joblib'))
        old_train = natural[natural._event_time >= 20] if 'recent_steps20plus' in old['variant'] else natural
        record = assess(original['model'], family, 'v3_incumbent_' + old['variant'], old_train, cal,
                        selection, original['feature_columns'])
        record.update(reused_checkpoint=True, columns=original['feature_columns'], window=None,
                      fit_info=dict(rows=len(old_train), fraud=int(old_train.target.sum())))
        incumbent_records[family] = copy.deepcopy(record)
        best, model = record, original['model']
        trials.append(copy.deepcopy(record))
        for label, candidate, window in candidate_models(dataset, family, train, smoke):
            saved = cache / (family + '_' + label + '.joblib')
            if saved.exists():
                bundle = joblib.load(saved)
                candidate, result = bundle['model'], bundle['record']
            else:
                fit = windowed(train, window)
                print('FIT', dataset, family, label, 'rows', len(fit), 'fraud', int(fit.target.sum()), flush=True)
                start = time.monotonic()
                candidate, info = fit_model(candidate, family, fit, columns)
                result = assess(candidate, family, label, fit, cal, selection, columns)
                result.update(columns=columns, window=window, fit_info=info, fit_seconds=time.monotonic() - start)
                joblib.dump(dict(model=candidate, record=result), saved, compress=3)
            result['passes_incumbent_floors'] = passes(result, record)
            trials.append(copy.deepcopy(result))
            (folder / 'trials.json').write_text(json.dumps(trials, indent=2))
            print(dataset, family, label, 'validationAP', round(result['validation']['pr_auc'], 4),
                  'gap', round(result['positive_pr_auc_gap'], 4), 'eligible', result['passes_incumbent_floors'], flush=True)
            if result['passes_incumbent_floors'] and result['fit_balance_score'] > best['fit_balance_score']:
                best, model = result, candidate
        forward_file = folder / (family + '_forward.json')
        if forward_file.exists():
            forward = json.loads(forward_file.read_text())
            if forward['variant'] != best['variant']:
                raise ValueError('Forward cache configuration mismatch; use a new experiment directory')
        else:
            fit_base = natural if best.get('reused_checkpoint') else train
            window = 15 if best.get('reused_checkpoint') and 'recent_steps20plus' in best['variant'] else best['window']
            print('FORWARD REFITS', dataset, family, best['variant'], flush=True)
            forward = dict(variant=best['variant'], folds=forward_refits(model, family, window, fit_base,
                                             natural, best['columns'], smoke))
            forward_file.write_text(json.dumps(forward, indent=2))
        best = copy.deepcopy(best)
        best['forward_validation'] = forward['folds']
        rows.append(best)
        chosen[family] = model
        print('SELECTED FAMILY', dataset, family, best['variant'], flush=True)
    incumbent_champion = incumbent_records[prior_report['winner']]
    eligible = [r for r in rows if passes(r, incumbent_champion)]
    champion = max([incumbent_champion, *eligible], key=lambda r: r['fit_balance_score'])
    winner = champion['model']
    if champion['variant'] != next(r['variant'] for r in rows if r['model'] == winner):
        forward = next(r['forward_validation'] for r in rows if r['model'] == winner)
        rows = [dict(copy.deepcopy(champion), forward_validation=forward) if r['model'] == winner else r for r in rows]
        chosen[winner] = joblib.load(previous / (winner + '.joblib'))['model']
    (folder / 'selection.json').write_text(json.dumps(dict(winner=winner, models=rows,
                                   incumbent_comparison=incumbent_records), indent=2))
    scores, calibration_scores, policy_rows = {}, {}, []
    for row in rows:
        family, model, feature_columns = row['model'], chosen[row['model']], row['columns']
        scores[family] = predict_scores(model, test[feature_columns])
        calibration_scores[family] = predict_scores(model, cal[feature_columns])
        selection_scores = predict_scores(model, selection[feature_columns])
        row['test_operating_points'] = {str(b): measure(test.target, scores[family], row['thresholds'][str(b)]) for b in BUDGETS}
        row['test'] = row['test_operating_points']['0.01']
        row['diagnostics']['test'] = diagnostic(test.target, scores[family])
        row['test_policies'] = {}
        for key, policy in row['policies'].items():
            for split, part, values in [('calibration', cal, calibration_scores[family]),
                                        ('validation', selection, selection_scores), ('test', test, scores[family])]:
                metrics = measure(part.target, values, policy['threshold'])
                metrics['alert_rate'] = float(np.mean(values >= policy['threshold']))
                policy_rows.append(dict(dataset=dataset, model=family, split=split, policy_key=key, **policy, **metrics))
                if split == 'test':
                    row['test_policies'][key] = metrics
        joblib.dump(dict(model=model, threshold=row['threshold'], thresholds=row['thresholds'],
                         policy_thresholds=row['policies'], selected_model=family, feature_columns=feature_columns,
                         dataset=dataset, variant=row['variant'], prediction_decimals=12,
                         history_assumptions=metadata['notes']), folder / (family + '.joblib'), compress=3)
    joblib.dump(joblib.load(folder / (winner + '.joblib')), folder / 'champion.joblib', compress=3)
    pd.DataFrame(policy_rows).to_csv(folder / 'threshold_policies.csv', index=False)
    pd.DataFrame([dict(model=r['model'], variant=r['variant'], threshold=r['threshold'],
                       fit_balance_score=r['fit_balance_score'], **{f'{s}_{k}': v for s in ('train', 'validation', 'test')
                       for k, v in r[s].items()}) for r in rows]).to_csv(folder / 'model_comparison.csv', index=False)
    pd.DataFrame(dict(target=test.target.to_numpy(), **scores)).to_csv(folder / 'test_predictions.csv', index=False)
    pd.DataFrame(dict(target=cal.target.to_numpy(), **calibration_scores)).to_csv(folder / 'calibration_predictions.csv', index=False)
    pd.DataFrame([dict(model=r['model'], **fold) for r in rows for fold in r['forward_validation']]).to_json(
        folder / 'forward_validation.json', orient='records', indent=2)
    plots(folder, rows, test.target, scores, {}, cal.target, calibration_scores)
    budget_plot(folder, rows)
    diagnostic_plot(folder, rows)
    bundle = joblib.load(folder / 'champion.joblib')
    (folder / 'example.json').write_text(test[bundle['feature_columns']].head(1).replace({np.nan: None}).to_json(orient='records', indent=2))
    (folder / 'report.json').write_text(json.dumps(dict(**protocol, winner=winner, total_candidates=len(trials), models=rows), indent=2))
    print('COMPLETE', dataset, winner, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=TARGETS, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.dataset, args.smoke)
