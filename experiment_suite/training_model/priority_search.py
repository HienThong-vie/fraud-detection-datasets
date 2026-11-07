"""Four supervised families selected for AP and recall at validation FPR budgets."""
from pathlib import Path
import argparse
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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from priority_data import prepare, ROOT, CAPS, project_directory, evaluation_directory
from supervised_search import templates, measure, plots
from supervised_support import native_frame

os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))

BUDGETS = (.005, .01, .05)
VERSION = 'priority_search_v2'


def predict_scores(model, frame):
    """Model score for the fraud class, rounded to 12 decimals so saved and reloaded models give identical
    results."""
    return np.round(np.asarray(model.predict_proba(frame)[:, 1], dtype='float64'), 12)


def fpr_threshold(y, scores, budget):
    """Lowest cutoff allowing at most floor(budget * n_negative) calibration errors."""
    negatives = np.sort(np.asarray(scores)[np.asarray(y) == 0])
    if not len(negatives) or not 0 <= budget < 1:
        raise ValueError('Calibration negatives and a budget in [0, 1) are required')
    permitted = int(np.floor(budget * len(negatives) + 1e-10))
    return float(np.nextafter(negatives[len(negatives) - permitted - 1], np.inf))


def selection_blocks(frame):
    """Split the selection period into two time-ordered halves; BAF has month-level time only, so month 5
    stays one block."""
    times = np.sort(frame._event_time.unique())
    if len(times) < 2:
        return [frame]
    boundary = times[max(1, len(times) // 2) - 1]
    blocks = [frame[frame._event_time <= boundary], frame[frame._event_time > boundary]]
    # Tiny positive counts cannot support meaningful separate AP estimates.
    return blocks if all(part.target.nunique() == 2 for part in blocks) else [frame]


def priority_utility(ap, operating):
    """Utility of one block: 0.65 x AP + 0.35 x mean recall over the budgets - 0.05 x mean relative FPR
    overshoot."""
    recalls = [m['recall'] for m in operating.values()]
    excess = [max(0., m['fpr'] - float(b)) / float(b) for b, m in operating.items()]
    return .65 * ap + .35 * float(np.mean(recalls)) - .05 * float(np.mean(excess))


def assess_trial(model, family, variant, train, calibration, selection, blocks, columns, fit_seconds):
    """Experiment v2 (superseded by feature_policy_search.assess): score one candidate on calibration and
    selection rows."""
    if family in ('random_forest', 'extra_trees'):
        model.steps[-1][1].set_params(n_jobs=1)
    cal = predict_scores(model, calibration[columns])
    thresholds = {str(b): fpr_threshold(calibration.target, cal, b) for b in BUDGETS}
    scores = predict_scores(model, selection[columns])
    training = measure(train.target, predict_scores(model, train[columns]), thresholds['0.01'])
    validation = measure(selection.target, scores, thresholds['0.01'])
    operating = {str(b): measure(selection.target, scores, thresholds[str(b)]) for b in BUDGETS}
    cal_metrics = {str(b): measure(calibration.target, cal, thresholds[str(b)]) for b in BUDGETS}
    block_rows = []
    for part in blocks:
        values = predict_scores(model, part[columns])
        metrics = {str(b): measure(part.target, values, thresholds[str(b)]) for b in BUDGETS}
        ap = metrics['0.01']['pr_auc']
        block_rows.append(dict(ap=ap, operating=metrics, utility=priority_utility(ap, metrics)))
    gap = max(0., training['pr_auc'] - validation['pr_auc'])
    utility = min(block['utility'] for block in block_rows) - .10 * gap
    return dict(model=family, variant=variant, threshold=thresholds['0.01'],
                thresholds=thresholds, selection_utility=utility, train=training,
                validation=validation, validation_operating_points=operating,
                calibration_operating_points=cal_metrics, validation_blocks=block_rows,
                positive_pr_auc_gap=gap, training_rows=len(train), fit_seconds=fit_seconds,
                prediction_decimals=12)


def candidates(dataset, columns, categorical, train, smoke):
    """Experiment v2: candidate pipelines of the four families."""
    base = templates(columns, categorical)
    preprocessing = base['logistic_regression'][0][1].steps[0][1]
    def wrapped(model):
        """Experiment v2: put a classifier behind the shared preprocessing."""
        return Pipeline([('preprocessor', clone(preprocessing)), ('classifier', model)])
    families = {
        'logistic_regression': [(f'C{c}', wrapped(LogisticRegression(C=c, class_weight='balanced',
                                                max_iter=1600, random_state=42))) for c in (.01, .1)],
        'random_forest': [], 'extra_trees': []}
    for name, kind in [('random_forest', RandomForestClassifier), ('extra_trees', ExtraTreesClassifier)]:
        for depth, leaf in [(6, 40), (10, 20)]:
            families[name].append((f'd{depth}_leaf{leaf}', wrapped(kind(
                n_estimators=160, max_depth=depth, min_samples_leaf=leaf, max_features=.7,
                class_weight='balanced', n_jobs=4, random_state=42))))
    if dataset in ('baf', 'sparkov', 'banksim'):
        from catboost import CatBoostClassifier
        families['catboost'] = []
        for depth, weights in [(4, None), (6, 'SqrtBalanced')]:
            estimator = CatBoostClassifier(iterations=600, depth=depth, learning_rate=.035,
                                          l2_leaf_reg=25, auto_class_weights=weights,
                                          loss_function='Logloss', random_seed=42, thread_count=4,
                                          allow_writing_files=False, verbose=False)
            transform = FunctionTransformer(native_frame, kw_args=dict(columns=columns,
                                            categorical=categorical), validate=False)
            families['catboost'].append((f'd{depth}_{weights}', Pipeline([
                ('prepare', transform), ('classifier', estimator)])))
    else:
        from xgboost import XGBClassifier
        weight = (train.target == 0).sum() / max((train.target == 1).sum(), 1)
        families['xgboost'] = []
        for depth, power in [(3, 0), (5, .5)]:
            families['xgboost'].append((f'd{depth}_weightpower{power}', wrapped(XGBClassifier(
                n_estimators=600, max_depth=depth, learning_rate=.035, min_child_weight=30,
                reg_lambda=25, reg_alpha=1., scale_pos_weight=float(weight ** power),
                subsample=.8, colsample_bytree=.8, objective='binary:logistic', eval_metric='aucpr',
                tree_method='hist', n_jobs=4, random_state=42))))
    for family in families:
        bundle = evaluation_directory(dataset) / 'supervised_search_v1' / (family + '.joblib')
        families[family].append(('v1_setting_retrained', clone(joblib.load(bundle)['model'])))
        for _, model in families[family]:
            estimator = model.steps[-1][1]
            if hasattr(estimator, 'n_jobs') and not isinstance(estimator, LogisticRegression):
                estimator.set_params(n_jobs=4)
            if family == 'catboost':
                estimator.set_params(thread_count=4)
            if smoke:
                if hasattr(estimator, 'n_estimators'):
                    estimator.set_params(n_estimators=5)
                if family == 'catboost':
                    estimator.set_params(iterations=5)
    return families


def budget_plot(out, rows):
    """Plot recall and actual FPR at every budget for each family (also used by v4)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, budget in zip(axes, BUDGETS):
        values = [r['test_operating_points'][str(budget)] for r in rows]
        bars = axis.bar([r['model'] for r in rows], [m['recall'] for m in values])
        axis.bar_label(bars, labels=[f"R={m['recall']:.1%}\nFPR={m['fpr']:.2%}" for m in values], fontsize=8)
        axis.set_ylim(0, 1.18)
        axis.set_title(f'Validation FPR budget {budget:.1%}')
        axis.tick_params(axis='x', rotation=25)
        axis.set_ylabel('Test recall; labels show actual test FPR')
    figure.tight_layout()
    figure.savefig(out / 'recall_at_fpr_budgets.png', dpi=160)
    plt.close(figure)


def _run(dataset, smoke=False):
    """Experiment priority_search_v2, kept for provenance; the active version is recorded in configs/final_models.json."""
    directory = project_directory(dataset)
    sys.path.insert(0, str(directory / 'src'))
    os.chdir(directory)
    out = evaluation_directory(dataset) / ('priority_search_smoke' if smoke else VERSION)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'report.json').exists():
        print('ALREADY COMPLETE', dataset, flush=True)
        return json.loads((out / 'report.json').read_text())
    frame, preparation = prepare(dataset, smoke)
    columns = preparation['feature_columns']
    train, calibration, selection, test = [frame[frame._split == name] for name in
                                          ('train', 'calibration', 'selection', 'test')]
    blocks = selection_blocks(selection)
    categorical = [c for c in columns if not pd.api.types.is_numeric_dtype(train[c])]
    protocol = dict(dataset=dataset, experiment=VERSION, training_device='local Mac CPU',
                    platform=platform.platform(), threads_per_estimator=4, supervised_only=True,
                    stable_scoring=True, prediction_decimals=12, forest_prediction_threads=1,
                    smoke=smoke, budgets=list(BUDGETS), default_budget=.01, seed=42,
                    preparation=preparation,
                    data_sha256=hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
                    selection_rule='Maximize weakest chronological-block utility: 65% AP + 35% mean recall across three calibrated FPR budgets, minus selection FPR excess penalty and 0.10 positive training/selection AP gap. No ROC-AUC or accuracy weight.',
                    limitations='Previously inspected test periods are exploratory, not a fresh final holdout. Validation FPR budgets are not guarantees on test or deployment. AP is average_precision_score, not trapezoidal integration. Historical label features, where used, require operational availability assumptions.')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    families = candidates(dataset, columns, categorical, train, smoke)
    trials = []; rows = []; selected = {}; cal_scores = {}; valid_scores = {}
    candidate_dir = out / 'candidates'
    candidate_dir.mkdir(exist_ok=True)
    for family, options in families.items():
        best = None
        for index, (variant, model) in enumerate(options):
            fit_train = train
            if dataset == 'elliptic' and index == 0:
                fit_train = train[train._event_time >= 20]
                variant += '_recent_steps20plus'
            saved = candidate_dir / f'{family}_{index}.joblib'
            if saved.exists():
                cached = joblib.load(saved)
                model, record = cached['model'], cached['record']
                fit_seconds = record['fit_seconds']
            else:
                started = time.monotonic()
                print('FIT', dataset, family, variant, 'rows', len(fit_train), flush=True)
                kwargs = {model.steps[-1][0] + '__cat_features': categorical} if family == 'catboost' else {}
                model.fit(fit_train[columns], fit_train.target, **kwargs)
                fit_seconds = time.monotonic() - started
                record = None
            if record is None or record.get('prediction_decimals') != 12:
                record = assess_trial(model, family, variant, fit_train, calibration, selection,
                                      blocks, columns, fit_seconds)
                joblib.dump(dict(model=model, record=record), saved, compress=3)
            trials.append(record)
            (out / 'trials.json').write_text(json.dumps(trials, indent=2))
            print(dataset, family, variant, 'AP', round(record['validation']['pr_auc'], 4),
                  'recall@1%', round(record['validation']['recall'], 4),
                  'utility', round(record['selection_utility'], 4), flush=True)
            if best is None or record['selection_utility'] > best['selection_utility']:
                best = record
                selected[family] = model
        rows.append(best)
        model = selected[family]
        cal_scores[family] = predict_scores(model, calibration[columns])
        valid_scores[family] = predict_scores(model, selection[columns])
    winner = max(rows, key=lambda row: row['selection_utility'])['model']
    # Selection is frozen before any test scoring, including all three operating thresholds.
    (out / 'selection.json').write_text(json.dumps(dict(winner=winner, models=rows), indent=2))
    scores = {}; operating_rows = []
    for row in rows:
        family = row['model']; model = selected[family]
        scores[family] = predict_scores(model, test[columns])
        row['test_operating_points'] = {str(b): measure(test.target, scores[family], row['thresholds'][str(b)]) for b in BUDGETS}
        row['test'] = row['test_operating_points']['0.01']
        for budget in BUDGETS:
            operating_rows.append(dict(model=family, validation_fpr_budget=budget,
                                       threshold=row['thresholds'][str(budget)],
                                       **row['test_operating_points'][str(budget)]))
        bundle = dict(model=model, threshold=row['threshold'], thresholds=row['thresholds'],
                      prediction_decimals=12,
                      feature_columns=columns, selected_model=family, dataset=dataset, variant=row['variant'],
                      history_assumptions=preparation['notes'])
        joblib.dump(bundle, out / (family + '.joblib'), compress=3)
    joblib.dump(joblib.load(out / (winner + '.joblib')), out / 'champion.joblib', compress=3)
    pd.DataFrame(operating_rows).to_csv(out / 'operating_points.csv', index=False)
    pd.DataFrame([dict(model=r['model'], variant=r['variant'], threshold=r['threshold'],
                       selection_utility=r['selection_utility'], **{f'{s}_{k}': v for s in ('train', 'validation', 'test') for k, v in r[s].items()})
                  for r in rows]).to_csv(out / 'model_comparison.csv', index=False)
    pd.DataFrame(dict(target=test.target.to_numpy(), **scores)).to_csv(out / 'test_predictions.csv', index=False)
    pd.DataFrame(dict(target=calibration.target.to_numpy(), **cal_scores)).to_csv(out / 'calibration_predictions.csv', index=False)
    plots(out, rows, test.target, scores, valid_scores, calibration.target, cal_scores)
    budget_plot(out, rows)
    report = dict(**protocol, winner=winner, total_fits=len(trials), models=rows)
    (out / 'example.json').write_text(test[columns].head(1).replace({np.nan: None}).to_json(orient='records', indent=2))
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    if not smoke and (ROOT / 'configs/datasets.json').exists():
        import subprocess
        subprocess.run([sys.executable, str(ROOT / 'utils/verify_priority.py'), '--dataset', dataset],
                       cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / 'utils/summarize_priority.py')], cwd=ROOT, check=True)
    print('COMPLETE', dataset, winner, next(r['test'] for r in rows if r['model'] == winner), flush=True)
    return report


def run(dataset, smoke=False):
    """Run experiment v2 once per dataset, guarded by a file lock."""
    import fcntl
    directory = evaluation_directory(dataset) / ('priority_search_smoke' if smoke else VERSION)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _run(dataset, smoke)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(CAPS))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.dataset, args.smoke)
