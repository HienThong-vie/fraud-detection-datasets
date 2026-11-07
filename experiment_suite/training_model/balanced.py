"""Validation-only model/threshold comparison; existing production files stay intact."""
from pathlib import Path
import argparse
import importlib
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
REPOS = ['ai4cyber-fraud', 'ieee-fraud-ml', 'paysim-fraud-ml', 'fraud-handbook-ml', 'sparkov-fraud-ml', 'banksim-fraud-ml', 'ulb-creditcard-fraud-ml', 'elliptic-bitcoin-aml-ml']

def metrics(y, s, threshold):
    tn, fp, fn, tp = confusion_matrix(y, s >= threshold, labels=[0, 1]).ravel()
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(pr_auc=float(average_precision_score(y, s)), roc_auc=float(roc_auc_score(y, s)), precision=float(p), recall=float(r), f1=float(2*p*r/max(p+r, 1e-12)), fpr=float(fp/max(fp+tn, 1)))

def threshold_for(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1], 1e-12)
    # Both constraints are computed using threshold-calibration rows only.
    negatives = np.sort(s[np.asarray(y) == 0])
    fpr = (len(negatives)-np.searchsorted(negatives, t, side='left'))/len(negatives)
    eligible = np.flatnonzero(fpr <= .05)
    if not len(eligible):
        return float(np.nextafter(s.max(), np.inf))
    near = eligible[f[eligible] >= .98*f[eligible].max()]
    best = near[np.argmax(np.minimum(p[near], r[near]))]
    return float(t[best])

def run(repo):
    registry = json.loads((ROOT/'configs/datasets.json').read_text())
    dataset = next(key for key, value in registry.items() if value['repository'] == repo)
    directory = ROOT/'pipelines'/dataset
    os.chdir(directory)
    sys.path.insert(0, str(directory/'src'))
    package = {'ai4cyber-fraud':'ai4cyber', 'ieee-fraud-ml':'ieee_fraud', 'paysim-fraud-ml':'paysim_fraud', 'fraud-handbook-ml':'handbook_fraud'}.get(repo, 'fraudbench')
    training = importlib.import_module(package+'.training')
    data = importlib.import_module(package+'.data')
    cfg = json.loads((directory/'configs/training.json').read_text())
    filename = {'ai4cyber-fraud':'base_model_data.csv.gz','ieee-fraud-ml':'ieee_model_data.csv.gz','paysim-fraud-ml':'paysim_model_data.csv.gz','fraud-handbook-ml':'handbook_model_data.csv.gz'}.get(repo,'model_data.csv.gz')
    frame = pd.read_csv(directory/'preprocessed_df'/filename)
    target = {'ai4cyber-fraud':'fraud_bool','ieee-fraud-ml':'isFraud','paysim-fraud-ml':'isFraud','fraud-handbook-ml':'TX_FRAUD'}.get(repo,'target')
    train = frame[frame._split == 'train']; val = frame[frame._split == 'validation']; test = frame[frame._split == 'test']
    weight = float((train[target] == 0).sum()/max((train[target] == 1).sum(), 1))
    cats = []
    if package == 'ai4cyber':
        config = training.TrainingConfig.from_json(directory/'configs/training.json')
        temporal = data.make_temporal_data(data.replace_missing_sentinels(data.load_baf_csv(directory/'data/raw/Base.csv', config)), config)
        columns = temporal.feature_columns; cats = temporal.categorical_columns
        candidates = {n: training.Pipeline([('preprocessor', data.build_preprocessor(temporal)), ('model', m)]) for n,m in training.build_models(config).items() if n != 'sgd_logistic'}
        candidates['catboost'] = training.build_catboost_pipeline(config, temporal)
    else:
        columns = data.MODEL_COLUMNS if hasattr(data,'MODEL_COLUMNS') else training.MODEL_FEATURES
        if package == 'ieee_fraud':
            config = training.TrainingConfig.from_json(directory/'configs/training.json')
        else:
            config = cfg
        if package == 'fraudbench' and 'catboost_iterations' in cfg:
            candidates = training.candidate_models(config); cats = data.CATEGORICAL_COLUMNS
        else:
            candidates = training.candidate_models(config, weight)
    boost = 'catboost' if 'catboost' in candidates else 'xgboost'
    step = 'model' if package in ('ai4cyber','ieee_fraud') else 'classifier'
    for depth in (3, 5):
        model = clone(candidates[boost]); estimator = model.named_steps[step]
        if boost == 'catboost':
            estimator.set_params(iterations=350, depth=depth, learning_rate=.05, l2_leaf_reg=10, thread_count=4)
        else:
            estimator.set_params(n_estimators=350, max_depth=depth, learning_rate=.04, min_child_weight=15, reg_lambda=10, reg_alpha=.5, subsample=.8, colsample_bytree=.8, n_jobs=4)
        candidates[f'{boost}_regularized_d{depth}'] = model
    # Processed validation tables do not always preserve timestamps. A fixed stratified
    # split avoids pretending their row order is chronological. Test remains temporal.
    calibration, selection = train_test_split(val, test_size=.5, random_state=731, stratify=val[target])
    rows, fitted = [], {}
    out = directory/'outputs/balanced_experiment'; out.mkdir(exist_ok=True)
    for name, model in candidates.items():
        kwargs = {step+'__cat_features':cats} if name.startswith('catboost') else {}
        model.fit(train[columns], train[target], **kwargs)
        threshold = threshold_for(calibration[target], model.predict_proba(calibration[columns])[:,1])
        m = metrics(selection[target],model.predict_proba(selection[columns])[:,1],threshold)
        tr = metrics(train[target],model.predict_proba(train[columns])[:,1],threshold)
        gap = max(0,tr['pr_auc']-m['pr_auc'])
        # Geometric mean rewards balanced precision/recall through F1 and ranking
        # through both AUCs; the gap penalty discourages memorization.
        score = (max(m['f1'],1e-12)**.5 * m['pr_auc']**.25 * m['roc_auc']**.25)-.1*gap
        rows.append(dict(model=name, threshold=threshold, selection_score=score, train_pr_auc=tr['pr_auc'], pr_auc_gap=gap, **m))
        fitted[name]=model
        print(repo,name,json.dumps(rows[-1]),flush=True)
        pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    best = max(rows,key=lambda r:r['selection_score'])
    model = fitted[best['model']]
    s = model.predict_proba(test[columns])[:,1]
    report = dict(repository=repo, selection=best, test=metrics(test[target],s,best['threshold']), limitations='Existing test periods have previously been inspected; this is an exploratory comparison, not a new untouched holdout. Threshold calibration and model selection use separate stratified halves of validation. No production replacement or refit.', rule='50% F1, 25% PR-AUC, 25% ROC-AUC geometric mean minus 0.1 times positive train/selection PR-AUC gap; threshold within 98% of calibration-best F1, prefer balanced precision/recall, calibration FPR <= 5%.')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    joblib.dump(dict(model=model,threshold=best['threshold'],feature_columns=list(columns),selected_model=best['model']),out/'model.joblib')
    pd.DataFrame(dict(target=test[target],score=s)).to_csv(out/'test_predictions.csv',index=False)
    print('COMPLETE',repo,json.dumps(report['test']),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('repo',choices=REPOS); args=parser.parse_args(); run(args.repo)
