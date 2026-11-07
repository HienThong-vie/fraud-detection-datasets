"""Local CPU search over exactly four supervised model families per dataset."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             confusion_matrix, PrecisionRecallDisplay, RocCurveDisplay,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from supervised_support import native_frame

ROOT=Path(__file__).resolve().parents[1]
FILES={'baf':('base_model_data.csv.gz','fraud_bool'),'ieee':('ieee_model_data.csv.gz','isFraud'),
       'paysim':('paysim_model_data.csv.gz','isFraud'),'handbook':('handbook_model_data.csv.gz','TX_FRAUD'),
       **{k:('model_data.csv.gz','target') for k in ['sparkov','banksim','ulb','elliptic']}}

def measure(y, scores, threshold):
    tn,fp,fn,tp=confusion_matrix(y,scores>=threshold,labels=[0,1]).ravel()
    precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1)
    return dict(precision=float(precision),recall=float(recall),f1=float(2*precision*recall/max(precision+recall,1e-12)),
                pr_auc=float(average_precision_score(y,scores)),roc_auc=float(roc_auc_score(y,scores)),
                fpr=float(fp/max(fp+tn,1)),accuracy=float((tp+tn)/len(y)),
                tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def choose_threshold(y,scores):
    precision,recall,threshold=precision_recall_curve(y,scores)
    precision,recall=precision[:-1],recall[:-1]
    negatives=np.sort(scores[np.asarray(y)==0])
    fpr=(len(negatives)-np.searchsorted(negatives,threshold,side='left'))/len(negatives)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    eligible=np.flatnonzero(fpr<=.05)
    if not len(eligible):
        return float(np.nextafter(scores.max(),np.inf))
    near=eligible[f1[eligible]>=.95*f1[eligible].max()]
    return float(threshold[near[np.argmax(np.minimum(precision[near],recall[near]))]])

def utility(metrics):
    return .45*metrics['f1']+.15*min(metrics['precision'],metrics['recall'])+.2*metrics['pr_auc']+.2*metrics['roc_auc']

def templates(columns,categorical):
    numeric=[c for c in columns if c not in categorical]
    preprocessor=ColumnTransformer([
        ('numeric',Pipeline([('missing',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),numeric),
        ('categorical',Pipeline([('missing',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),('encode',OneHotEncoder(handle_unknown='ignore',min_frequency=5))]),categorical)
    ],sparse_threshold=1)
    def wrapped(estimator):
        return Pipeline([('preprocessor',clone(preprocessor)),('classifier',estimator)])
    families={
        'logistic_regression':[(f'C{c}',wrapped(LogisticRegression(C=c,class_weight='balanced',max_iter=1200,random_state=42))) for c in (.1,1)],
        'random_forest':[], 'extra_trees':[]}
    for name,kind in [('random_forest',RandomForestClassifier),('extra_trees',ExtraTreesClassifier)]:
        for depth,leaf in [(8,16),(12,8)]:
            families[name].append((f'd{depth}_leaf{leaf}',wrapped(kind(n_estimators=140,max_depth=depth,min_samples_leaf=leaf,class_weight='balanced',n_jobs=4,random_state=42))))
    return families

def plots(out,rows,test_y,test_scores,validation_scores,cal_y,cal_scores):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=['pr_auc','roc_auc','precision','recall','f1','fpr']
    figure,axes=plt.subplots(2,3,figsize=(15,9))
    names=[r['model'] for r in rows]
    for axis,key in zip(axes.flat,keys):
        values=[r['test'][key] for r in rows]
        bars=axis.bar(names,values,color=['#2563eb','#0f766e','#7c3aed','#db7026'])
        axis.bar_label(bars,labels=[f'{value:.4f}' for value in values],padding=3,fontsize=9)
        axis.set_title(key.upper()+' (test)');axis.set_ylim(0,max(.01,max(values)*1.25) if key=='fpr' else 1);axis.tick_params(axis='x',rotation=25)
    figure.suptitle('Four supervised classifiers — frozen validation-selected settings')
    figure.tight_layout();figure.savefig(out/'model_comparison.png',dpi=160);plt.close(figure)
    for filename,kind in [('precision_recall_curves.png',PrecisionRecallDisplay),('roc_curves.png',RocCurveDisplay)]:
        figure,axis=plt.subplots(figsize=(8,6))
        for name,scores in test_scores.items():
            kind.from_predictions(test_y,scores,name=name,ax=axis)
        figure.tight_layout();figure.savefig(out/filename,dpi=160);plt.close(figure)
    figure,axes=plt.subplots(2,2,figsize=(10,9))
    for axis,row in zip(axes.flat,rows):
        ConfusionMatrixDisplay.from_predictions(test_y,test_scores[row['model']]>=row['threshold'],ax=axis,colorbar=False,values_format='d')
        axis.set_title(row['model'])
    figure.tight_layout();figure.savefig(out/'confusion_matrices.png',dpi=160);plt.close(figure)
    figure,axis=plt.subplots(figsize=(10,5));x=np.arange(4);width=.25
    for offset,split in [(-1,'train'),(0,'validation'),(1,'test')]:
        axis.bar(x+offset*width,[r[split]['pr_auc'] for r in rows],width,label=split)
    axis.set_xticks(x,names);axis.set_ylabel('PR-AUC');axis.set_ylim(0,1);axis.legend();axis.set_title('Training, selection-validation and temporal test gaps')
    figure.tight_layout();figure.savefig(out/'generalization_gap.png',dpi=160);plt.close(figure)
    figure,axes=plt.subplots(2,2,figsize=(11,8))
    for axis,row in zip(axes.flat,rows):
        p,r,t=precision_recall_curve(cal_y,cal_scores[row['model']]);f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-12)
        axis.plot(t,p[:-1],label='precision');axis.plot(t,r[:-1],label='recall');axis.plot(t,f,label='F1');axis.axvline(row['threshold'],color='black',linestyle='--');axis.set_title(row['model']+' (calibration only)');axis.legend()
    figure.tight_layout();figure.savefig(out/'threshold_tradeoffs.png',dpi=160);plt.close(figure)

def run(dataset,project=None,smoke=False,include_prior=False,plots_only=False):
    directory=Path(project).resolve() if project else (ROOT/'pipelines'/dataset if (ROOT/'pipelines'/dataset).exists() else ROOT)
    os.chdir(directory);sys.path.insert(0,str(directory/'src'))
    filename,target=FILES[dataset];prepared=directory/'preprocessed_df'/filename
    if not prepared.exists():
        prepared=directory/'.local_datasets'/dataset/'prepared'/filename
    out=directory/'outputs'/('supervised_search_smoke' if smoke else 'supervised_search_v1')
    out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(prepared)
    previous=joblib.load(directory/'artifacts/production_model.joblib')
    columns=previous['feature_columns']
    train=frame[frame._split=='train'];val=frame[frame._split=='validation'];test=frame[frame._split=='test']
    if smoke:
        train=train.groupby(target,group_keys=False).head(60);val=val.groupby(target,group_keys=False).head(45);test=test.groupby(target,group_keys=False).head(30)
    calibration,selection=train_test_split(val,test_size=.6,random_state=912,stratify=val[target])
    part1,part2=train_test_split(selection,test_size=.5,random_state=913,stratify=selection[target])
    categorical=[c for c in columns if not pd.api.types.is_numeric_dtype(train[c])]
    if plots_only:
        report=json.loads((out/'report.json').read_text())
        predictions=pd.read_csv(out/'test_predictions.csv')
        scores={r['model']:predictions[r['model']].to_numpy() for r in report['models']}
        calibration_scores={}
        for row in report['models']:
            model=joblib.load(out/(row['model']+'.joblib'))['model']
            calibration_scores[row['model']]=model.predict_proba(calibration[columns])[:,1]
        plots(out,report['models'],test[target],scores,{},calibration[target],calibration_scores)
        print('CHARTS',dataset,flush=True)
        return report
    families=templates(columns,categorical)
    if dataset in ('baf','sparkov','banksim'):
        from catboost import CatBoostClassifier
        families['catboost']=[]
        for depth,weights in [(3,'Balanced'),(5,'SqrtBalanced')]:
            estimator=CatBoostClassifier(iterations=350,depth=depth,learning_rate=.05,l2_leaf_reg=10,auto_class_weights=weights,loss_function='Logloss',random_seed=42,thread_count=4,allow_writing_files=False,verbose=False)
            prepare=FunctionTransformer(native_frame,kw_args={'columns':list(columns),'categorical':categorical},validate=False)
            families['catboost'].append((f'd{depth}_{weights}',Pipeline([('prepare',prepare),('classifier',estimator)])))
    else:
        from xgboost import XGBClassifier
        weight=float((train[target]==0).sum()/max((train[target]==1).sum(),1))
        base=clone(families['random_forest'][0][1])
        families['xgboost']=[]
        for depth,power in [(3,1),(5,.5)]:
            estimator=XGBClassifier(n_estimators=350,max_depth=depth,learning_rate=.04,min_child_weight=15,reg_lambda=10,reg_alpha=.5,scale_pos_weight=weight**power,subsample=.8,colsample_bytree=.8,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',n_jobs=4,random_state=42)
            model=clone(base).set_params(classifier=estimator)
            families['xgboost'].append((f'd{depth}_weightpower{power}',model))
    evidence=dict(dataset=dataset,platform=platform.platform(),machine=platform.machine(),training_device='local Mac CPU',threads_per_estimator=4,prepared_sha256=hashlib.sha256(prepared.read_bytes()).hexdigest(),seed=42,calibration_seed=912,selection_split_seed=913,rows={k:len(v) for k,v in [('train',train),('calibration',calibration),('selection_1',part1),('selection_2',part2),('test',test)]},supervised_only=True,families=list(families),trials_per_family=2,smoke=smoke)
    (out/'protocol.json').write_text(json.dumps(evidence,indent=2))
    rows=[];trials=[];selected={};test_scores={};cal_scores={};validation_scores={}
    prior_report=None
    if include_prior:
        if not (out/'report.json').exists():
            raise ValueError('Run the prespecified search first, then screen the prior configuration')
        prior_report=json.loads((out/'report.json').read_text())
        trials=json.loads((out/'trials.json').read_text())
        previous_pipeline=previous.get('model',previous.get('pipeline'))
        class_name=type(previous_pipeline.steps[-1][1]).__name__
        baseline_family={'LogisticRegression':'logistic_regression','RandomForestClassifier':'random_forest','ExtraTreesClassifier':'extra_trees','CatBoostClassifier':'catboost','XGBClassifier':'xgboost'}[class_name]
        families={name:[] for name in families}
        families[baseline_family]=[('prior_accepted_configuration',clone(previous_pipeline))]
        for row in prior_report['models']:
            family=row['model'];selected[family]=joblib.load(out/(family+'.joblib'))['model']
            cal_scores[family]=selected[family].predict_proba(calibration[columns])[:,1]
            validation_scores[family]=selected[family].predict_proba(selection[columns])[:,1]
    for family,options in families.items():
        best=next((r for r in prior_report['models'] if r['model']==family),None) if prior_report else None
        for variant,model in options:
            started=time.monotonic()
            estimator_name=model.steps[-1][0]
            if smoke and hasattr(model.steps[-1][1],'n_estimators'):
                model.set_params(**{estimator_name+'__n_estimators':5})
            if smoke and family=='catboost':model.set_params(**{estimator_name+'__iterations':5})
            params={estimator_name+'__cat_features':categorical} if family=='catboost' else {}
            model.fit(train[columns],train[target],**params)
            cal=model.predict_proba(calibration[columns])[:,1]
            threshold=choose_threshold(calibration[target],cal)
            scores=model.predict_proba(selection[columns])[:,1]
            valid=measure(selection[target],scores,threshold)
            training=measure(train[target],model.predict_proba(train[columns])[:,1],threshold)
            blocks=[measure(part[target],model.predict_proba(part[columns])[:,1],threshold) for part in (part1,part2)]
            gap=max(0,training['pr_auc']-valid['pr_auc'])
            score=min(utility(b) for b in blocks)-.03*gap
            record=dict(model=family,variant=variant,threshold=threshold,selection_utility=score,train=training,validation=valid,validation_blocks=blocks,positive_pr_auc_gap=gap,fit_seconds=time.monotonic()-started)
            trials.append(record)
            (out/'trials.json').write_text(json.dumps(trials,indent=2))
            print(dataset,family,variant,'validation F1',round(valid['f1'],4),'PR-AUC',round(valid['pr_auc'],4),'utility',round(score,4),flush=True)
            if best is None or score>best['selection_utility']:
                best=record;selected[family]=model;cal_scores[family]=cal;validation_scores[family]=scores
        rows.append(best)
    winner=max(rows,key=lambda row:row['selection_utility'])['model']
    # Freeze each family's candidate and overall winner before accessing test scores.
    (out/'selection.json').write_text(json.dumps({'winner':winner,'models':rows},indent=2))
    for row in rows:
        family=row['model'];model=selected[family]
        test_scores[family]=model.predict_proba(test[columns])[:,1]
        row['test']=measure(test[target],test_scores[family],row['threshold'])
        bundle=dict(model=model,threshold=row['threshold'],feature_columns=list(columns),selected_model=family,dataset=dataset,variant=row['variant'])
        joblib.dump(bundle,out/(family+'.joblib'),compress=3)
    joblib.dump(joblib.load(out/(winner+'.joblib')),out/'champion.joblib',compress=3)
    pd.DataFrame([{'model':r['model'],'variant':r['variant'],'threshold':r['threshold'],'selection_utility':r['selection_utility'],**{f'{split}_{k}':v for split in ('train','validation','test') for k,v in r[split].items()}} for r in rows]).to_csv(out/'model_comparison.csv',index=False)
    pd.DataFrame({'target':test[target].to_numpy(),**test_scores}).to_csv(out/'test_predictions.csv',index=False)
    evidence['prior_configuration_screened']=include_prior
    evidence['total_fits']=len(trials)
    (out/'protocol.json').write_text(json.dumps(evidence,indent=2))
    result=dict(**evidence,winner=winner,models=rows,selection_rule='Maximize worst utility across two selection-validation subsets minus 0.03 * positive train/validation PR-AUC gap. Utility: 45% F1, 15% min(precision,recall), 20% PR-AUC, 20% ROC-AUC. Calibration threshold within 95% of best F1, favor balanced precision/recall, calibration FPR <= 5%.',limitations='Existing temporal test periods have already been inspected. Exploratory, not an untouched final evaluation. Validation subsets are stratified because exported tables do not consistently retain event timestamps. No test metric chooses a candidate. BAF is Base only; variants retain original reports. No production checkpoint replacement.')
    (out/'report.json').write_text(json.dumps(result,indent=2))
    plots(out,rows,test[target],test_scores,validation_scores,calibration[target],cal_scores)
    print('COMPLETE',dataset,winner,json.dumps(next(r['test'] for r in rows if r['model']==winner)),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True,choices=list(FILES));parser.add_argument('--project-root');parser.add_argument('--smoke',action='store_true');parser.add_argument('--include-prior',action='store_true');parser.add_argument('--plots-only',action='store_true');args=parser.parse_args();run(args.dataset,args.project_root,args.smoke,args.include_prior,args.plots_only)
