"""One local API and dashboard for eight independent fraud pipelines."""
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal
from utils.runtime import ROOT, datasets, predict, execute
from utils.experiments import fitting_ready, features_ready, test_shift_warning

app=FastAPI(title='Fraud Detection Suite',version='1.0.0')

class PredictionRequest(BaseModel):
    records: list[dict] = Field(min_length=1,max_length=1000)
    variant: Literal['selected','production','balanced','supervised','priority','fitting','features'] = 'selected'
    fpr_budget: Literal[.005,.01,.02,.03,.05] = .05
    threshold_policy: Literal['point','ucb95'] = 'point'
    review_capacity: Literal[.005,.01,.02,.05] | None = None

@app.get('/')
def dashboard():
    return FileResponse(ROOT/'dashboard/index.html')

@app.get('/datasets')
def list_datasets():
    return datasets()

def valid(dataset):
    if dataset not in datasets():
        raise HTTPException(404,'Unknown dataset')

@app.get('/datasets/{dataset}')
def detail(dataset: str):
    valid(dataset)
    output=ROOT/'evaluation'/dataset
    result=dict(id=dataset,**datasets()[dataset])
    frozen=ROOT/'configs/final_models.json'
    result['selected']=json.loads(frozen.read_text())['datasets'][dataset] if frozen.exists() else None
    result['metrics']=json.loads((output/('base/metrics.json' if dataset=='baf' else 'metrics.json')).read_text())
    result['balanced']=json.loads((output/'balanced_experiment/report.json').read_text())
    result['example']=json.loads((ROOT/'examples'/f'{dataset}.json').read_text())
    result['charts']=[str(p.relative_to(output)) for p in output.rglob('*.png') if 'demo' not in str(p) and 'smoke' not in str(p)]
    latest=output/'supervised_search_v1/report.json'
    result['supervised']=json.loads(latest.read_text()) if latest.exists() else None
    priority=output/'priority_search_v2/report.json'
    result['priority']=json.loads(priority.read_text()) if priority.exists() else None
    fitting=output/'fit_balance_v3/report.json'
    result['fitting']=json.loads(fitting.read_text()) if fitting_ready(dataset) else None
    if result['fitting']:
        for row in result['fitting']['models']:
            row['test_shift_warning']=test_shift_warning(row)
    upgraded=output/'feature_policy_v4/report.json'
    result['features']=json.loads(upgraded.read_text()) if features_ready(dataset) else None
    cluster_version='clustering_extended_v2' if (output/'clustering_extended_v2/verification.json').exists() else 'clustering_v1'
    clustering=output/cluster_version/'report.json'
    cluster_verification=output/cluster_version/'verification.json'
    result['clustering']=(json.loads(clustering.read_text())
                          if clustering.exists() and cluster_verification.exists()
                          and json.loads(cluster_verification.read_text()).get('status')=='passed' else None)
    if result['clustering']:
        result['clustering']['output_folder']=cluster_version
    if result['features']:
        for row in result['features']['models']:
            row['test_shift_warning']=test_shift_warning(row)
        champion=next(row for row in result['features']['models'] if row['model']==result['features']['winner'])
        result['features']['availability_warning']=(
            'Elliptic retained an offline v3 benchmark with supplied forward-neighbour aggregates. '
            'It is not a validated at-transaction-time model; incoming-only candidates are reported separately.'
            if dataset=='elliptic' and champion.get('reused_checkpoint') else '')
    fitting_status=ROOT/'evaluation/fit_balance_v3/status.json'
    result['fitting_status']=json.loads(fitting_status.read_text()) if fitting_status.exists() else None
    status=ROOT/'evaluation/priority_search_v2/status.json'
    result['training_status']=json.loads(status.read_text()) if status.exists() else None
    if result['training_status']:
        result['training_status']['completed']=[name for name in datasets()
            if (ROOT/'evaluation'/name/'priority_search_v2/report.json').exists()]
    return result

@app.get('/datasets/{dataset}/schema')
def schema(dataset: str,variant: Literal['selected','production','balanced','supervised','priority','fitting','features']='selected',
           fpr_budget: float=.05, threshold_policy: Literal['point','ucb95']='point', review_capacity: float | None=None):
    valid(dataset)
    selected_features=variant=='selected' and json.loads((ROOT/'configs/final_models.json').read_text())['datasets'][dataset]['experiment']=='feature_policy_v4'
    if fpr_budget not in ((.005,.01,.02,.03,.05) if variant == 'features' or selected_features else (.005,.01,.05)):
        raise HTTPException(422,'Unsupported FPR budget for this variant')
    if review_capacity not in (None,.005,.01,.02,.05):
        raise HTTPException(422,'Unsupported review-capacity scenario')
    if variant != 'features' and not selected_features and (threshold_policy != 'point' or review_capacity is not None):
        raise HTTPException(422,'Policy scenarios require the features variant')
    try:
        return json.loads(execute(dataset,[str(ROOT/'utils/worker.py'),dataset,variant,str(fpr_budget),
                         threshold_policy,str(review_capacity) if review_capacity is not None else 'none']))
    except RuntimeError as exc:
        raise HTTPException(503,str(exc)) from exc

@app.post('/predict/{dataset}')
def prediction(dataset: str,request: PredictionRequest):
    valid(dataset)
    try:
        return {'predictions':predict(dataset,request.records,request.variant,request.fpr_budget,request.threshold_policy,request.review_capacity),'notice':'Model scores are review signals; analyst review is required. Calibration FPR/confidence and review-capacity limits do not guarantee future-period compliance.'}
    except RuntimeError as exc:
        raise HTTPException(422,str(exc)) from exc

@app.get('/evaluation/{dataset}/{filename:path}')
def evaluation_file(dataset: str,filename: str):
    valid(dataset)
    base=(ROOT/'evaluation'/dataset).resolve()
    path=(base/filename).resolve()
    if not path.is_relative_to(base) or not path.is_file() or path.suffix not in ('.png','.csv','.json','.md'):
        raise HTTPException(404,'Evaluation file not found')
    return FileResponse(path)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8050)
