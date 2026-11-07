"""Portable checks for either the complete suite or a single-dataset branch."""
import ast
import json
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient
from app import app

ROOT = Path(__file__).resolve().parents[1]
SELECTED = json.loads((ROOT/'configs/final_models.json').read_text())['datasets']


@pytest.mark.parametrize('dataset', list(SELECTED))
def test_selected_default(dataset):
    selected = SELECTED[dataset]
    client = TestClient(app)
    schema = client.get('/datasets/'+dataset+'/schema')
    assert schema.status_code == 200, schema.text
    response = client.post('/predict/'+dataset, json={'records':schema.json()['example']})
    assert response.status_code == 200, response.text
    prediction = response.json()['predictions'][0]
    assert prediction['threshold'] == selected['operating_points']['0.05']['threshold']


@pytest.mark.parametrize('dataset', list(SELECTED))
def test_prepared_evidence(dataset):
    result = json.loads((ROOT/'evaluation'/dataset/'prepared_workflow/score.json').read_text())
    if dataset == 'baf':
        manifest = json.loads((ROOT/'final_model_data/Base_v4_final_model_data_manifest.json').read_text())
        assert manifest['rows'] == 650000
        check = json.loads((ROOT/'evaluation/baf/independent_check_v4_result.json').read_text())
        assert check['status'] == 'passed'
    else:
        manifest = json.loads((ROOT/'final_model_data'/dataset/'manifest.json').read_text())
        assert manifest['exact_round_trip_verified']
        assert result['matches_frozen_test_metrics']
        assert manifest['checkpoint_sha256'] == SELECTED[dataset]['checkpoint_sha256']


def test_feature_policy_functions_documented():
    tree = ast.parse((ROOT/'training_model/feature_policy_search.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert ast.get_docstring(node), node.name


def test_csv_tampering_rejected(tmp_path):
    sys.path.insert(0, str(ROOT/'training_model'))
    from prepared_workflow import load_prepared
    (tmp_path/'manifest.json').write_text(json.dumps({'dataset':'demo','csv_sha256':'incorrect'}))
    (tmp_path/'model_data.csv').write_text('x\n1\n')
    with pytest.raises(ValueError, match='checksum'):
        load_prepared(tmp_path, 'demo')


def test_refit_uses_train_only_and_is_reloadable(tmp_path):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    sys.path.insert(0, str(ROOT/'training_model'))
    from prepared_workflow import refit
    from priority_search import fpr_threshold, predict_scores
    frame = pd.DataFrame({'x':np.arange(80,dtype=float),'target':np.tile([0,1],40),
                          '_split':np.repeat(['train','calibration','selection','test'],20)})
    original = Pipeline([('scale',StandardScaler()),('classifier',LogisticRegression())])
    bundle = {'model':original,'feature_columns':['x']}
    result = refit(bundle,frame,'logistic_regression')
    assert result['model'].steps[0][1].mean_[0] == 9.5
    assert not hasattr(original.steps[0][1], 'mean_')
    calibration = frame[frame._split == 'calibration']
    scores = predict_scores(result['model'],calibration[['x']])
    assert result['thresholds']['0.05'] == fpr_threshold(calibration.target,scores,.05)
    path = tmp_path/'refit.joblib'
    joblib.dump(result,path)
    np.testing.assert_array_equal(predict_scores(joblib.load(path)['model'],frame[['x']]),
                                  predict_scores(result['model'],frame[['x']]))
