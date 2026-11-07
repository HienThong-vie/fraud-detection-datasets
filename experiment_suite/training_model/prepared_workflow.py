"""Export, score, refit and predict with a dataset's prepared features and frozen selected model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'training_model'))
from priority_search import predict_scores, fpr_threshold
from supervised_search import measure


def digest(path):
    """Stream a file checksum without holding its bytes in memory."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def selected_bundle(dataset, model_path=None):
    """Load a trusted local bundle; the default must match the frozen registry checksum."""
    selected = json.loads((ROOT/'configs/final_models.json').read_text())['datasets'][dataset]
    sys.path.insert(0, str(ROOT/'pipelines'/dataset/'src'))
    path = Path(model_path) if model_path else ROOT/selected['checkpoint']
    if not model_path and digest(path) != selected['checkpoint_sha256']:
        raise ValueError('Selected checkpoint checksum differs from registry')
    return selected, joblib.load(path)


def export(dataset, out):
    """Write full-precision prepared rows and prove an exact dtype/value round trip."""
    selected, bundle = selected_bundle(dataset)
    out = Path(out)
    if (out/'model_data.csv').exists():
        raise FileExistsError('Preserving existing CSV; choose a new --data-dir')
    if selected['experiment'] == 'feature_policy_v4':
        from upgrade_data import prepare
        frame, _, _ = prepare(dataset)
    else:
        from priority_data import prepare
        frame, _ = prepare(dataset)
    frame = frame.reset_index(drop=True)
    dtypes = {column:str(dtype) for column,dtype in frame.dtypes.items()}
    out.mkdir(parents=True, exist_ok=True)
    path = out/'model_data.csv'
    frame.to_csv(path, index=False, float_format='%.17g')
    restored = pd.read_csv(path, dtype=dtypes, float_precision='round_trip', keep_default_na=False, na_values=[''])
    pd.testing.assert_frame_equal(frame, restored, check_exact=True)
    dictionary = []
    for column in frame:
        meaning = {'target':'Binary target label, never a model input.',
                   '_event_time':'Original event-time ordering, excluded from model inputs.',
                   '_split':'Frozen train/calibration/selection/test partition.',
                   '_sample_weight':'Training-only sampling weight; not an input feature.'}.get(
                       column, 'Prepared feature: '+column+'. See pipeline documentation for derivation.')
        dictionary.append(dict(column=column,type=dtypes[column],meaning=meaning,
                               model_input='yes' if column in bundle['feature_columns'] else 'no'))
    pd.DataFrame(dictionary).to_csv(out/'dictionary.csv', index=False)
    manifest = dict(dataset=dataset, experiment=selected['experiment'],rows=len(frame),columns=len(frame.columns),
        csv_sha256=digest(path),csv_bytes=path.stat().st_size,dtypes=dtypes,
        frame_sha256=hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest(),
        feature_columns=bundle['feature_columns'],checkpoint_sha256=selected['checkpoint_sha256'],
        periods={name:dict(rows=len(part),positives=int(part.target.sum())) for name,part in frame.groupby('_split')},
        exact_round_trip_verified=True,float_format='%.17g')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def load_prepared(directory, dataset):
    """Accept only the declared dataset, checksum, dtypes and frozen frame fingerprint."""
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    path = directory/'model_data.csv'
    if manifest['dataset'] != dataset or digest(path) != manifest['csv_sha256']:
        raise ValueError('Prepared dataset identity/checksum differs')
    frame = pd.read_csv(path,dtype=manifest['dtypes'],float_precision='round_trip',keep_default_na=False,na_values=[''])
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()
    if fingerprint != manifest['frame_sha256']:
        raise ValueError('Prepared frame values, order or dtypes differ')
    return frame, manifest


def score(bundle, frame, thresholds):
    """Measure selection/test only after thresholds have been fixed elsewhere."""
    result = {}
    for name in ('selection','test'):
        part = frame[frame._split==name]
        values = predict_scores(bundle['model'],part[bundle['feature_columns']])
        result[name] = {budget:dict(threshold=float(threshold),alert_rate=float(np.mean(values>=threshold)),**measure(part.target,values,threshold))
                        for budget,threshold in thresholds.items()}
    return result


def refit(bundle, frame, family):
    """Fit frozen hyperparameters on training rows only; do not reselect or overwrite a champion."""
    model = clone(bundle['model'])
    part = frame[frame._split=='train']
    transform, classifier = model.steps[0][1], model.steps[-1][1]
    values = transform.fit_transform(part[bundle['feature_columns']],part.target)
    weights = part.get('_sample_weight',pd.Series(1.,index=part.index)).to_numpy(dtype=float)
    weights = weights/weights.mean()
    kwargs = dict(sample_weight=weights)
    if family=='catboost':
        kwargs['cat_features'] = [column for column in values if not pd.api.types.is_numeric_dtype(values[column])]
    if family=='xgboost':
        classifier.set_params(early_stopping_rounds=None)
    classifier.fit(values,part.target,**kwargs)
    calibration = frame[frame._split=='calibration']
    values = predict_scores(model,calibration[bundle['feature_columns']])
    thresholds = {budget:fpr_threshold(calibration.target,values,float(budget)) for budget in ('0.05','0.01')}
    return dict(bundle,model=model,threshold=thresholds['0.05'],thresholds=thresholds,refitted_from_prepared=True)


def predict(bundle, record, manifest, budget):
    """Predict prepared feature records; no raw-record feature inference is invented."""
    records = json.loads(Path(record).read_text())
    frame = pd.DataFrame(records if isinstance(records,list) else [records])
    columns = bundle['feature_columns']
    missing = set(columns)-set(frame)
    if missing:
        raise ValueError('Prepared record is missing features: '+', '.join(sorted(missing)))
    for column in columns:
        if manifest['dtypes'][column] not in ('object','str','string'):
            frame[column] = pd.to_numeric(frame[column],errors='coerce')
    values = predict_scores(bundle['model'],frame[columns])
    threshold = bundle['thresholds'][budget]
    return [dict(score=float(value),threshold=float(threshold),flagged=bool(value>=threshold)) for value in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['export','score','refit','predict'])
    parser.add_argument('dataset')
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--record',type=Path)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--output',type=Path,help='New directory for refit outputs')
    parser.add_argument('--result',type=Path,help='New JSON file for aggregate results')
    parser.add_argument('--budget',choices=['0.05','0.01'],default='0.05')
    args = parser.parse_args()
    if args.result and args.result.exists():
        raise FileExistsError(args.result)
    directory = args.data_dir or ROOT/'final_model_data'/args.dataset
    if args.mode=='export':
        result = export(args.dataset,directory)
    else:
        selected,bundle = selected_bundle(args.dataset,args.model)
        if args.mode=='predict':
            if args.record is None:
                parser.error('predict requires --record with prepared features')
            manifest = json.loads((directory/'manifest.json').read_text())
            if manifest['dataset'] != args.dataset or manifest['checkpoint_sha256'] != selected['checkpoint_sha256']:
                raise ValueError('Prepared manifest belongs to another dataset or checkpoint')
            result = predict(bundle,args.record,manifest,args.budget)
        else:
            frame,manifest = load_prepared(directory,args.dataset)
            if manifest['checkpoint_sha256'] != selected['checkpoint_sha256']:
                raise ValueError('Prepared data came from another selected checkpoint')
            if args.mode=='refit':
                output = args.output or ROOT/'checkpoint'/args.dataset/'prepared_refit'
                if output.exists():
                    raise FileExistsError('Preserving existing refit; choose another --output')
                bundle = refit(bundle,frame,selected['model'])
                output.mkdir(parents=True,exist_ok=False)
                joblib.dump(bundle,output/'refit.joblib')
            result = dict(dataset=args.dataset,mode=args.mode,experiment=selected['experiment'],
                          metrics=score(bundle,frame,{b:bundle['thresholds'][b] for b in ('0.05','0.01')}))
            if args.mode=='score' and args.model is None:
                for budget in ('0.05','0.01'):
                    for metric,value in selected['operating_points'][budget]['test_metrics'].items():
                        assert abs(result['metrics']['test'][budget][metric]-value)<1e-12,(budget,metric)
                result['matches_frozen_test_metrics'] = True
            if args.mode=='refit':
                (output/'metrics.json').write_text(json.dumps(result,indent=2))
    if args.result:
        args.result.parent.mkdir(parents=True,exist_ok=True)
        args.result.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
