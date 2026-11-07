"""Reload and score every saved family model in an isolated dataset process."""
from pathlib import Path
import argparse
import json
import sys
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dataset');args=parser.parse_args()
    dataset=args.dataset
    pipeline=ROOT/'pipelines'/dataset
    sys.path.insert(0,str(pipeline/'src'))
    sys.path.insert(0,str(ROOT/'training_model'))
    out=ROOT/'evaluation'/dataset/'supervised_search_v1'
    report=json.loads((out/'report.json').read_text())
    prepared=next((pipeline/'preprocessed_df').glob('base_model_data.csv.gz')) if dataset=='baf' else next((pipeline/'preprocessed_df').glob('*.gz'))
    frame=pd.read_csv(prepared,nrows=3)
    for row in report['models']:
        bundle=joblib.load(out/(row['model']+'.joblib'))
        scores=bundle['model'].predict_proba(frame[bundle['feature_columns']])[:,1]
        assert np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
        assert bundle['threshold']==row['threshold']
    print(dataset+': four model checkpoints reload and predict successfully')
