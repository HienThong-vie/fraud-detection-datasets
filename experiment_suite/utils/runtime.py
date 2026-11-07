"""Common paths and isolated execution for dataset-specific Python packages."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def datasets():
    """Dataset registry read from configs/datasets.json."""
    return json.loads((ROOT/'configs/datasets.json').read_text())

def pipeline(dataset):
    """Folder of a dataset's pipeline; unknown names are rejected."""
    if dataset not in datasets():
        raise ValueError('Unknown dataset: '+dataset)
    return ROOT/'pipelines'/dataset

def execute(dataset, arguments, payload=None):
    """Run a script in a separate Python process with the dataset's own package on the path; return its
    output or raise with the error."""
    directory = pipeline(dataset)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(directory/'src'),str(directory)]), MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'))
    result = subprocess.run([sys.executable,*arguments],cwd=directory,env=env,input=json.dumps(payload),text=True,capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:] or result.stdout[-4000:])
    return result.stdout

def predict(dataset, records, variant='selected', fpr_budget=.05, threshold_policy='point', review_capacity=None):
    """Score records with the chosen model version, FPR budget and policy through worker.py."""
    if variant not in ('selected','production','balanced','supervised','priority','fitting','features'):
        raise ValueError('Unknown model variant')
    output = execute(dataset,[str(ROOT/'utils/worker.py'),dataset,variant,str(fpr_budget),
                              threshold_policy,str(review_capacity) if review_capacity is not None else 'none'],records)
    return json.loads(output)
