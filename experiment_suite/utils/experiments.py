"""Only independently verified fitting experiments are exposed for prediction."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def fitting_ready(dataset):
    folder = ROOT / 'evaluation' / dataset / 'fit_balance_v3'
    marker = folder / 'verification.json'
    return ((folder / 'report.json').is_file() and marker.is_file()
            and json.loads(marker.read_text()).get('status') == 'passed')


def test_shift_warning(row):
    if row['validation']['pr_auc'] - row['test']['pr_auc'] > .10:
        return ('Large validation-to-test AP drop: investigate temporal/prevalence changes. '
                'This is an exploratory diagnostic, not a model-selection rule.')
    return ''


def features_ready(dataset):
    folder = ROOT / 'evaluation' / dataset / 'feature_policy_v4'
    marker = folder / 'verification.json'
    return ((folder / 'report.json').is_file() and marker.is_file()
            and json.loads(marker.read_text()).get('status') == 'passed')
