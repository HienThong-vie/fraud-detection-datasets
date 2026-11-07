"""Promote the frozen BAF champion after matching production and experiment inference."""
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training_model'))
sys.path.insert(0, str(ROOT / 'pipelines/baf/src'))
from finalize_evidence import sha
from ai4cyber.inference import ModelService


def main():
    registry = json.loads((ROOT / 'configs/final_models.json').read_text())['datasets']['baf']
    source = ROOT / registry['checkpoint']
    assert sha(source) == registry['checkpoint_sha256']
    bundle = joblib.load(source)
    destination = ROOT / 'checkpoint/baf'
    backup = destination / 'legacy_7919ca0'
    backup.mkdir(exist_ok=True)
    for name in ('production_model.joblib', 'model_card.json'):
        if (destination / name).exists() and not (backup / name).exists():
            shutil.copy2(destination / name, backup / name)
    raw_columns = json.loads((ROOT / 'preprocessed_df/baf/priority_v2/metadata.json').read_text())['feature_columns']
    classifier = bundle['model'].steps[-1][1]
    retained = bundle['model'].steps[0][1].named_steps['screen'].columns_
    bundle.update(pipeline=bundle['model'], experiment='feature_policy_v4', model_name='catboost',
        threshold=registry['operating_points']['0.05']['threshold'], raw_feature_columns=raw_columns,
        headline_fpr_budget=.05, secondary_fpr_budget=.01,
        protected_features_excluded=['customer_age', 'income', 'employment_status'],
        global_feature_importance=dict(zip(retained, map(float, classifier.feature_importances_))),
        notice='Experimental review score, not proof of fraud. Threshold calibrated to 5% FPR; actual future FPR may differ.',
        source_checkpoint_sha256=registry['checkpoint_sha256'])
    staged = destination / 'production_model.candidate.joblib'
    joblib.dump(bundle, staged, compress=3)
    frame = pd.read_pickle(ROOT / 'preprocessed_df/baf/feature_policy_v4/data.pkl')
    test = frame[frame._split == 'test']
    service = ModelService(staged)
    example = test[bundle['feature_columns']].head(1).replace({np.nan: None}).to_dict('records')[0]
    scored = service.predict(example)
    expected = round(float(bundle['model'].predict_proba(pd.DataFrame([example]))[0, 1]), 12)
    assert np.isclose(scored['score'], expected, atol=1e-12)
    assert np.isclose(scored['threshold'], bundle['threshold'])
    old_frame = pd.read_pickle(ROOT / 'preprocessed_df/baf/priority_v2/data.pkl')
    raw_example = old_frame[old_frame._split == 'test'][raw_columns].head(1).replace({np.nan: None}).to_dict('records')[0]
    assert np.isclose(service.predict(raw_example)['score'], expected, atol=1e-12)
    staged.replace(destination / 'production_model.joblib')
    card = dict(dataset='baf', model='catboost', experiment='feature_policy_v4',
        source_checkpoint=registry['checkpoint'], source_sha256=registry['checkpoint_sha256'],
        production_sha256=sha(destination / 'production_model.joblib'),
        parameters={k: classifier.get_params()[k] for k in ('depth','iterations','l2_leaf_reg','learning_rate','auto_class_weights')},
        pipeline_inputs=len(bundle['feature_columns']), retained_inputs=len(retained),
        headline_fpr_budget=.05, secondary_fpr_budget=.01, operating_points=registry['operating_points'],
        raw_and_engineered_input_equivalence_verified=True,
        limitations='Previously inspected test periods; exploratory comparison. Age-group FPR disparity remains. No causal feature attribution supplied.',
        previous_checkpoint='legacy_7919ca0/production_model.joblib')
    (destination / 'model_card.json').write_text(json.dumps(card, indent=2))
    print('PROMOTED BAF', card['production_sha256'], flush=True)


if __name__ == '__main__':
    main()
