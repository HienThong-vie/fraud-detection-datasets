# Conda workflow

Run from the suite directory. Python and the model-library versions are pinned to the evaluated environment.

```bash
conda env create -f environment.yml
conda activate fraudsuite
python -m pytest tests -q
python app.py
```

Prepare data, train and predict in a second terminal with the same environment:

```bash
conda activate fraudsuite
python -c "import sys; sys.path.insert(0,'training_model'); from priority_data import prepare; prepare('baf')"
python training_model/feature_policy_search.py --dataset baf
python training_model/extended_clustering.py --dataset baf
python predict.py --dataset baf --variant selected --fpr-budget 0.05 --input evaluation/baf/feature_policy_v4/example.json
```

Raw files belong under `dataset/<id>/`. Training commands preserve completed experiments. To perform a fresh search, use a new versioned output directory in a separate working copy rather than overwriting evidence.

The local verification environment uses `conda env create --prefix .conda-env -f environment.yml` to avoid interfering with any existing named environment. Activate that copy with `conda activate /absolute/path/to/.conda-env`.

Conda manages Python and pip; pip installs the pinned model libraries. This is intentionally a conda environment with pip dependencies. Mac/Linux training uses `fcntl` locks; inference can run on Windows.
