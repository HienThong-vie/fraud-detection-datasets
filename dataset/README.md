# Dataset

The six `.csv` links expose BAF Base and Variants I-V directly in this folder. They resolve to the local
files under `data/raw` without creating a second 1.3 GB copy. Raw data is excluded from Git. Download or
verify it with `python scripts/download_baf.py`.

The Assignment 2 ZIP uses an additional Base-only export:

- `Base_final_model_data.csv` is generated locally and ignored by Git because of its size and licence.
- `Base_final_model_data_dictionary.csv` documents column type, meaning, and model-input status.
- `Base_final_model_data_manifest.json` records row/column counts, file size, SHA-256, and licence.

After real training, regenerate it with `python scripts/export_base_for_zip.py`. Include those three
files in the private coursework ZIP; do not publish the large CSV separately.

