# Dataset

- [Official daily simulated data](https://github.com/Fraud-Detection-Handbook/simulated-data-raw)
- [Fraud Detection Handbook](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook)
- [Dataset generation notebook](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook/blob/main/Chapter_3_GettingStarted/SimulatedDataset.ipynb)

Run `python scripts/download_handbook.py` to download the pinned official daily `.pkl` files into
`data/raw/`. Only load pickle files obtained from this trusted source. Pickle files can execute code when
opened, and the raw-data repository does not display a separate explicit data licence file.
