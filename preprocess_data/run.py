from __future__ import annotations

from handbook_fraud.config import PROJECT_ROOT, load_config
from handbook_fraud.data import load_raw, prepare_splits, save_processed
from handbook_fraud.profiling import main as profile_main


def main() -> None:
    config = load_config()
    frame = load_raw(config["dataset_directory"])
    splits = prepare_splits(frame, config)
    destination = PROJECT_ROOT / "preprocessed_df" / "handbook_model_data.csv.gz"
    save_processed(splits, destination)
    profile_main([])
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
