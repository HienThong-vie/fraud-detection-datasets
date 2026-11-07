from __future__ import annotations

import json

from fraudbench.config import PROJECT_ROOT, load_config
from fraudbench.data import load_splits

if __name__ == "__main__":
    config = load_config(PROJECT_ROOT / "configs" / "training.json")
    splits, _ = load_splits(config)
    example = splits["test"][0].iloc[0].to_dict()
    destination = PROJECT_ROOT / "examples" / "sample_transaction.json"
    destination.write_text(json.dumps(example, indent=2), encoding="utf-8")
    print(f"Saved {destination}")
