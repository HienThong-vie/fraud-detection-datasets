from __future__ import annotations

import json

from handbook_fraud.config import PROJECT_ROOT


def main() -> None:
    metrics_path = PROJECT_ROOT / "outputs" / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit("No evaluation output found. Run python training_model/train.py first.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print(json.dumps(metrics, indent=2))
    print(f"Charts: {PROJECT_ROOT / 'outputs'}")


if __name__ == "__main__":
    main()
