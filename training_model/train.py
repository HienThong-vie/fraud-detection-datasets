from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "experiment_suite"
    subprocess.run([sys.executable, "training_model/priority_search.py", "--dataset", "handbook"], cwd=root, check=True)
