from pathlib import Path
import sys
from supervised_search import run

if __name__ == '__main__':
    run('sparkov', project=Path(__file__).resolve().parents[1])
    run('sparkov', project=Path(__file__).resolve().parents[1], include_prior=True)
