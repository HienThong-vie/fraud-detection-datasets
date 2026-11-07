import argparse
import json
from pathlib import Path
from utils.runtime import datasets, predict

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True,choices=list(datasets()))
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--variant',choices=['production','balanced','supervised','priority','fitting','features'],default='production')
    parser.add_argument('--fpr-budget',type=float,choices=[.005,.01,.02,.03,.05],default=.01)
    parser.add_argument('--threshold-policy',choices=['point','ucb95'],default='point')
    parser.add_argument('--review-capacity',type=float,choices=[.005,.01,.02,.05])
    args=parser.parse_args()
    records=json.loads(args.input.read_text())
    print(json.dumps(predict(args.dataset,records if isinstance(records,list) else [records],args.variant,args.fpr_budget,args.threshold_policy,args.review_capacity),indent=2))
