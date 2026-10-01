from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nbfc_ml.predictor import ModelBundle

parser = argparse.ArgumentParser(description="Score one aggregate NBFC observation")
parser.add_argument("--disbursement", type=float, required=True)
parser.add_argument("--loan-count", type=float, required=True)
parser.add_argument("--avg-ticket", type=float, required=True)
parser.add_argument("--period-index", type=int, required=True)
args = parser.parse_args()

bundle = ModelBundle()
print(json.dumps(bundle.predict(args.disbursement, args.loan_count, args.avg_ticket, args.period_index), indent=2))
