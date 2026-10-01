from __future__ import annotations

import argparse
import json


def train_command() -> None:
    from .pipeline import run_pipeline
    print(json.dumps(run_pipeline(), indent=2))


def predict_command() -> None:
    parser = argparse.ArgumentParser(description="Score one aggregate NBFC observation")
    parser.add_argument("--disbursement", type=float, required=True)
    parser.add_argument("--loan-count", type=float, required=True)
    parser.add_argument("--avg-ticket", type=float, required=True)
    parser.add_argument("--period-index", type=int, required=True)
    args = parser.parse_args()
    from .predictor import ModelBundle
    bundle = ModelBundle()
    print(json.dumps(bundle.predict(args.disbursement, args.loan_count, args.avg_ticket, args.period_index), indent=2))
