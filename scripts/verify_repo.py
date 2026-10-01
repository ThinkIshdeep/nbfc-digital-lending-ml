from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbfc_ml.data import build_transition_dataset, load_core_data, validate_core_data  # noqa: E402


def main() -> int:
    df = load_core_data()
    validate_core_data(df)
    trans = build_transition_dataset(df)

    expected = {
        "core_rows": 305,
        "nbfc_ids": 61,
        "transition_rows": 135,
        "transition_nbfc_groups": 51,
    }
    actual = {
        "core_rows": len(df),
        "nbfc_ids": df["nbfc_respondent_id"].nunique(),
        "transition_rows": len(trans),
        "transition_nbfc_groups": trans["nbfc_respondent_id"].nunique(),
    }
    for key, value in expected.items():
        if actual[key] != value:
            raise SystemExit(f"{key}: expected {value}, got {actual[key]}")

    required = [
        ROOT / "artifacts/models/classifier.joblib",
        ROOT / "artifacts/models/regressor.joblib",
        ROOT / "artifacts/models/kmeans.joblib",
        ROOT / "artifacts/models/kmeans_scaler.joblib",
        ROOT / "artifacts/models/isolation_forest.joblib",
        ROOT / "artifacts/models/isolation_scaler.joblib",
        ROOT / "artifacts/metrics/model_registry.json",
        ROOT / "artifacts/metrics/temporal_holdout_metrics.csv",
        ROOT / "artifacts/metrics/temporal_holdout_predictions.csv",
        ROOT / "dashboard/NBFC_Interactive_Dashboard.html",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing required artifacts: " + ", ".join(missing))

    registry = json.loads((ROOT / "artifacts/metrics/model_registry.json").read_text())
    if registry["best_classifier"] != "Random Forest Classifier":
        raise SystemExit("Unexpected classifier selection")
    if registry["best_regressor"] != "Gradient Boosting Regressor":
        raise SystemExit("Unexpected regressor selection")

    manifest = json.loads((ROOT / "docs/DATASET_MANIFEST.json").read_text())
    if not manifest.get("files"):
        raise SystemExit("Dataset manifest is empty")
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.exists():
            raise SystemExit(f"Manifest file missing from repository: {item["path"]}")

    temporal = pd.read_csv(ROOT / "artifacts/metrics/temporal_holdout_metrics.csv")
    if set(temporal["task"]) != {"classification", "regression"}:
        raise SystemExit("Temporal evaluation is missing a supervised task")
    if len(pd.read_csv(ROOT / "artifacts/metrics/temporal_holdout_predictions.csv")) != 51:
        raise SystemExit("Unexpected temporal holdout prediction count")

    print(json.dumps({"status": "ok", **actual, "temporal_holdout_rows": 51}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
