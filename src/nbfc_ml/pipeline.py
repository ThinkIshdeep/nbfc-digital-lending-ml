from __future__ import annotations

import hashlib
from pathlib import Path

from .config import CORE_DATASET, FIGURES_ROOT
from .data import build_transition_dataset, load_core_data, validate_core_data
from .models import train_all
from .plots import generate_plots


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_pipeline() -> dict:
    df = load_core_data()
    validate_core_data(df)
    trans = build_transition_dataset(df)
    latest = df[df["period"] == "As on Dec 31 2020"].copy()
    outputs = train_all(trans, latest)
    generate_plots(df, outputs.classification_metrics, outputs.regression_metrics, FIGURES_ROOT)

    result = {
        "core_rows": int(len(df)),
        "nbfc_ids": int(df["nbfc_respondent_id"].nunique()),
        "transition_rows": int(len(trans)),
        "transition_nbfc_groups": int(trans["nbfc_respondent_id"].nunique()),
        "best_classifier": outputs.metadata["best_classifier"],
        "best_regressor": outputs.metadata["best_regressor"],
        "data_sha256": sha256(CORE_DATASET),
    }
    return result
