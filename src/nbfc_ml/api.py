from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .config import DATA_ROOT, METRICS_ROOT, PROJECT_ROOT
from .predictor import ModelBundle

app = FastAPI(
    title="NBFC Digital Lending Intelligence API",
    version=__version__,
    description=(
        "Analytical service for aggregate NBFC digital-lending patterns. "
        "Not a credit-approval, pricing or borrower-risk decision service."
    ),
)

FRONTEND_ROOT = PROJECT_ROOT / "frontend" / "static"
if FRONTEND_ROOT.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_ROOT), name="static")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_disbursement_crore: float = Field(ge=0)
    current_loan_count: float = Field(ge=0)
    current_average_ticket_rupees: float = Field(gt=0)
    period_index: int = Field(ge=0, le=3)


@lru_cache(maxsize=1)
def get_bundle() -> ModelBundle:
    return ModelBundle()


def _load_csv(name: str) -> pd.DataFrame:
    path = DATA_ROOT / name
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Data artifact not found: {name}")
    return pd.read_csv(path)


def _registry() -> dict:
    path: Path = METRICS_ROOT / "model_registry.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _clean_records(frame: pd.DataFrame) -> list[dict]:
    # JSON has no NaN value. Convert pandas missing values to None.
    return json.loads(frame.where(pd.notna(frame), None).to_json(orient="records"))


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    index = FRONTEND_ROOT / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not installed")
    return FileResponse(index)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "nbfc-digital-lending-intelligence", "version": __version__}


@app.get("/ready")
def ready() -> dict:
    try:
        get_bundle()
        return {"status": "ready", "version": __version__}
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "version": __version__, "detail": str(exc)},
        ) from exc


@app.get("/v1/model-info")
def model_info() -> dict:
    registry = _registry()
    if not registry:
        return {"status": "unavailable"}
    return registry


@app.post("/v1/predict")
def predict(request: PredictionRequest) -> dict:
    result = get_bundle().predict(
        request.current_disbursement_crore,
        request.current_loan_count,
        request.current_average_ticket_rupees,
        request.period_index,
    )
    result["model_version"] = __version__
    return result


@app.get("/v1/dashboard/summary")
def dashboard_summary() -> dict:
    summary = _load_csv("digital_lending_period_summary.csv")
    latest = summary.iloc[-1]
    registry = _registry()
    return {
        "core_cells": int(summary["row_cells"].iloc[0]),
        "transition_rows": int(registry.get("transition_rows", 0)),
        "latest_period": str(latest["period"]),
        "latest_disbursement_crore": float(latest["amount_total_crore"]),
        "latest_loan_count": float(latest["loan_count_total"]),
        "latest_weighted_average_ticket_rupees": float(latest["weighted_average_ticket_rupees"]),
        "nbfc_groups": int(registry.get("nbfc_groups", 0)),
        "latest_complete_observations": int(registry.get("latest_complete_observations", 0)),
        "anomaly_flags": int(registry.get("anomaly_flags", 0)),
    }


@app.get("/v1/dashboard/trends")
def dashboard_trends() -> dict:
    frame = _load_csv("digital_lending_period_summary.csv")
    cols = [
        "period",
        "amount_total_crore",
        "loan_count_total",
        "weighted_average_ticket_rupees",
        "amount_observations",
        "loan_count_observations",
    ]
    return {"periods": _clean_records(frame[cols])}


@app.get("/v1/dashboard/product-mix")
def dashboard_product_mix() -> list[dict]:
    frame = _load_csv("digital_lending_product_mix.csv")
    frame = frame.sort_values("nbfc_amount_crore", ascending=False)
    return _clean_records(
        frame[
            [
                "loan_purpose",
                "nbfc_amount_crore",
                "nbfc_loan_count",
                "nbfc_avg_ticket_rupees",
            ]
        ]
    )


@app.get("/v1/dashboard/model-lab")
def dashboard_model_lab() -> dict:
    registry = _registry()
    cls = _clean_records(pd.read_csv(METRICS_ROOT / "classification_metrics.csv"))
    reg = _clean_records(pd.read_csv(METRICS_ROOT / "regression_metrics.csv"))
    fi = _clean_records(pd.read_csv(METRICS_ROOT / "classification_feature_importance.csv"))
    temporal = pd.read_csv(METRICS_ROOT / "temporal_holdout_metrics.csv")

    def row(task: str, model: str) -> dict:
        selected = temporal[(temporal["task"] == task) & (temporal["model"] == model)]
        if selected.empty:
            return {}
        return selected.iloc[0].to_dict()

    return {
        "best_classifier": registry.get("best_classifier"),
        "best_regressor": registry.get("best_regressor"),
        "classification": cls,
        "regression": reg,
        "feature_importance": fi,
        "temporal": {
            "classifier_baseline": row("classification", "Majority Class Baseline"),
            "classifier": row("classification", registry.get("best_classifier", "")),
            "regressor_baseline": row("regression", "Persistence Baseline"),
            "regressor": row("regression", registry.get("best_regressor", "")),
        },
    }


@app.get("/v1/dashboard/segments")
def dashboard_segments() -> list[dict]:
    frame = pd.read_csv(METRICS_ROOT / "kmeans_profiles.csv")
    labels = {
        1: ("Low-volume / high-ticket", "Smaller observed disbursement and loan counts, with the largest mean ticket among the three groups."),
        2: ("High-volume lending", "The largest observed loan count and disbursement profile in the latest complete period."),
        3: ("Scaled small-ticket", "Lower mean disbursement with materially higher loan volume and a smaller mean ticket."),
    }
    records = []
    for item in _clean_records(frame):
        cluster = int(item["cluster"])
        label, description = labels.get(cluster, (f"Cluster {cluster}", "Observed lending behaviour segment."))
        item["label"] = label
        item["description"] = description
        records.append(item)
    return records


@app.get("/v1/dashboard/anomalies")
def dashboard_anomalies() -> list[dict]:
    frame = pd.read_csv(METRICS_ROOT / "latest_clusters_anomalies.csv")
    flagged = frame[frame["anomaly_flag"] == -1].copy().sort_values("anomaly_score")
    return _clean_records(
        flagged[
            [
                "nbfc_respondent_id",
                "period",
                "digital_disbursement_crore",
                "digital_loan_count",
                "average_ticket_rupees",
                "anomaly_flag",
                "anomaly_score",
            ]
        ]
    )
