from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import ARTIFACT_ROOT, METRICS_ROOT


class ModelBundle:
    """Load the versioned supervised/unsupervised artifacts used by the API."""

    def __init__(self, artifact_root: Path = ARTIFACT_ROOT):
        self.root = artifact_root
        self.classifier = joblib.load(self.root / "classifier.joblib")
        self.regressor = joblib.load(self.root / "regressor.joblib")
        self.kmeans_scaler = joblib.load(self.root / "kmeans_scaler.joblib")
        self.kmeans = joblib.load(self.root / "kmeans.joblib")
        self.isolation_scaler = joblib.load(self.root / "isolation_scaler.joblib")
        self.isolation_forest = joblib.load(self.root / "isolation_forest.joblib")
        registry_path = METRICS_ROOT / "model_registry.json"
        self.registry = (
            json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
        )

    @staticmethod
    def _transition_frame(
        disbursement: float, loan_count: float, avg_ticket: float, period_index: int
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "current_disbursement_crore": disbursement,
                    "current_loan_count": loan_count,
                    "current_average_ticket_rupees": avg_ticket,
                    "period_index": period_index,
                }
            ]
        )

    def predict(
        self, disbursement: float, loan_count: float, avg_ticket: float, period_index: int
    ) -> dict:
        x = self._transition_frame(disbursement, loan_count, avg_ticket, period_index)
        cls = int(self.classifier.predict(x)[0])
        cls_prob = None
        if hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(x)[0]
            classes = list(self.classifier.classes_)
            if 1 in classes:
                cls_prob = float(proba[classes.index(1)])

        kx = pd.DataFrame(
            [
                {
                    "log_disbursement": np.log1p(disbursement),
                    "log_loan_count": np.log1p(loan_count),
                    "log_average_ticket": np.log1p(avg_ticket),
                }
            ]
        )
        cluster = int(self.kmeans.predict(self.kmeans_scaler.transform(kx))[0]) + 1

        ax = pd.DataFrame(
            [
                {
                    "digital_disbursement_crore": disbursement,
                    "digital_loan_count": loan_count,
                    "average_ticket_rupees": avg_ticket,
                }
            ]
        )
        ax_scaled = self.isolation_scaler.transform(ax)
        anomaly_flag = int(self.isolation_forest.predict(ax_scaled)[0])
        anomaly_score = float(self.isolation_forest.decision_function(ax_scaled)[0])

        return {
            "next_disbursement_increased": bool(cls),
            "increase_probability": cls_prob,
            "predicted_next_disbursement_crore": max(0.0, float(self.regressor.predict(x)[0])),
            "cluster": cluster,
            "anomaly_flag": (
                "potentially_unusual" if anomaly_flag == -1 else "within_observed_pattern"
            ),
            "anomaly_score": anomaly_score,
            "classifier": self.registry.get("best_classifier"),
            "regressor": self.registry.get("best_regressor"),
        }
