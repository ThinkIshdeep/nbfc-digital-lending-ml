from __future__ import annotations

from dataclasses import dataclass
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ARTIFACT_ROOT, FIGURES_ROOT, METRICS_ROOT

CLASS_FEATURES = [
    "current_disbursement_crore",
    "current_loan_count",
    "current_average_ticket_rupees",
    "period_index",
]
CLUSTER_FEATURES = [
    "digital_disbursement_crore",
    "digital_loan_count",
    "average_ticket_rupees",
]


@dataclass
class TrainingOutputs:
    classification_metrics: pd.DataFrame
    regression_metrics: pd.DataFrame
    confusion: pd.DataFrame
    feature_importance: pd.DataFrame
    clusters: pd.DataFrame
    anomalies: pd.DataFrame
    temporal_metrics: pd.DataFrame
    metadata: dict


def build_models(random_state: int = 42) -> tuple[dict, dict]:
    classifiers = {
        "Logistic Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=random_state
                    ),
                ),
            ]
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            min_samples_leaf=3,
            max_features=0.8,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }
    regressors = {
        "Dummy Median": DummyRegressor(strategy="median"),
        "Ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=500,
            random_state=random_state,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            random_state=random_state,
            n_estimators=200,
            max_depth=2,
            learning_rate=0.03,
            loss="huber",
        ),
    }
    return classifiers, regressors


def _classification_row(name: str, y_true: pd.Series, pred: np.ndarray) -> list:
    return [
        name,
        accuracy_score(y_true, pred),
        precision_score(y_true, pred, zero_division=0),
        recall_score(y_true, pred, zero_division=0),
        f1_score(y_true, pred, zero_division=0),
    ]


def _regression_row(name: str, y_true: pd.Series, pred: np.ndarray) -> list:
    return [
        name,
        mean_absolute_error(y_true, pred),
        mean_squared_error(y_true, pred) ** 0.5,
        r2_score(y_true, pred),
    ]


def _temporal_holdout(
    trans: pd.DataFrame,
    best_clf_name: str,
    best_reg_name: str,
    random_state: int,
) -> pd.DataFrame:
    """Evaluate the selected supervised models on the final chronological transition."""
    holdout_label = trans["to_period"].iloc[-1]
    train = trans[trans["to_period"] != holdout_label].copy()
    test = trans[trans["to_period"] == holdout_label].copy()
    if train.empty or test.empty:
        raise ValueError("Temporal holdout requires both training and final-period test rows")

    classifiers, regressors = build_models(random_state)
    rows: list[dict] = []

    # Majority-class baseline.
    majority = int(train["next_disbursement_increased"].mode().iloc[0])
    cls_pred = np.full(len(test), majority, dtype=int)
    rows.append(
        {
            "task": "classification",
            "model": "Majority Class Baseline",
            "accuracy": accuracy_score(test["next_disbursement_increased"], cls_pred),
            "precision": precision_score(test["next_disbursement_increased"], cls_pred, zero_division=0),
            "recall": recall_score(test["next_disbursement_increased"], cls_pred, zero_division=0),
            "f1": f1_score(test["next_disbursement_increased"], cls_pred, zero_division=0),
            "mae_crore": np.nan,
            "rmse_crore": np.nan,
            "r2": np.nan,
            "train_rows": len(train),
            "test_rows": len(test),
            "test_period": holdout_label,
        }
    )

    clf = classifiers[best_clf_name].fit(train[CLASS_FEATURES], train["next_disbursement_increased"])
    cls_pred = clf.predict(test[CLASS_FEATURES])
    rows.append(
        {
            "task": "classification",
            "model": best_clf_name,
            "accuracy": accuracy_score(test["next_disbursement_increased"], cls_pred),
            "precision": precision_score(test["next_disbursement_increased"], cls_pred, zero_division=0),
            "recall": recall_score(test["next_disbursement_increased"], cls_pred, zero_division=0),
            "f1": f1_score(test["next_disbursement_increased"], cls_pred, zero_division=0),
            "mae_crore": np.nan,
            "rmse_crore": np.nan,
            "r2": np.nan,
            "train_rows": len(train),
            "test_rows": len(test),
            "test_period": holdout_label,
        }
    )

    # Persistence baseline: next period equals current period.
    naive_pred = test["current_disbursement_crore"].to_numpy(dtype=float)
    rows.append(
        {
            "task": "regression",
            "model": "Persistence Baseline",
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "mae_crore": mean_absolute_error(test["next_disbursement_crore"], naive_pred),
            "rmse_crore": mean_squared_error(test["next_disbursement_crore"], naive_pred) ** 0.5,
            "r2": r2_score(test["next_disbursement_crore"], naive_pred),
            "train_rows": len(train),
            "test_rows": len(test),
            "test_period": holdout_label,
        }
    )

    reg = regressors[best_reg_name].fit(train[CLASS_FEATURES], train["next_disbursement_crore"])
    reg_pred = reg.predict(test[CLASS_FEATURES])
    rows.append(
        {
            "task": "regression",
            "model": best_reg_name,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "mae_crore": mean_absolute_error(test["next_disbursement_crore"], reg_pred),
            "rmse_crore": mean_squared_error(test["next_disbursement_crore"], reg_pred) ** 0.5,
            "r2": r2_score(test["next_disbursement_crore"], reg_pred),
            "train_rows": len(train),
            "test_rows": len(test),
            "test_period": holdout_label,
        }
    )

    # Persist predictions so a reviewer can inspect every chronological holdout row.
    pred_frame = test[["nbfc_respondent_id", "from_period", "to_period"]].copy()
    pred_frame["actual_increase"] = test["next_disbursement_increased"].to_numpy()
    pred_frame["predicted_increase"] = cls_pred
    pred_frame["actual_disbursement_crore"] = test["next_disbursement_crore"].to_numpy()
    pred_frame["predicted_disbursement_crore"] = reg_pred
    pred_frame.to_csv(METRICS_ROOT / "temporal_holdout_predictions.csv", index=False)

    return pd.DataFrame(rows)


def train_all(trans: pd.DataFrame, latest: pd.DataFrame, random_state: int = 42) -> TrainingOutputs:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    X = trans[CLASS_FEATURES].astype(float)
    groups = trans["nbfc_respondent_id"].to_numpy()
    y_cls = trans["next_disbursement_increased"].astype(int).to_numpy()
    y_reg = trans["next_disbursement_crore"].astype(float).to_numpy()
    cv = GroupKFold(n_splits=5)
    classifiers, regressors = build_models(random_state)

    clf_rows = []
    clf_preds = {}
    for name, model in classifiers.items():
        pred = cross_val_predict(model, X, y_cls, cv=cv, groups=groups)
        clf_preds[name] = pred
        clf_rows.append(_classification_row(name, pd.Series(y_cls), pred))
    clf = pd.DataFrame(clf_rows, columns=["model", "accuracy", "precision", "recall", "f1"])
    best_clf_name = clf.sort_values("f1", ascending=False).iloc[0]["model"]
    best_clf = classifiers[best_clf_name].fit(X, y_cls)
    joblib.dump(best_clf, ARTIFACT_ROOT / "classifier.joblib", compress=3)

    cm = confusion_matrix(y_cls, clf_preds[best_clf_name], labels=[0, 1])
    confusion = pd.DataFrame(
        cm,
        index=["Actual Decrease/Same", "Actual Increase"],
        columns=["Predicted Decrease/Same", "Predicted Increase"],
    )

    if best_clf_name == "Random Forest Classifier":
        imp = best_clf.feature_importances_
    else:
        imp = np.abs(best_clf.named_steps["model"].coef_[0])
    feature_importance = (
        pd.DataFrame({"feature": CLASS_FEATURES, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    reg_rows = []
    for name, model in regressors.items():
        pred = cross_val_predict(model, X, y_reg, cv=cv, groups=groups)
        reg_rows.append(_regression_row(name, pd.Series(y_reg), pred))
    reg = pd.DataFrame(reg_rows, columns=["model", "MAE_crore", "RMSE_crore", "R2"])
    best_reg_name = reg.sort_values("RMSE_crore").iloc[0]["model"]
    best_reg = regressors[best_reg_name].fit(X, y_reg)
    joblib.dump(best_reg, ARTIFACT_ROOT / "regressor.joblib", compress=3)

    temporal_metrics = _temporal_holdout(trans, best_clf_name, best_reg_name, random_state)

    latest = latest.dropna(subset=CLUSTER_FEATURES).copy()
    latest = latest[latest[CLUSTER_FEATURES].ge(0).all(axis=1)].copy()
    latest["log_disbursement"] = np.log1p(latest["digital_disbursement_crore"])
    latest["log_loan_count"] = np.log1p(latest["digital_loan_count"])
    latest["log_average_ticket"] = np.log1p(latest["average_ticket_rupees"])
    cluster_features = ["log_disbursement", "log_loan_count", "log_average_ticket"]
    k_scaler = StandardScaler().fit(latest[cluster_features])
    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=20)
    latest["cluster"] = kmeans.fit_predict(k_scaler.transform(latest[cluster_features])) + 1
    joblib.dump(k_scaler, ARTIFACT_ROOT / "kmeans_scaler.joblib", compress=3)
    joblib.dump(kmeans, ARTIFACT_ROOT / "kmeans.joblib", compress=3)

    profiles = (
        latest.groupby("cluster")
        .agg(
            nbfc_count=("nbfc_respondent_id", "count"),
            mean_disbursement_crore=("digital_disbursement_crore", "mean"),
            mean_loan_count=("digital_loan_count", "mean"),
            mean_average_ticket_rupees=("average_ticket_rupees", "mean"),
        )
        .reset_index()
        .sort_values("cluster")
    )

    anomaly_features = ["digital_disbursement_crore", "digital_loan_count", "average_ticket_rupees"]
    anomaly_scaler = StandardScaler().fit(latest[anomaly_features])
    contamination = min(max(6 / max(len(latest), 1), 0.01), 0.49)
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=random_state, n_jobs=-1)
    latest["anomaly_flag"] = iso.fit_predict(anomaly_scaler.transform(latest[anomaly_features]))
    latest["anomaly_score"] = iso.decision_function(anomaly_scaler.transform(latest[anomaly_features]))
    joblib.dump(anomaly_scaler, ARTIFACT_ROOT / "isolation_scaler.joblib", compress=3)
    joblib.dump(iso, ARTIFACT_ROOT / "isolation_forest.joblib", compress=3)

    clf.to_csv(METRICS_ROOT / "classification_metrics.csv", index=False)
    reg.to_csv(METRICS_ROOT / "regression_metrics.csv", index=False)
    confusion.to_csv(METRICS_ROOT / "classification_confusion_matrix.csv")
    feature_importance.to_csv(METRICS_ROOT / "classification_feature_importance.csv", index=False)
    profiles.to_csv(METRICS_ROOT / "kmeans_profiles.csv", index=False)
    latest.to_csv(METRICS_ROOT / "latest_clusters_anomalies.csv", index=False)
    temporal_metrics.to_csv(METRICS_ROOT / "temporal_holdout_metrics.csv", index=False)

    metadata = {
        "schema_version": "2026-10-01",
        "best_classifier": best_clf_name,
        "best_regressor": best_reg_name,
        "transition_rows": int(len(trans)),
        "nbfc_groups": int(trans["nbfc_respondent_id"].nunique()),
        "latest_complete_observations": int(len(latest)),
        "anomaly_flags": int((latest["anomaly_flag"] == -1).sum()),
        "classification_metrics": clf.set_index("model").to_dict(orient="index"),
        "regression_metrics": reg.set_index("model").to_dict(orient="index"),
        "temporal_holdout": temporal_metrics.astype(object).where(pd.notna(temporal_metrics), None).to_dict(orient="records"),
        "random_state": random_state,
        "group_cv": "5-fold GroupKFold grouped by nbfc_respondent_id",
        "temporal_holdout_period": str(trans["to_period"].iloc[-1]),
        "feature_columns": CLASS_FEATURES,
        "target_columns": ["next_disbursement_increased", "next_disbursement_crore"],
        "serving_boundary": "Aggregate analytical scoring only; not a credit approval or risk-decision system.",
    }
    (METRICS_ROOT / "model_registry.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return TrainingOutputs(clf, reg, confusion, feature_importance, profiles, latest, temporal_metrics, metadata)
