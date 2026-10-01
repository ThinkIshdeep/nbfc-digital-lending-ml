from fastapi.testclient import TestClient
from nbfc_ml.api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict():
    payload = {
        "current_disbursement_crore": 100.0,
        "current_loan_count": 10000,
        "current_average_ticket_rupees": 10000,
        "period_index": 3,
    }
    r = client.post("/v1/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "predicted_next_disbursement_crore" in data
    assert "cluster" in data


def test_ready_and_model_info():
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"

    info = client.get("/v1/model-info")
    assert info.status_code == 200
    payload = info.json()
    assert payload["best_classifier"] == "Random Forest Classifier"
    assert payload["best_regressor"] == "Gradient Boosting Regressor"
