from fastapi.testclient import TestClient

from nbfc_ml.api import app

client = TestClient(app)


def test_dashboard_summary():
    response = client.get("/v1/dashboard/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["core_cells"] == 61
    assert body["transition_rows"] == 135


def test_dashboard_model_lab():
    response = client.get("/v1/dashboard/model-lab")
    assert response.status_code == 200
    body = response.json()
    assert body["best_classifier"] == "Random Forest Classifier"
    assert body["best_regressor"] == "Gradient Boosting Regressor"
    assert body["temporal"]["classifier"]["f1"] < body["classification"][1]["f1"]


def test_frontend_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "NBFC Digital Lending Intelligence" in response.text
