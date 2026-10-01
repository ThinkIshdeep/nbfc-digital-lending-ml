from nbfc_ml.predictor import ModelBundle


def test_saved_models_produce_valid_output():
    bundle = ModelBundle()
    result = bundle.predict(100.0, 10_000.0, 10_000.0, 3)
    assert isinstance(result["next_disbursement_increased"], bool)
    assert result["predicted_next_disbursement_crore"] >= 0
    assert result["cluster"] in {1, 2, 3}
    assert result["anomaly_flag"] in {"potentially_unusual", "within_observed_pattern"}
