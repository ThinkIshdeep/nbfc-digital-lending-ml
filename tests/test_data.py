from nbfc_ml.data import build_transition_dataset, load_core_data, validate_core_data


def test_core_dataset_shape_and_uniqueness():
    df = load_core_data()
    validate_core_data(df)
    assert len(df) == 305
    assert df["nbfc_respondent_id"].nunique() == 61
    assert df.duplicated(["nbfc_respondent_id", "period"]).sum() == 0


def test_transition_dataset_shape():
    df = load_core_data()
    transitions = build_transition_dataset(df)
    assert len(transitions) == 135
    assert transitions["nbfc_respondent_id"].nunique() == 51
    assert set(transitions["next_disbursement_increased"].unique()) <= {0, 1}


def test_temporal_holdout_artifact():
    import pandas as pd
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    metrics = pd.read_csv(root / "artifacts/metrics/temporal_holdout_metrics.csv")
    predictions = pd.read_csv(root / "artifacts/metrics/temporal_holdout_predictions.csv")
    assert set(metrics["task"]) == {"classification", "regression"}
    assert len(predictions) == 51
