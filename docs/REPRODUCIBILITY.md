# Reproducibility Runbook

## Clean rebuild

```bash
python -m venv .venv
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python scripts/train.py
pytest -q
python scripts/verify_repo.py
```

## Expected dataset counts

- Core visible respondent-period cells: **305**
- Visible respondent IDs: **61**
- Complete current→next transitions: **135**
- NBFC groups in ML transitions: **51**
- Complete latest-period observations: **58**
- Potentially unusual latest-period observations: **6**

Small floating-point differences can occur across operating systems or library versions; model structure and validation protocol should remain the same.

## Artifact outputs

After training:

```text
artifacts/models/
  classifier.joblib
  regressor.joblib
  kmeans.joblib
  kmeans_scaler.joblib
  isolation_forest.joblib
  isolation_scaler.joblib

artifacts/metrics/
  classification_metrics.csv
  regression_metrics.csv
  classification_confusion_matrix.csv
  classification_feature_importance.csv
  kmeans_profiles.csv
  latest_clusters_anomalies.csv
  model_registry.json
```

## Operational checks

Before sharing a training result, confirm:

1. The core data hash changed only when expected.
2. Row counts remain consistent or changes are documented.
3. GroupKFold is still used for supervised validation.
4. Metrics are generated from held-out predictions, not training predictions.
5. The model registry identifies the exact input data and feature schema.


## Evaluation files

The training run also writes:

```text
artifacts/metrics/
  temporal_holdout_metrics.csv
  temporal_holdout_predictions.csv
```

These show how the selected models behave on the final chronological transition separately from the primary group-aware cross-validation benchmark.
