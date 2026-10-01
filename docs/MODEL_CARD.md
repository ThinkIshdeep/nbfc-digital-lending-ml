# Model Card

## Model family

This project packages four analytical model families:

1. Random Forest / Logistic classification for **next-period increase direction**.
2. Gradient Boosting / tree regression for **next-period disbursement amount**.
3. K-Means for **latest-period behavioural segmentation**.
4. Isolation Forest for **candidate unusual-observation detection**.

## Intended users

Students, analysts, researchers, ML engineers and reviewers inspecting aggregate NBFC digital-lending patterns.

## Inputs

Classification/regression inputs:

- current disbursement in ₹ crore,
- current loan count,
- current average ticket in ₹,
- period index.

Cluster/anomaly inputs:

- current/latest disbursement,
- loan count,
- average ticket.

## Validation

The supervised models use 5-fold `GroupKFold`, grouping on `nbfc_respondent_id`. This is intentional because each respondent can contribute observations in multiple periods.

## Metrics

Classification: accuracy, precision, recall, F1.

Regression: MAE, RMSE, R².

Clustering/anomaly detection are exploratory outputs and do not have a claim of external ground-truth accuracy.

## Known limitations

- Only five reporting periods are available in the core table.
- The dataset is aggregate and respondent-anonymised.
- The sample is small for production ML.
- Public data do not contain proprietary internal analytics adoption measurements.
- Reporting coverage may differ between periods.
- Cross-validation metrics may be unstable with small group counts.

## Fairness / risk considerations

No borrower-level protected attributes are present in the packaged modelling data. This does not make the system suitable for credit decisions. Production lending use would require borrower-level validation, fairness testing, explainability controls, adverse-action requirements where applicable, monitoring, human oversight, and regulatory/legal review.

## Safety boundary

The API should be treated as an **analytical pattern engine**, not an automated lending decision engine.


## Additional validation boundary

A chronological holdout for the final observed transition is also computed. Its purpose is to test whether the selected supervised models generalise to a later period. Because the core dataset has only five reporting periods, this is a small diagnostic test and should not be interpreted as production validation.
