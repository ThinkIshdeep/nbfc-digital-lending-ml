# Evaluation Protocol

The supervised models are evaluated in two different ways because they answer different questions.

## 1. Entity-aware cross-validation

The primary benchmark uses five-fold `GroupKFold`, with `nbfc_respondent_id` as the group. Every transition belonging to one respondent stays in the same fold.

This prevents the simplest entity leakage: the same NBFC appearing in both a training fold and its validation fold.

The reported metrics are out-of-fold predictions produced by `cross_val_predict`. They are not training-set scores.

## 2. Chronological holdout

The final observed transition (`As on Dec 31 2020`) is held out as a separate chronological test. Candidate-model selection still follows the primary GroupKFold benchmark; the selected model is then fit on earlier transitions and evaluated on the final period.

A chronological test is deliberately included because a model can perform well under grouped cross-validation while struggling to generalise to a later regime. With only five reporting periods, this test is small and should be treated as diagnostic evidence rather than a production validation set.

## Baselines

Two simple baselines are included in the chronological test:

- Majority class for direction prediction.
- Persistence (`next amount = current amount`) for disbursement regression.

The purpose is not to manufacture a high score. A model should be compared with a simple rule before anyone calls the model useful.

## What to report

For classification, report accuracy, precision, recall and F1. For regression, report MAE, RMSE and R². Always state the evaluation protocol and test period alongside the numbers.
