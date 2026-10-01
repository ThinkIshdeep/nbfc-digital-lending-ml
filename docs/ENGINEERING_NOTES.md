# Engineering Notes

This repository is intentionally small enough to understand end-to-end while following a few practices that matter in real ML systems.

## Reproducibility

- Python dependencies are pinned.
- The training seed is explicit.
- Source snapshots are stored separately from processed analytical tables.
- Training writes a model registry and dataset hash.
- CI runs tests, linting, repository integrity checks and Python compilation.

## Leakage control

The ML rows are created as current→next-period transitions. Only current-period features are used as supervised inputs. The next-period amount/direction is the target.

Cross-validation groups rows by respondent. A final chronological holdout is evaluated separately.

## Artifact contract

`artifacts/models/` contains only models required for serving. `artifacts/metrics/` contains evaluation outputs and the model registry. The API reads these artifacts without retraining at request time.

## Serving boundary

The HTTP service is a local analytical scoring service. It does not contain authentication, rate limiting, borrower PII, approval logic, or credit policy. Those concerns are intentionally outside this academic portfolio project.

## Why no MLflow/DVC dependency?

Adding a platform is not automatically an engineering improvement. This dataset is small and fully included in the repository. A simple registry JSON, pinned dependencies, source manifest, deterministic training command and CI checks provide clearer reproducibility here without requiring a hosted tracking or storage service.
