# Operations Runbook

## Local service

```bash
uvicorn nbfc_ml.api:app --host 0.0.0.0 --port 8000
```

## Health check

```bash
curl http://localhost:8000/health
```

Expected status: HTTP 200 with `status: ok`.

## API docs

Open `http://localhost:8000/docs` in a browser.

## Retraining

Run `python scripts/train.py`. The pipeline writes new artifacts deterministically using `random_state=42`.

## Rollback

Because artifacts are committed/archived by release, restore the previous `artifacts/models/` and matching `artifacts/metrics/model_registry.json` set as one unit.

## Production hardening required before external deployment

Authentication, TLS, rate limiting, request logging, monitoring, dependency scanning, secret management, model versioning, artifact signing, access controls, and human review should be added before internet-facing deployment.
