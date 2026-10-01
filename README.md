# NBFC Digital Lending — Machine Learning Pipeline

A reproducible Python project for analysing **digital loan disbursement patterns in Non-Banking Financial Companies (NBFCs)** using public RBI data.

The repository contains the complete path from source snapshots to cleaned data, transition features, model training, evaluation, persisted artifacts, and HTTP serving.

> **Boundary:** this is an aggregate research/portfolio system, not a production credit-underwriting model. The public data do not expose proprietary NBFC analytics usage or borrower-level credit outcomes.

## Results

The packaged core dataset contains **305 respondent-period cells across 61 visible RBI respondent rows**. The supervised ML table contains **135 complete current→next-period transitions across 51 respondent groups**.

| Task | Selected model | Group-aware CV | Final chronological check |
|---|---|---:|---:|
| Direction classification | Random Forest | F1 **0.809** | F1 **0.444** |
| Amount regression | Gradient Boosting | RMSE **₹1,112.69 Cr**, R² **0.221** | RMSE **₹1,795.42 Cr**, R² **-4.356** |

The second column of results is deliberately shown. The model looks reasonable under entity-group cross-validation but generalises poorly to the final observed period. That is important evidence about the limits of this dataset, not something to hide.

The unsupervised layer produces **3 K-Means segments** and **6 candidate anomaly flags** for the latest complete observations.

## What is being predicted

The supervised model uses four current-period features:

- `current_disbursement_crore`
- `current_loan_count`
- `current_average_ticket_rupees`
- `period_index`

It predicts:

1. **Direction:** whether next-period reported disbursement is higher than the current period.
2. **Amount:** next-period reported disbursement in ₹ crore.

The clustering and anomaly components operate on current/latest-period lending behaviour. None of these outputs is a borrower credit decision.

## Evaluation

### Entity-aware cross-validation

The primary benchmark uses five-fold `GroupKFold` with `nbfc_respondent_id` as the group. Every transition for one respondent stays in the same fold.

Metrics come from out-of-fold predictions rather than training predictions.

### Chronological holdout

The final observed transition ending **31 December 2020** is held out as a separate test. Model selection still follows the primary GroupKFold benchmark; the selected models are then fit on earlier transitions and checked on the final period.

Two simple baselines are reported for the chronological test:

- majority class for direction prediction
- persistence (`next amount = current amount`) for amount prediction

Full details are in [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Repository

```text
.
├── .github/                 # CI + issue/PR templates
├── academic/                # Report, presentation, viva guide
├── artifacts/
│   ├── figures/             # Analysis and model-comparison figures
│   ├── metrics/             # Metrics, predictions, registry
│   └── models/              # Models used by the API
├── configs/                 # Versioned training configuration
├── dashboard/               # Standalone interactive HTML dashboard
├── data/
│   ├── raw/rbi/             # Public source snapshots
│   └── processed/           # Clean analytical tables
├── docs/                    # Architecture, provenance, evaluation, model card
├── scripts/                 # Train, score, verify, manifest utilities
├── src/nbfc_ml/             # Reusable Python package
├── tests/                   # Data, API and predictor tests
├── Dockerfile
├── Makefile
├── pyproject.toml
└── requirements*.txt
```

## Quick start

### Environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### Install

```bash
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

### Train

```bash
python scripts/train.py
```

The training command validates the source-derived data, reconstructs the current→next transition table, evaluates candidate supervised models with GroupKFold, runs the final chronological holdout, fits serving artifacts, performs clustering/anomaly detection, and writes results under `artifacts/`.

### Verify

```bash
python scripts/verify_repo.py
```

### Test

```bash
pytest -q
```

### Local score

```bash
python scripts/predict.py --disbursement 100 --loan-count 10000 --avg-ticket 10000 --period-index 3
```

## API

Start the local service:

```bash
uvicorn nbfc_ml.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Process health |
| GET | `/ready` | Artifact/model readiness |
| GET | `/v1/model-info` | Selected models + evaluation metadata |
| POST | `/v1/predict` | Analytical scoring |
| GET | `/docs` | OpenAPI / Swagger UI |

Example request:

```bash
curl -X POST "http://localhost:8000/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "current_disbursement_crore": 100.0,
    "current_loan_count": 10000,
    "current_average_ticket_rupees": 10000,
    "period_index": 3
  }'
```

The API loads persisted artifacts at startup/request time. It does not retrain models for individual requests.


## Product layer

The repository also includes a model-backed web application. FastAPI serves the prediction and dashboard endpoints and the frontend at `/`. The browser never retrains or reimplements the model; it reads the same persisted artifacts used by the CLI/API.

### Application views

- **Overview** — observed disbursement and loan-count trends, product mix and model generalisation health.
- **Predict** — analytical scenario scoring through `POST /v1/predict`.
- **Model Lab** — candidate metrics, selected models, feature importance and chronological holdout results.
- **Segments** — latest-period K-Means profiles and candidate anomaly observations.
- **Data** — dataset coverage, feature schema and explicit research boundary.

### Dashboard endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/dashboard/summary` | KPI and dataset summary |
| GET | `/v1/dashboard/trends` | Period-level activity metrics |
| GET | `/v1/dashboard/product-mix` | Lending-purpose mix |
| GET | `/v1/dashboard/model-lab` | Model metrics and holdout results |
| GET | `/v1/dashboard/segments` | Cluster profiles |
| GET | `/v1/dashboard/anomalies` | Candidate anomaly observations |

Run the application locally with:

```bash
uvicorn nbfc_ml.api:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/`.

## Docker

```bash
docker build -t nbfc-digital-lending-ml .
docker run --rm -p 8000:8000 nbfc-digital-lending-ml
```

Or:

```bash
docker compose up --build
```

## Data provenance

The core digital-lending table comes from RBI Working Group on Digital Lending material. Supporting datasets include RBI NBFC sectoral credit, RBI-DPI, the RBI NBFC registry, and related public RBI tables.

The repository keeps source snapshots separate from analytical derivatives. [`docs/DATASET_MANIFEST.json`](docs/DATASET_MANIFEST.json) records byte counts and SHA-256 hashes for every packaged raw/processed data file.

The RBI digital-lending source heading refers to **62 NBFCs**, while the visible respondent rows retained in this package are **61**. No synthetic respondent is added.

See [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).

## Engineering choices

The project is intentionally small enough to inspect end-to-end.

- Dependencies are pinned.
- Training has an explicit seed.
- Data quality checks are automated.
- Entity-aware validation and a chronological holdout are both implemented.
- Model artifacts are persisted and served without retraining requests.
- The model registry records the selected models, feature schema, targets, validation protocol and serving boundary.
- CI runs linting, Python compilation, tests, and repository-integrity checks.
- Dataset source/processed hashes are kept in a versioned manifest.

There is no MLflow/DVC/cloud service requirement. For this small, fully packaged dataset, adding those systems would add setup without adding evidence.

See [`docs/ENGINEERING_NOTES.md`](docs/ENGINEERING_NOTES.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Limitations

The dataset is aggregate, respondent-anonymised, and short in time span. It does not directly measure internal analytics-system adoption, borrower credit risk, approval quality, repayment, default, processing time, or profitability.

The negative final-period R² is retained because it is a real limitation of the observed sample. The project should therefore be presented as a **research and portfolio ML pipeline**, not as a validated lending production model.

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) and [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Academic material

The `academic/` directory contains the field-project report, presentation and viva study guide used alongside the technical repository.

## License

MIT. See [`LICENSE`](LICENSE).
