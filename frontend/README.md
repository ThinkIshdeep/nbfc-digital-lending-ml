# NBFC Digital Lending Intelligence — Frontend

A lightweight web interface for exploring the NBFC digital-lending analysis and interacting with the machine-learning models through the project API.

The frontend is intentionally dependency-free. It uses standard HTML, CSS, and JavaScript and communicates with the FastAPI backend over HTTP. Model inference is performed by the backend using the trained model artifacts; no prediction logic is duplicated in the browser.

## What the frontend does

The application provides a single interface for five parts of the project.

### Overview

The main analytics view provides a high-level picture of the underlying dataset.

It includes:

* Core digital-lending observations
* ML transition observations
* Latest-period disbursement
* Average ticket size
* Disbursement trend
* Digital loan-count trend
* Loan-purpose mix
* Model generalisation results

The values shown here are loaded from the project API and analytical outputs rather than being hard-coded into the interface.

### Prediction

The prediction page allows a user to enter a current NBFC-period observation:

* Current disbursement amount
* Current loan count
* Current average ticket size
* Period index

The frontend sends these values to:

```text
POST /v1/predict
```

The response includes:

* Predicted direction of next-period disbursement
* Probability associated with the classification output
* Estimated next-period disbursement
* Behaviour segment
* Anomaly signal
* Model/version information

This is an analytical prediction interface, not an automated lending or credit-approval system.

### Model Lab

The Model Lab exposes the model evaluation results used in the project.

It covers:

* Classification model comparison
* Regression model comparison
* Selected production artifacts
* Feature importance
* Group-aware validation results
* Chronological out-of-time evaluation

The chronological evaluation is shown separately from grouped cross-validation because the two answer different questions. Grouped validation helps control for repeated observations from the same NBFC, while the final-period holdout gives a more realistic indication of how the models behave on a later observation.

### Segments

This view presents the unsupervised analysis performed on the latest complete period.

It includes:

* K-Means cluster profiles
* Number of NBFC observations in each segment
* Mean disbursement
* Mean loan count
* Mean average ticket
* Isolation Forest anomaly candidates

An anomaly flag is treated as an observation requiring review. It is not automatically interpreted as an error, fraud indicator, or credit-risk event.

### Data

The Data view documents the analytical inputs and their boundaries.

The project currently works with:

| Dataset                        | Observations | Purpose                       |
| ------------------------------ | -----------: | ----------------------------- |
| RBI digital-lending core table |          305 | Primary lending analysis      |
| ML transition dataset          |          135 | Classification and regression |
| NBFC groups used by ML         |           51 | Entity-aware validation       |
| RBI sectoral credit            |           75 | Credit-market context         |
| RBI-DPI                        |           13 | Digitalisation context        |
| NBFC registry                  |        9,091 | Entity/reference information  |

The public data are aggregate observations. They do not provide borrower-level income, credit score, repayment history, approval outcomes, processing time, or proprietary NBFC analytics-system usage.

## Application architecture

The browser is responsible for presentation and user interaction.

```text
┌──────────────────────────────┐
│        Web Frontend          │
│      HTML / CSS / JS         │
└──────────────┬───────────────┘
               │ HTTP
               ▼
┌──────────────────────────────┐
│        FastAPI Backend       │
│   Dashboard + Prediction API │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Trained Artifacts      │
│  Classification / Regression │
│  Clustering / Anomaly Model  │
└──────────────────────────────┘
```

The separation is intentional:

* The frontend does not contain the trained models.
* The frontend does not recreate preprocessing.
* The browser does not calculate model predictions independently.
* The API is the single inference interface.

This keeps the analytical logic in the Python application rather than spreading it across two implementations.

## API endpoints used by the frontend

### Health

```text
GET /health
```

Basic service health check.

### Readiness

```text
GET /ready
```

Checks whether the required model artifacts can be loaded.

### Model information

```text
GET /v1/model-info
```

Returns information about the available model artifacts and versions.

### Prediction

```text
POST /v1/predict
```

Runs classification, regression, segmentation and anomaly analysis for an input observation.

### Dashboard data

```text
GET /v1/dashboard/summary
GET /v1/dashboard/trends
GET /v1/dashboard/product-mix
GET /v1/dashboard/model-lab
GET /v1/dashboard/segments
GET /v1/dashboard/anomalies
```

These endpoints provide the datasets required by the frontend views.

## Running locally

The frontend is served as a static application.

Start the backend first:

```powershell
& "$env:USERPROFILE\.venvs\nbfc-ml\Scripts\python.exe" -m uvicorn nbfc_ml.api:app --host 127.0.0.1 --port 8000 --reload
```

Then open a second terminal:

```powershell
cd frontend
& "$env:USERPROFILE\.venvs\nbfc-ml\Scripts\python.exe" -m http.server 5500
```

Open:

```text
http://127.0.0.1:5500
```

The API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

The frontend expects the backend API to be available on:

```text
http://127.0.0.1:8000
```

## Project structure

```text
frontend/
├── index.html
├── styles.css
├── app.js
└── README.md
```

### `index.html`

Defines the application structure, navigation, views, forms and dashboard containers.

### `styles.css`

Contains the complete visual system for the application, including layout, cards, tables, controls, responsive behavior and status states.

### `app.js`

Handles:

* API requests
* View navigation
* Data rendering
* Prediction form submission
* Loading states
* Error states
* Table generation
* Chart rendering
* Model-result presentation

## Data behavior

The frontend does not fabricate current market data.

Values displayed by the application come from one of three sources:

1. Processed project datasets
2. Persisted ML evaluation artifacts
3. Live responses from the local FastAPI service

The interface may display analytical labels such as cluster names or anomaly status, but those labels are derived from the underlying project outputs.

## Error handling

The application handles common service failures such as:

* Backend unavailable
* Prediction request failure
* Missing dashboard data
* Invalid input
* Model service not ready

When the API is unavailable, the frontend displays an error state instead of silently substituting fake values.

## Scope

This application demonstrates an end-to-end data and machine-learning workflow using public NBFC and RBI datasets.

It is intended for:

* Academic evaluation
* Data-science portfolio demonstration
* ML workflow inspection
* Exploratory lending analysis

It is not intended to:

* approve or reject real borrowers
* determine creditworthiness
* replace an NBFC underwriting system
* provide regulatory or financial advice
* identify fraud from anomaly flags alone

## Why the interface separates model performance from model claims

The project reports both validation results and a chronological holdout evaluation.

This distinction matters because a model can perform reasonably well under one validation design while generalising poorly to a later period. The frontend therefore exposes the evaluation results instead of presenting a single headline accuracy number as proof of model quality.

The machine-learning outputs should be interpreted as a project-scale analytical experiment given the limited number of observations and available features.

## Technology

**Frontend**

* HTML5
* CSS3
* Vanilla JavaScript

**Backend**

* Python
* FastAPI

**Machine Learning**

* pandas
* NumPy
* scikit-learn
* joblib

No frontend framework or client-side ML runtime is required.

## Related project components

The frontend is one layer of the larger repository.

```text
data/        Source and processed datasets
src/         Python data and ML implementation
artifacts/   Trained models and evaluation outputs
tests/       Automated tests
frontend/    This web application
docs/        Engineering and methodology documentation
academic/    Field-project report and presentation material
```

For the complete setup and ML workflow, see the repository root `README.md`.

## Research boundary

The central research question concerns the relationship between data-driven/digital lending activity and loan-disbursement patterns.

The available public datasets do not directly measure proprietary analytics usage by individual NBFCs. Therefore, the application presents observed patterns and predictive experiments rather than claiming that internal analytics systems caused changes in disbursement.

That distinction is part of the project methodology, not a limitation hidden from the user.
