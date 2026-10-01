# Architecture

```text
RBI public source snapshots
        │
        ▼
data/raw/rbi/
        │
        ▼
load_core_data() ──► validate_core_data()
        │
        ▼
NBFC respondent × period table (305 visible cells)
        │
        ├──────────────► descriptive plots / summaries
        │
        ▼
build_transition_dataset()
        │
        ▼
current period features + next-period target
        │
        ├── GroupKFold ──► Classification models
        │                       └─► best classifier → classifier.joblib
        │
        ├── GroupKFold ──► Regression models
        │                       └─► best regressor → regressor.joblib
        │
        ├── latest period ──► log transform + K-Means
        │                       └─► kmeans + scaler
        │
        └── latest period ──► Isolation Forest
                                └─► isolation model + scaler

Saved artifacts ──► FastAPI ──► /v1/predict
                    │
                    └──────────► /health
```

## Design principles

- **Separation of concerns:** data validation, feature construction, model training, scoring, and serving are separate modules.
- **Deterministic training:** fixed random seed and explicit model parameters.
- **Leakage control:** entity-aware cross-validation using respondent ID as the group.
- **Traceability:** source snapshots, processed derivatives, configuration, hashes, metrics, and saved model registry.
- **No borrower-level claims:** the models operate on aggregate respondent-period records.
