# Contributing

## Workflow

1. Create a focused branch.
2. Keep data transformations deterministic and documented.
3. Do not introduce synthetic observations into the official-data pipeline.
4. Add or update a test for changed behaviour.
5. Run `pytest -q` before committing.
6. Update documentation when changing features, data assumptions, or model behaviour.

## Data changes

Never overwrite public source snapshots with altered files. Add the new source under `data/raw/rbi/`, document the source URL and acquisition date in `docs/DATA_PROVENANCE.md`, then regenerate `data/processed/` outputs.

## Model changes

Any change to features, target definitions, cross-validation strategy, or preprocessing should be reflected in `docs/MODEL_CARD.md` and the generated metrics registry.
