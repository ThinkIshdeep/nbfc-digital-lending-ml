from __future__ import annotations

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("NBFC_ML_DATA_ROOT", PROJECT_ROOT / "data" / "processed"))
ARTIFACT_ROOT = Path(os.getenv("NBFC_ML_ARTIFACT_ROOT", PROJECT_ROOT / "artifacts" / "models"))
METRICS_ROOT = PROJECT_ROOT / "artifacts" / "metrics"
FIGURES_ROOT = PROJECT_ROOT / "artifacts" / "figures"

PERIODS = ["FY2017", "FY2018", "FY2019", "FY2020", "As on Dec 31 2020"]
CORE_DATASET = DATA_ROOT / "digital_lending_long_clean.csv"
TRANSITION_DATASET = DATA_ROOT / "NBFC_ML_Transition_Dataset.csv"
