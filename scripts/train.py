from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbfc_ml.pipeline import run_pipeline

if __name__ == "__main__":
    result = run_pipeline()
    print("\nNBFC ML TRAINING COMPLETE")
    for key, value in result.items():
        print(f"{key}: {value}")
