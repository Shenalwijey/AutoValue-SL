import os
import runpy
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent / "car-price-prediction"


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    os.chdir(PROJECT_DIR)
    if "--train" in sys.argv:
        sys.path.insert(0, str(PROJECT_DIR))
        from src.predict import train_and_store_model

        train_and_store_model()
    else:
        runpy.run_path(str(PROJECT_DIR / "app.py"), run_name="__main__")
