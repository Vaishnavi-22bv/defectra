"""
Run Defectra prediction from the project root.

Usage:
    python predictions/predict.py
"""

from pathlib import Path
import runpy
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_DIR / "backend"
BACKEND_PREDICT = BACKEND_DIR / "predictions" / "predict.py"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    runpy.run_path(str(BACKEND_PREDICT), run_name="__main__")
except ModuleNotFoundError as exc:
    missing_module = exc.name or "a required package"
    raise SystemExit(
        f"Missing Python package: {missing_module}\n"
        "Use the backend environment from the project root:\n"
        r"    .\backend\.venv\Scripts\python.exe predictions\predict.py"
    ) from exc
