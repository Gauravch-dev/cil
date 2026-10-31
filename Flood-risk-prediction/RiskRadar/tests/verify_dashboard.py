"""Compatibility entrypoint for the current rainfall/flood dashboard verifier.

The former annual-risk checks are retained in legacy/verify_annual_dashboard.py.
"""
from pathlib import Path
import runpy

if __name__=="__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/verify_rainfall_flood_dashboard.py"),
                  run_name="__main__")
