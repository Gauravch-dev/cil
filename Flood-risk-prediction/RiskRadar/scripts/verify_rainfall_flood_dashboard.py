"""Compatibility runner for the current two-page interface verification."""
from pathlib import Path
import runpy

if __name__=='__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/verify_two_page_dashboard.py'),run_name='__main__')
