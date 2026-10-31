# Local setup · current river dashboard

Run from `D:\ADS_MINI\Flood-risk-prediction-system\RiskRadar`.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

Dashboard: http://127.0.0.1:8501/. Only Streamlit is required; the former API on port 8000 is not used. `.\start-riskradar.ps1` starts a hidden local process, refuses an occupied port and saves `.run/processes.json` and logs. Do not stop unrelated processes to free a port.

For a fresh Python 3.12 environment, install `requirements-river.txt`. It pins the direct packages used by this version. `requirements-local.lock.txt` preserves the earlier complete environment resolution; sampling/CatBoost dependencies in `requirements-training.txt` belong to older studies and are not required by the new regressor.

The existing saved artifacts load without downloading or retraining. Forecast dates are historical, selected from eligible 2023–2025 windows. Rainfall history uses its existing catalog and weather cache; other locations may need a historical archive download. New river forecasts never call the old annual-risk API.

## Verify

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/verify_river_stage.py
.\.venv\Scripts\python.exe scripts/verify_river_dashboard.py
```

Streamlit interaction checks need local loopback networking on Windows. The first command currently passes 24 tests, the third 27 interaction checks. `tests/verify_dashboard.py` targets the archived annual-risk app and is retained as historical verification, not a verifier for this entrypoint.

## Reproduce the experiment

`scripts/fetch_river_stage.py` acquires only the selected three gauges; manifests and checksums preserve provenance. `scripts/train_river_stage.py prepare`, `select`, and `evaluate` are sequential stages. They refuse to overwrite existing frozen protocol, selection or final results. Do not remove the guard or select/tune again using final test results. A future study needs a new prespecified version and fresh evaluation period.

`scripts/document_river_stage.py` rebuilds current Markdown documentation from frozen results without fitting or selecting models. Runtime package versions are saved under `experiments/river_stage_v1/runtime_versions.json`.

## Important files

- `river_forecast.py`: conservative QC, causal features, numeric prediction and stage rules.
- `river_page.py`, `app.py`, `assets/dashboard.css`: current two-page dashboard.
- `data/river_stage/processed/model_dataset.csv`: the merged, eligible modelling table.
- `experiments/river_stage_v1/forecast_bundle.joblib`: selected pipeline, feature list, station thresholds and calibrated error ranges.
- `experiments/river_stage_v1/selection.json`, `final_results.json`, `test_predictions.csv`: frozen evidence.
- `legacy/annual_risk_dashboard.py`: source snapshot of the preceding app. Its paths assume the original root; restore to a separate root before attempting to run it.

Original setup instructions and old API details are retained in `legacy/LOCAL_SETUP_before_river_forecast.md`.
