# Run the corrected rainfall and flood analysis app

Working directory: `D:\ADS_MINI\Flood-risk-prediction-system\RiskRadar`.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

Open http://127.0.0.1:8501/. `.\start-riskradar.ps1` starts the same dashboard in a hidden background process with logs/PID under `.run/`, refuses occupied ports and does not stop unrelated services. The original API is not required.

A fresh Python 3.12 environment can install pinned direct packages from requirements-rainfall-flood.txt. Saved datasets and models already exist; main analysis/inventory pages work locally. Rainfall history has a cached near-Mumbai series; uncached locations need historical archive access.

## Verify

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/verify_rainfall_flood_release.py
.\.venv\Scripts\python.exe scripts/verify_rainfall_flood_dashboard.py
.\.venv\Scripts\python.exe tests/verify_history_page.py
```

Streamlit AppTest requires local loopback networking on Windows. tests/verify_dashboard.py is a compatibility entrypoint for the current verifier; the old annual-risk verifier is archived in legacy/verify_annual_dashboard.py.

## Reproduction and files

scripts/prepare_rainfall_flood_release.py joins existing sources and packages previously frozen severity models; it refuses to overwrite a frozen release. scripts/document_rainfall_flood_release.py rebuilds Markdown documentation from saved evidence. Original training remains in scripts/compare_indofloods_xgboost.py and experiments/indofloods_severity_v1/; its staged commands refuse to overwrite frozen selection/test results. A new study requires a new prespecified version and fresh evaluation data.

rainfall_flood.py defines shared data/inference semantics; rainfall_flood_page.py and app.py render the app. data/rainfall_flood/ holds matched_events.csv, flood_inventory.csv and manifest.json. experiments/rainfall_flood_release_v1/ holds the inference comparison bundle, manifests and verification. reports/Final_Rainfall_and_Flood_Risk_Analysis_Report.docx is the current report; final_ps_work/ retains its reproducible evidence and visual QA.

The river entrypoint is preserved in legacy/river_forecast_entrypoint.py and its original modules/models remain unchanged. Earlier setup is retained in legacy/LOCAL_SETUP_river_stage_release.md.

## Rainfall-based prediction

Open Prediction in the top navigation. This uses the saved near-Mumbai point only, with 14 past values and a next-three-day categorical target. `rainfall_risk.py`, `prediction_page.py`, `scripts/train_rainfall_risk.py` and `experiments/rainfall_risk_v1/` preserve the new lifecycle. Training refuses to overwrite frozen results. Run `scripts/verify_rainfall_risk.py` to verify causal replay and thresholds. No new weather locations are downloaded.
