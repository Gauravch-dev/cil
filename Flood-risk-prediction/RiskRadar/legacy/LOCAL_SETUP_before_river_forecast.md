# Local RiskRadar setup

Source: https://github.com/nitinc264/Flood-prediction-Model.git

Cloned commit: `2d823f3692e891ac92de4435bbe2c006131c30b3`.
Active project: `D:\ADS_MINI\Flood-risk-prediction-system\RiskRadar`.

The previous rainfall project and FloodSense files, datasets, models and environments were permanently removed after explicit confirmation. The parent folder's original Git history and editor settings remain. This clone has its own original Git remote and history.

## Start

In PowerShell, from this folder:

```powershell
.\start-riskradar.ps1
```

Dashboard: http://127.0.0.1:8501/
API docs: http://127.0.0.1:8000/docs

The launcher starts both services in hidden windows and writes logs and process IDs under `.run/`. It refuses occupied ports. The application is bound to localhost. For manual startup in two terminals:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501 --server.headless=true --browser.gatherUsageStats=false
```

## Environment

Python 3.12.10 in `.venv/`. `requirements-local.txt` matches XGBoost 3.2.0 and scikit-learn 1.6.1 to versions embedded in the supplied model/encoder files. Plotly 6.9.0 retains the tested chart stack. The working resolution is frozen in `requirements-local.lock.txt`.

To reproduce in a new Python 3.12 environment, run `python -m pip install -r requirements-local.lock.txt`.

The primary UI has been redesigned in `app.py`, `dashboard_ui.py`, `assets/dashboard.css` and `.streamlit/config.toml`. No model retraining or changes to `api.py`, `flood_fast.py`, the pretrained artifacts or locations CSV were made. The model remains the repository's supplied model; its reported training accuracy was not independently reproduced. The supplied training helper was not run or used to overwrite the model.

## Verification

The maintained dashboard verifier is `tests/verify_dashboard.py`. These checks cover top navigation, removal of technical panels, model/API preservation, input labels, actual predictions, rainfall comparisons, example selection/deselection, result retention, failed/invalid submissions, downloads and rainfall analysis while risk checks are offline. Old `.run/` scripts are historical snapshots and are not the maintained test suite.

The top navigation contains only **Rainfall history** and **Check flood risk**, with rainfall history as the landing page. There is no sidebar or model/connection/source-detail panel. Three rainfall controls are visible first on the risk page; remaining terrain, soil and climate inputs are under **Land & weather details** with help tooltips. All 14 inputs remain available. Presets are examples, not measurements of a selected location. Category labels retain the supplied encodings in requests. Predictions persist in the current browser session; the last eight checks can be compared. **Save result** downloads a CSV summary. The risk category, gauge and color scale represent the last checked conditions; editing inputs does not change that result until the user checks again.

The rainfall explorer slider queries the same model for the exact changed annual-rainfall value while holding all other submitted inputs fixed. A separate colored marker and card show the explored score, category and percentage-point change without overwriting the checked assessment. The line connects sampled rainfall scenarios, not dates; it is not feature importance or a future forecast. Input validation ensures annual rainfall is between one and twelve times the wettest-month rainfall. Zero rainfall is supported without an invalid slider range.

The simplified dashboard passes **39 integration checks, 0 failed**. The maintained verifier is `tests/verify_dashboard.py`; run `.\.venv\Scripts\python.exe tests/verify_dashboard.py` while both local services are running. Checks include the exact API response at the explorer value and preservation of the checked result/history.

The former Risk Map is preserved in `legacy/risk_map.py`, with its old styling in `legacy/dashboard.css`. Its unchanged source CSV contains 2,538 saved predictions at 468 coordinate pairs. It is not imported, routed or displayed by the current application.

Default example remains risk score 0.0779, LOW (7.8%). The wetter example produces 0.7840, HIGH (78.4%). These checks establish runtime functionality, not real-world predictive accuracy. The redesign replaces deprecated mapbox charts with MapLibre and replaces deprecated widget width arguments. The launcher uses polling file detection and automatic reruns so edits appear in the running UI.

## Historical rainfall and reproducible data preparation

Run `.\.venv\Scripts\python.exe scripts/prepare_historical_data.py` once to download the original public GloFlo dataset cited in the upstream README, verify its published checksum, audit it and save a station catalog. The script also retrieves daily ERA5 rainfall for the default source station near Mumbai, 2000–2025. To cache another source station, use `--station STATION_ID` with a valid ID from `data/processed/stations.csv`. The historical page supports all source countries and loads uncached locations on demand. Downloaded locations work offline. The initial global source is about 35 MB; one daily archive is about 180 KB. Raw downloads and weather caches are ignored by Git; the source catalog, data audit and acquisition script are retained so the data can be reproduced.

The page provides annual totals, deviations from the selected-period mean, a monthly heatmap, daily date-range zoom and a CSV download. Exact source details remain in the acquisition JSON on disk; only short attribution links are shown in the UI. Weather dates, units, array lengths, missing values and duplicate dates are validated. Incomplete months/years do not receive misleading totals. Historical rainfall is a gridded estimate near the station, not a catchment mean, and is not automatically used as classifier input.

See [MODEL_REPORT.md](MODEL_REPORT.md) for source attribution and the model evaluation audit. The source contains no binary flood/no-flood labels or reproducible original target-generation pipeline. The supplied model's advertised accuracy cannot be independently verified from these files. No fake labels or synthetic test metrics have been substituted.

Run checks from this folder after data preparation; integration checks require the local services to be running:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
.\.venv\Scripts\python.exe tests/verify_dashboard.py
.\.venv\Scripts\python.exe tests/verify_history_page.py
```

Latest result: **64 checks passed, 0 failed** (39 dashboard integrations, 15 historical integrations, 10 data unit tests). These counts are software validation, not ML accuracy.
