# RiskRadar · Maharashtra River-Level Forecast

A controlled dashboard for next-24-hour river-peak forecasting and historical rainfall analysis. Three gauges: Kurundwad, Narsingpur and Takli. Archived CWC observations drive a newly trained numeric XGBoost regressor; official station thresholds derive warning/danger status.

**This version replays historical forecasts; it does not fetch live 2026 readings or issue emergency alerts.** It predicts gauge stage, not district inundation or flood damage.

## Run locally

From this directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

Open [RiskRadar](http://127.0.0.1:8501/). Alternatively, `.\start-riskradar.ps1` launches the dashboard in a hidden background process and records logs/PID under `.run/`. The primary app does not require the old FastAPI service.

## Use

- **River forecast:** choose a gauge, year and available date. Inspect the forecast peak, stage, range and past readings. Example buttons use only observations available at the issue time. Optionally reveal the recorded outcome or download the forecast.
- **Rainfall history:** explore yearly totals/anomalies, monthly heatmaps and daily rainfall. Existing cached history works offline; an uncached location requires downloading its historical archive.

## ML lifecycle and honest results

Train 2014–2020, select 2021, calibrate ranges 2022, test 2023–2025. Compare 22 ML candidates across five algorithms and three simple baselines. Freeze XGBoost using validation alone. Final test: 1,156 forecasts; MAE **0.258 m**, RMSE **0.517 m**. **Trend baseline MAE is better (0.205 m)**; ML reduces RMSE but does not demonstrate overall superiority. There are only two observed danger episodes, of which one was predicted in advance. This is insufficient for an operational warning claim.

Read [MODEL_REPORT](MODEL_REPORT.md) for full results, [REVIEW](REVIEW.md) for limitations, and [LOCAL_SETUP](LOCAL_SETUP.md) for commands and files.

## Architecture

`CWC archive → conservative QC/hourly bins → causal trailing features → frozen regressor → numeric 24-hour peak → official stage rules → Streamlit/Plotly`

`river_forecast.py` shares feature and inference logic with training. `river_page.py` contains the user interface; `historical_page.py` retains rainfall analysis. Source manifests, merged data, splits and model artifacts are on disk under `data/river_stage/` and `experiments/river_stage_v1/`.

## History and provenance

The original repository is [nitinc264/Flood-prediction-Model](https://github.com/nitinc264/Flood-prediction-Model). Its former annual-risk dashboard is preserved in `legacy/annual_risk_dashboard.py`, and its original model remains unchanged. Prior nationwide/annual-flow studies and original documentation are retained under `experiments/` and [legacy](legacy/README.md); their claims do not describe the new river-level model.
