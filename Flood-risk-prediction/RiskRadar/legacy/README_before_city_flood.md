# Rainfall & Flood Risk Analysis Using Historical Weather Data

RiskRadar analyses historical rainfall, the rain preceding recorded floods, and observed flood severity. The main application now supports the stated analysis problem. Earlier river-level forecasting is preserved in history.

## Dashboard

- **Prediction:** visible three-day rainfall-risk ML/baseline estimates and a manual what-if editor.

- **Rainfall & floods:** state/year filters, seasonal event counts, rainfall windows, observed severity, conditional severe shares and actual-event exploration. Optional held-out ML severity comparisons are labelled experimental.
- **Rainfall history:** existing yearly totals, anomalies, monthly heatmaps, daily rainfall and downloads for Indian catalog locations.
- **Flood records:** the separate India Flood Inventory, with state/year filters and downloads.

## Data and scope

The matched event table contains **4,493 events at 155 gauges across 15 covered states/territories** through 2019. The separate inventory contains **6,876 reports**. A cached ERA5 series near Mumbai provides 9,497 daily values during 2000–2025; it is not silently substituted for catchment event rainfall.

**Flood and Severe Flood are both actual floods. Verified non-flood days are absent.** Missing reports are never recoded as no flood. The ML comparison estimates severity conditional on a known event; it is not daily flood/no-flood prediction or live warnings.

## Models and results

The frozen project study compares 32 XGBoost fits across catchment and rainfall feature sets, plus rainfall logistic regression and constant baselines. Train before 2010, validation 2010–2014, test 2015–2019. Validation selects catchment XGBoost. This release packages the existing fitted project pipelines without retraining or changing selection after inspecting test results.

On 625 test events, selected-model precision is **54.29%**, recall **64.85%**, F1 **0.5910** and average precision **0.5528**. The rainfall version raises recall to **80.20%**, but precision is **50.43%** and false severe classifications increase. Neither beats always-severe F1 **0.6383**. These are analytical comparisons with limited predictive reliability.

## Run

From this directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

Open [RiskRadar](http://127.0.0.1:8501/). `.\start-riskradar.ps1` starts the same app in a hidden background process and records its PID/logs under `.run/`. The old FastAPI service is not required.

## Evidence and report

[Final academic report](reports/Final_Rainfall_and_Flood_Risk_Analysis_Report.docx), [MODEL_REPORT](MODEL_REPORT.md), [REVIEW](REVIEW.md), [validation](EVALUATION_RESULTS.md), [setup](LOCAL_SETUP.md).

`rainfall_flood.py` contains shared data semantics/inference; `rainfall_flood_page.py` renders the primary UI. Prepared data live in `data/rainfall_flood/`; release models/manifests in `experiments/rainfall_flood_release_v1/`.

Sources: [INDOFLOODS](https://zenodo.org/records/14584655), [IFI-Impacts](https://zenodo.org/records/16994648), [Open-Meteo/ERA5](https://open-meteo.com/en/docs/historical-weather-api). Original codebase provenance: [nitinc264/Flood-prediction-Model](https://github.com/nitinc264/Flood-prediction-Model). Original models, prior experiments and earlier reports are preserved; see [legacy](legacy/README.md).

## Visible Prediction module and regional map

The top navigation now includes **Prediction**. It provides saved-observation replay and a manual 14-day rainfall editor, an explicit **Predict risk** button, Low/Medium/High ML and baseline indicators, an optional actual outcome, sensitivity examples and a CSV download.

The NEW indicator predicts the category of the **next three-day rainfall total** at the existing near-Mumbai ERA5 point. It is a **rainfall-based indicator**, not flood occurrence or a calibrated flood probability. Local training wet-window percentiles define Low below 70.40 mm, Medium from 70.40 to below 103.15 mm, and High at/above 103.15 mm. Rainfall is the only available dynamic weather predictor; no humidity or temperature observations are invented.

Existing data only are used. The rainfall prediction is limited to the saved near-Mumbai weather point. The regional map uses existing INDOFLOODS gauge records: colours show relative **observed severe shares among recorded floods**, using training-gauge terciles, with fewer than ten selected events shown gray. It does not extrapolate the Mumbai model across districts.

Rainfall history now includes a monsoon comparison, heavy-rain days above the baseline wet-day 95th percentile, and monthly departures from a fixed 2000–2015 calendar-month baseline. Months at least two baseline standard deviations wetter are flagged; incomplete periods remain unknown.

The previous event-severity estimates remain available in the historical explorer and retain their conditional meaning. Their Flood/Severe Flood labels are distinct from the new rainfall-category labels.
