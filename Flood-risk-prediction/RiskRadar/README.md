# Rainfall and Flood Risk Analysis Using Historical Weather Data

The public application has exactly **two pages: Rainfall and Flood**.

- **Rainfall:** saved historical rainfall near Mumbai, a period selector and one chart at a time with Daily, Monthly, Yearly and Monsoon views; unusually wet days/months and CSV download.
- **Flood:** the original RiskRadar-style annual rainfall, climate and catchment/land/soil inputs. Typical, Drier and Wetter examples; editable area details; Low/Medium/High result; a visual category gauge and CSV download.

The Flood page uses the original saved RiskRadar predictor locally, with the original 13-feature equations, categorical encoder order and risk bands. There are no Kurundwad river-elevation or next-24-hour inputs on this page. The former city classifier is preserved in experiments/kurundwad_flood_risk_v1 but is no longer the public predictor.

The screen is an **annual area-condition susceptibility indicator**, not a time-specific flood alert. It does not show model/evaluation panels or label its internal score as a calibrated flood probability. Technical evidence stays in MODEL_REPORT.md and EVALUATION_RESULTS.md.

The exact deployed XGBoost model is evaluated without fitting or tuning in [the existing-model evaluation](evaluation/existing_model/EVALUATION.md). Reproduced results are 96.89% accuracy, 65.64% positive precision, 79.08% positive recall and 71.74% positive F1 on the inferred original holdout. These are reconstruction metrics rather than independent unseen-data validation. Run `.venv\Scripts\python.exe scripts/evaluate_existing_model.py` to reproduce scoring. Separate retraining studies do not evaluate the deployed artifact.

## Inputs and behaviour

Three main inputs are annual rainfall, wettest-month rainfall and rainfall variation. Land & weather details contains mean temperature, temperature variation, runoff potential, drainage density/texture, elevation difference, ruggedness, infiltration, climate type, land cover and soil class. All six climate, eleven land-cover and twenty-five soil categories are readable and available.

Rainfall fields accept nonnegative values without the old arbitrary slider maxima. Validation rejects missing/nonfinite/negative measurements, unknown categories, runoff outside 0–100 and inconsistent rainfall totals. A wettest month cannot exceed its year, and annual total cannot exceed twelve wettest months. Original feature-engineering caps are preserved where the fitted predictor requires them; raw annual rainfall is not clipped. Invalid or changed inputs never show a stale assessment. Presets and manual inputs persist across page navigation.

## Run

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

Open [RiskRadar](http://127.0.0.1:8501/). The optional start-riskradar.ps1 launcher starts the same app in the background. No separate API service or live data feed is required for either public page. The original api.py is preserved for reference.

## Main files

monitor_page.py is the two-page shell. rainfall_page.py renders historical analysis. riskradar_page.py renders the restored assessment. riskradar_predictor.py validates inputs and calls the original flood_model.pkl, encoders.pkl and feature_columns.pkl. tests/test_riskradar_predictor.py compares the new path against the independent original api.py. scripts/verify_two_page_dashboard.py checks real chart values, all risk categories, edits, invalid inputs and navigation.

Original source: [nitinc264/Flood-prediction-Model](https://github.com/nitinc264/Flood-prediction-Model). Dataset/target reconstruction: evaluation/upstream_forensics/PROVENANCE_FINDINGS.md. Older models, sources, reports and interfaces are preserved in history. The Word report under reports describes an older release and has not been updated for this restoration.
