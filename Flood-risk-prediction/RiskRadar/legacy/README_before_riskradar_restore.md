# Rainfall and Flood Risk Analysis Using Historical Weather Data

The **Flood** page provides a simple **Kurundwad flood-stage risk check**. Enter the observation date, current river elevation, the elevations six and 24 hours earlier, and a recent daily rainfall report if available. Select **Check flood risk** for a Lower/Elevated result covering the following 24 hours. No API or live feed supplies these manual inputs.

**Exact target:** whether measured river stage reaches the official Kurundwad warning level of **535.64 m** at any point in the next 24 hours. This is a local physical flood-stage risk indicator, not a district/street inundation prediction. The current warning/danger status is displayed separately from the future prediction.

The fitted **Logistic Regression** model was selected using validation F1 from three ML classifiers. It uses eight derived features from three river readings, the optional rainfall report, its availability and calendar month. The numerical score is not shown as a calibrated flood probability. Rainfall is a minor predictor; river stage and changes dominate.

## Evaluation

Train 2014–2020: 1,039 daily origins / 18 warning-level cases. Validation 2021–2022: 305 / 10. Test 2023–2025: 422 / 14. Missing future observation windows are excluded as unknown; no artificial no-flood labels are invented. Validation determines both model and decision threshold before test inference. These source years were already inspected in previous work, so this is retrospective follow-up, not independent prospective validation.

| Model / baseline | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression (selected) | 99.29% | 86.67% | 92.86% | 89.66% |
| Decision Tree | 97.87% | 63.16% | 85.71% | 72.73% |
| Random Forest | 98.82% | 73.68% | 100.00% | 84.85% |
| Current-level persistence | 99.76% | 100.00% | 92.86% | 96.30% |
| Positive six-hour trend | 100.00% | 100.00% | 100.00% | 100.00% |
| Always below warning | 96.68% | 0.00% | 0.00% | 0.00% |

The selected ML classifier detects 13 of 14 warning-level cases, misses one and makes two false alerts. It exceeds 75% on the four requested overall test metrics. **This does not establish superior ML performance:** both physical baselines do better on this small test.

**Advance-warning limit:** among 409 origins currently below warning there is only **one** future crossing. ML catches it but makes two false alerts: precision **33.33%**, recall **100%**, F1 **50%**. Most positive days are ongoing high water and are dependent within a few flood periods. The overall results must not be described as 89.66% F1 for predicting new floods in advance. More independent crossing episodes and complete rainfall observations are needed for that claim.

## Application

- **Rainfall** opens by default: a period selector, three rainfall summaries and one interactive chart with Daily, Monthly, Yearly and Monsoon views. Unusual rainfall is highlighted; data can be saved.
- **Flood:** one manual form beside the result and river-reading chart. Model names, performance metrics and evaluation panels are absent from the public interface. Readings and rainfall selections are remembered between pages.

The site has exactly two pages. `monitor_page.py` is the shell; `rainfall_page.py` and `city_flood_page.py` render the pages. Older nationwide analysis, maps and inventory interfaces are preserved outside the public navigation. `scripts/verify_two_page_dashboard.py` verifies both pages and displayed values.

The previous Mumbai rainfall-category module, nationwide event-severity work and river regressors are preserved; they do not drive the main Prediction form.

## Run

From the RiskRadar directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

Open [RiskRadar](http://127.0.0.1:8501/). The existing `start-riskradar.ps1` launcher starts the same local app in the background.

## Sources and files

Only existing files are used. CWC/NWIC manual river observations, official station thresholds already verified against published CWC metadata, and co-located Kurundwad rainfall reports provide this experiment. The rain reports are sparse and nonzero-only: omitted days remain unknown, never zero. Manual users may enter a measured zero, but the source training reports contain no observed zero; behaviour in that case is unvalidated.

`city_flood.py` contains dataset preparation and identical manual/training feature calculation; `city_flood_page.py` is the simple form; `scripts/train_city_flood.py` preserves the three-model comparison. `experiments/kurundwad_flood_risk_v1/` contains the protocol, splits, models, selection, test predictions and verification. Frozen runs refuse overwrite. `MODEL_REPORT.md` records the full limits and model coefficients.

Previous primary documentation is saved under `legacy/*_before_city_flood.md`. The Word report in `reports/` describes the prior rainfall-category release and has not been revised for this new classifier.

Data attribution and source/datum discussion: [saved source findings](research/flood_observations_20261024/FINDINGS.md), [CWC SOP](https://cwc.gov.in/sites/default/files/sopapril2025.pdf#page=46). Original repository provenance: [nitinc264/Flood-prediction-Model](https://github.com/nitinc264/Flood-prediction-Model). The new classifier was trained locally; the inherited model is not used by this form.
