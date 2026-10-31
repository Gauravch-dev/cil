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

## Model design and frozen selection

The target is derived only from 24 fully observed future hourly bins. Current, six-hour and 24-hour level inputs must exist in the conservatively cleaned manual CWC series. There is one issue per day at 08:00 IST; future windows do not overlap. The original broad datum/rate quality rules are reused without tuning against the new model's scores. Raw CWC reconstruction verifies all 1,766 labels and input rows.

Features: current level minus warning, current-minus-six-hour level, current-minus-24-hour level, six-hour change/6 minus 24-hour change/24, latest reported daily rainfall within the past 30 hours, rainfall-availability indicator, cyclic month sine/cosine. Today's 08:30 rainfall report is unavailable to an 08:00 forecast. No future weather, future stage, peak, duration or label is a predictor. Rain report acquisition times are treated as availability; source publication delays and later corrections cannot be reconstructed from this archive.

Training-only median imputation and scaling precede balanced Logistic Regression C=1, balanced Decision Tree max_depth=3/min_samples_leaf=3, or balanced Random Forest 200 trees/max_depth=5/min_samples_leaf=3. No SMOTE or test resampling is used. Thresholds 0.05–0.95 in steps of 0.01 are compared on validation F1, then recall and precision. The selected Logistic Regression threshold is **0.95**, frozen before test inference. Class weights and the very high fitted cutoff do not make raw scores calibrated probabilities.

Validation selected Logistic Regression: accuracy 99.02%, precision 76.92%, recall 100%, F1 86.96%. Decision Tree F1 63.64%; Random Forest F1 76.92%. Final test ROC-AUC 0.9909 and average precision 0.9145. The confusion matrix is [[406,2],[1,13]] in below-warning / future-warning order.

The positive-trend baseline is current level + 2 × positive six-hour rise, compared with the same official warning. Persistence uses current stage directly; always-below-warning illustrates the misleading 96.68% majority accuracy. Baselines were declared in the protocol before final inference and are reported without claiming an ML advantage.

## Practical limits and further data

The source threshold/datum history is assumed consistent with previously verified metadata. Warning exceedance is not observed damage or inundation, and river flooding differs from urban surface-water flooding. Training has 18 positive daily origins; test has 14, mostly consecutive within very few events. This small sample does not justify narrow performance guarantees. All eligible test cases remain included; no months or districts were chosen from test success. Kurundwad was selected because it has existing co-located rainfall and stage data.

To validate a genuine advance-warning model, collect continuous stage and complete, dated local/upstream rainfall covering many independent warning-crossing floods and normal periods, with consistent gauge datum and warning history. Reserve later unseen flood seasons before further tuning. No further tuning is performed on this frozen test.
