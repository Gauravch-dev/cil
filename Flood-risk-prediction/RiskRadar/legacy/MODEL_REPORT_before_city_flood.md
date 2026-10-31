# Rainfall and Flood Risk Analysis Using Historical Weather Data

## Final scope and target

The project analyses historical rainfall and its relationship to recorded floods and observed severity. The primary UI no longer substitutes numeric river-level forecasting for this problem. The recorded-flood ML output is **Severe Flood versus Flood among known recorded events**. Both labels contain genuine floods; zero does not mean no flood. There are no verified non-flood observations in the supplied files, and no occurrence classifier or live flood probability is claimed.

## Data preparation and matching

INDOFLOODS supplies 4,548 events and matching rainfall rows, metadata for 214 gauges and catchment attributes for 155 public gauges. Unique EventID joins events/rainfall one-to-one; GaugeID joins metadata/catchments many-to-one. Restricted gauges are not negative examples.

All 55 onsets in 2020 are excluded because the antecedent rainfall product ends in 2019; 54 have T10d=0. This date-based source exclusion was fixed before fitting, not chosen using test errors. The descriptive table contains **4,493 rows and 166 analytical columns**: **2,887 Flood / 1,606 Severe Flood** events at **155 gauges in 15 covered states/territories**. Five further cross-boundary events are excluded from supervised partitions.

T1d–T10d are cumulative pre-onset rainfall depths in mm. Differences reconstruct ten daily depths; every plotted day precedes onset. Missing coverage is not dry weather. Train-only imputation and category encoding handle predictor gaps. Raw checksums, joins, labels and prepared-data hashes are preserved.

IFI's 6,876 reported-event rows remain separate; multi-state events count once in a national view. Missing reports never become no-flood days. Whole-period district impacts/DFSI are not predictors. The existing ERA5 archive is a separate point rainfall series near Mumbai, 2000–2025, rather than an invented match to every catchment.

## EDA

Three-day rainfall has mean 95.58 mm, median 65.79 mm and maximum 809.32 mm. 80.21% of supported events start June–September. Pooled rainfall/severity Pearson correlation is -0.0408; after within-gauge centring it is 0.1266. Both relationships are weak; the changed sign exposes gauge composition. These associations among floods are not causal effects or occurrence probabilities.

The report includes seven measured figures: rainfall seasonality, event seasonality, rainfall distributions by severity, conditional severe shares, model comparison, confusion matrix and dated antecedent rainfall. Group boundaries come from training rainfall quartiles; shares divide severe events by observed floods in that group. Empty groups remain unknown.

## Features and chronological lifecycle

The common model uses 19 numeric climate/terrain/location fields, four categorical descriptors and two cyclic month inputs: 25 raw features, 41 encoded. The rainfall model adds ten windows and four indicators (maximum/std daily rain, rainy-day count, recent-three-day share): 39 raw features, 55 encoded. Modern static summaries are retrospective proxies, not reconstructed real-time historical covariates.

Realised peak/discharge, event duration/end date, recession, volume, warning/danger levels, IDs and aggregate impacts are excluded from predictors. The publisher labels are retained rather than moved to improve metrics.

| Period | Samples | Flood | Severe |
|---|---:|---:|---:|
| Training before 2010 | 3250 | 2167 | 1083 |
| Validation 2010–2014 | 613 | 386 | 227 |
| Test 2015–2019 | 625 | 332 | 293 |

Each XGBoost family compares 16 prespecified configurations, 32 fits total, with weight 1 or the training negative/positive ratio. Validation/test retain natural distributions; no test resampling or synthetic non-flood labels. Numeric/category preprocessing and logistic scaling fit training only.

Highest validation average precision, then F1, selects the model/configuration. Thresholds maximise validation F1 with F2 tie-break. Validation selects **catchment XGBoost**, AP 0.6873 and F1 0.6730, versus rainfall AP 0.6588/F1 0.6203. Frozen thresholds are 0.300/0.125/0.250 for catchment/rainfall/logistic.

The selected model uses 450 trees, depth 2, learning rate 0.04, minimum child weight 8, L2 penalty 10 and class weight 1. Rainfall XGBoost uses depth 4 with the same other values. This release packages the previously trained project models; it does not claim new training, a fresh independent test or a test-selected replacement.

## Actual frozen chronological metrics

| Model | Precision | Recall | F1 | F2 | Average precision | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Catchment XGBoost | 0.5429 | 0.6485 | 0.5910 | 0.6242 | 0.5528 | 0.6013 | 0.2682 |
| Rainfall XGBoost | 0.5043 | 0.8020 | 0.6192 | 0.7173 | 0.6118 | 0.6241 | 0.2763 |
| Rainfall logistic | 0.4844 | 0.8464 | 0.6161 | 0.7363 | 0.5225 | 0.5669 | 0.2687 |
| Training prevalence | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4688 | 0.5000 | 0.2674 |
| Always severe | 0.4688 | 1.0000 | 0.6383 | 0.8152 | 0.4688 | 0.5000 | 0.5312 |

Catchment confusion: TN 172, FP 160, FN 103, TP 190. Rainfall confusion: TN 101, FP 231, FN 58, TP 235. False positives are genuine other floods estimated as severe, not false occurrence alerts.

Both XGBoost F1 values are below always-severe F1 0.6383. AP exceeds the severe prevalence 0.4688, but operating-point usefulness and calibration are weak. Gauge-cluster bootstrap intervals for rainfall-minus-static F1/AP cross zero. Selected-model ROC-AUC is 0.7412 on 318 known-gauge test events versus 0.4229 on 307 new-gauge events. No gauges/months are dropped from these diagnostics and the validation selection remains unchanged.

## Application and limitations

Three pages provide matched rainfall/flood analysis, preserved Indian rainfall history and a separate report inventory. Filters, seasonal/rainfall-window charts, actual-event exploration and downloads work. Optional held-out severity estimates are labelled experimental and conditional; no score is shown as a future flood probability.

Missing non-event observations prevent occurrence modelling. Other limits include known-onset alignment, uneven monitoring/reporting, dependent gauge events, modern proxy vintages, unavailable upstream inflow/reservoir/soil-moisture inputs and weak geographic transfer. The valid release is an analytical system, not an operational warning service.

Future occurrence prediction requires consistent daily spatial units, independently verified event/non-event observations, causal weather/hydrological features, new chronologically held-out years/locations and independent calibration. Reusing an inspected test with new algorithms is not new confirmation.

Sources: [INDOFLOODS](https://zenodo.org/records/14584655), [paper](https://doi.org/10.1175/BAMS-D-24-0008.1), [IFI](https://zenodo.org/records/16994648), [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api).

Evidence: [original frozen severity study](experiments/indofloods_severity_v1/COMPARISON_REPORT.md), [actual metrics](experiments/indofloods_severity_v1/final_results.json), [release manifest](experiments/rainfall_flood_release_v1/manifest.json), [runtime verification](experiments/rainfall_flood_release_v1/verification.json), [final academic report](reports/Final_Rainfall_and_Flood_Risk_Analysis_Report.docx).

## Visible Prediction module and regional map

The top navigation now includes **Prediction**. It provides saved-observation replay and a manual 14-day rainfall editor, an explicit **Predict risk** button, Low/Medium/High ML and baseline indicators, an optional actual outcome, sensitivity examples and a CSV download.

The NEW indicator predicts the category of the **next three-day rainfall total** at the existing near-Mumbai ERA5 point. It is a **rainfall-based indicator**, not flood occurrence or a calibrated flood probability. Local training wet-window percentiles define Low below 70.40 mm, Medium from 70.40 to below 103.15 mm, and High at/above 103.15 mm. Rainfall is the only available dynamic weather predictor; no humidity or temperature observations are invented.

Existing data only are used. The rainfall prediction is limited to the saved near-Mumbai weather point. The regional map uses existing INDOFLOODS gauge records: colours show relative **observed severe shares among recorded floods**, using training-gauge terciles, with fewer than ten selected events shown gray. It does not extrapolate the Mumbai model across districts.

Rainfall history now includes a monsoon comparison, heavy-rain days above the baseline wet-day 95th percentile, and monthly departures from a fixed 2000–2015 calendar-month baseline. Months at least two baseline standard deviations wetter are flagged; incomplete periods remain unknown.

The previous event-severity estimates remain available in the historical explorer and retain their conditional meaning. Their Flood/Severe Flood labels are distinct from the new rainfall-category labels.

## New rainfall risk experiment

Only the existing 9,497-day ERA5 series is used. Inputs are current rain, lags through 13 days, 3/5/7/14-day sums, trailing max/std, rainy-day counts, consecutive wet days, rainfall change and calendar seasonality. Targets are the next three-day sum classified with training-only 90th/95th wet-window percentiles. No observed flood label is fabricated.

Train 2000–2015: 5,828 eligible origins. Validation 2016–2019: 1,458. Test 2020–2025: 2,189. Three-day windows crossing boundaries are purged. Preprocessing/scaling and balanced weights fit training only. Validation and test retain natural class distributions.

Compare Logistic Regression, Extra Trees and HistGradientBoosting, two configurations each, plus persistence, seasonal-class frequencies and always-Low baselines. Validation macro F1, then High-class F2, selects the model and class operating cutoffs. Persistence wins validation (0.4794); Extra Trees is the best validation ML candidate (0.4719). The overall winner remains the baseline. The Prediction page explicitly compares the ML prototype with that reference; it does not claim test-selected superiority.

| Candidate | Validation macro F1 | Test macro F1 | High precision | High recall | High AP |
|---|---:|---:|---:|---:|---:|
| logistic_1 | 0.4396 | 0.4054 | 0.1742 | 0.3651 | 0.1874 |
| logistic_01 | 0.4457 | 0.4212 | 0.1885 | 0.3651 | 0.1864 |
| extra_trees_10 | 0.4719 | 0.4743 | 0.3182 | 0.3333 | 0.2171 |
| extra_trees_30 | 0.4350 | 0.4681 | 0.2911 | 0.3651 | 0.2300 |
| hist_gradient_15 | 0.4686 | 0.4233 | 0.2264 | 0.1905 | 0.1781 |
| hist_gradient_31 | 0.4696 | 0.4297 | 0.2105 | 0.2540 | 0.2076 |
| persistence | 0.4794 | 0.4341 | 0.2698 | 0.2698 | 0.0938 |
| seasonal | 0.4018 | 0.3776 | 0.0000 | 0.0000 | 0.1431 |
| always_low | 0.3238 | 0.3239 | 0.0000 | 0.0000 | 0.0288 |

The test contains 2,069 Low, 57 Medium and 63 High origins. The ML prototype detects 21 of 63 High cases and produces 45 High-category false alerts; persistence detects 17 and also produces 46 false High alerts. Always-Low accuracy is 94.52%, illustrating why headline accuracy is misleading. Extra Trees has 31.82% High precision and 33.33% High recall. These limitations preclude a dependable warning claim.

Daily targets overlap over three-day windows, so origins are dependent; no count of independent flood episodes or new prospective confirmation is claimed. Model scores are never shown as flood probabilities. Saved models, protocol, selection, per-model test classes and replay verification are in `experiments/rainfall_risk_v1/`. The source weather file and prior flood models remain unchanged.
