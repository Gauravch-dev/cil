# Maharashtra River-Level Forecast and Flood-Stage Warning

RiskRadar now demonstrates numeric next-24-hour river-peak forecasting at three Maharashtra CWC gauges. A regressor predicts level change; fixed official thresholds derive Below warning / Warning / Danger. This is historical replay, not live warning delivery or district inundation prediction.

## Observations, target and features

- Source: 161,683 manual river-level records from CWC/NWDP, with original files, timestamps, resource IDs and SHA-256 checksums retained.
- Gauges: Kurundwad (Krishna, Kolhapur), Narsingpur and Takli (Bhima, Solapur).
- Target: maximum **observed** level in the next 24 hourly bins after 08:00 IST, excluding the current reading. Every label bin must contain an actual observation. This does not establish the unsampled continuous physical peak.
- Inputs: station, current level relative to warning, warning–danger gap, lag changes through 72 hours, trailing means/maxima/minima/std, positive rises, acceleration and calendar seasonality. No future rainfall or future river readings enter inference.
- Rainfall remains a separate historical-analysis module. Sparse co-located gauge rainfall is not filled with zero or used as a river-model predictor. Reservoir releases and upstream inflows are unavailable as features.
- Rule labels represent **gauge threshold exceedance**, not independently verified inundation, damage or district flood occurrence.

## Data preparation and coverage

Same-time conflicting readings remain unknown. Fixed broad physical bounds, gross jump checks and a causal stale-reading rule flag questionable data; no guessed decimal/datum repairs. Hourly bins are right-closed and right-labelled, so an 08:30 reading cannot enter an 08:00 forecast. Features permit at most two hours of forward filling and require at least 65 actually observed past-72-hour bins. Future labels are never filled.

The merged model dataset contains **4,441** eligible station-days. Coverage is selective and favours high-frequency observation seasons. Missing/excluded windows are not labelled normal. The quality audit contains overlapping exclusion reasons; they must not be added as disjoint counts. This archive cannot establish source availability or reporting delays at historical issue time. Current published official threshold datum/history is assumed consistent with the selected series.

| Period | Purpose | Samples |
|---|---|---:|
| 2014–2020 | Training | 2397 |
| 2021 | Model selection | 453 |
| 2022 | Separate forecast-range calibration | 435 |
| 2023–2025 | Final chronological test | 1156 |

Windows crossing partitions are purged. One origin per day avoids overlapping 24-hour labels, but consecutive days and gauges remain correlated. The final test was evaluated only after model selection and range calibration. Test results have not triggered retuning or replacement of the frozen selection.

## Compared models and selection

Twenty-two candidates compare Ridge, Random Forest, Extra Trees, HistGradientBoosting and XGBoost. Training-only station-balanced weights are compared with optional 4× emphasis on training peaks above official warning. Three baselines also participate: persistence, damped positive six-hour trend and previous-day maximum.

The prespecified 2021 objective equally weights macro-station MAE overall and macro-station MAE during observed warning-level target windows. **XGBoost (`xgboost_1_w1`)** wins with validation score 0.2417, versus 0.4359 for the best baseline. It uses 400 trees, depth 4, learning rate 0.04, minimum child weight 5 and L2 penalty 20. It predicts the numeric change from current level; adding the current level produces the forecast peak. It is not a classifier and does not output a flood probability.

## Actual chronological test results

| Predictor | MAE (m) | RMSE (m) | Warning-period MAE (m) |
|---|---:|---:|---:|
| Frozen XGBoost | 0.2578 | 0.5169 | 0.2653 |
| Damped positive trend | 0.2053 | 0.5543 | 0.1665 |
| Persistence | 0.2062 | 0.5872 | 0.3055 |
| Previous-day maximum | 0.3600 | 0.7073 | 0.4175 |

**ML superiority is not established.** XGBoost has lower RMSE than these baselines but worse overall MAE than persistence and damped trend. The trend baseline also has better warning-period MAE and danger-onset recall. The validation winner remains frozen to avoid choosing a replacement from the final test. This is an experimental ML lifecycle demonstration, not an operational flood warning system.

Pooled R² is 0.9718, computed after subtracting each station's official warning level. Raw gauge offsets around 400–540 m would inflate pooled R². R² is not classification accuracy. Per-station results appear in the detailed experiment report.

| Rule applied to predicted peak | Actual positive station-days | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Warning or above | 20 | 19 | 3 | 1 | 86.36% | 95.00% | 0.9048 |
| Danger or above | 6 | 5 | 2 | 1 | 71.43% | 83.33% | 0.7692 |
| Danger, current reading below danger | 3 | 2 | 2 | 1 | 50.00% | 66.67% | 0.5714 |

The last row removes origins already above danger, but dips within a continuing episode can still produce multiple positive days. A separate audit groups observed danger readings separated by at most 24 hours: **2 observed episodes, 2 with an eligible advance origin, 1 predicted in advance**. Kurundwad July 2024 was detected 23 hours before the first observed crossing; Takli September 2025 was missed. These two episodes cannot establish reliable generalisation, and unobserved gaps can hide or fragment episodes. The trend baseline detects all three positive danger-onset station-days with one false alert; this does not prove reliable episode-level performance either.

The separate 2022 calibration produces 90%-nominal empirical ranges. Actual test coverage is 93.08%, with mean full width 1.456 m. Coverage is not guaranteed in future weather regimes. The range can span warning/danger thresholds; the UI explicitly flags uncertain status.

## Dashboard and saved artifacts

The main app has only River forecast and Rainfall history. Gauge/year/date controls, past-observation example presets, numeric peak and status, a forecast range, interactive level chart, optional recorded outcomes and CSV export work locally. The forecast line represents a window maximum, not an invented hourly trajectory. Archived dates are explicit. The former annual flood-risk UI/model and prior experiments remain preserved outside the primary navigation.

Saved artifacts include the merged dataset, fixed split manifest, acquisition/quality audits, all 22 pipelines, validation/test predictions, selection, model metadata and an inference bundle. Frozen source, split, original-model and bundle hashes verify; all 1,156 held-out predictions replay within numerical precision. Nine truncated-history runtime checks confirm future observations are unnecessary.

Technical verification: **24 unit tests passed; 27 dashboard interaction checks passed; 0 failed**. These checks establish runtime/causality behaviour, not predictive reliability. Earlier annual-model/API evaluations are separate historical work and do not describe this app.

## Sources and detailed evidence

- [CWC manual hourly observations via NWDP](https://nwdp.nwic.gov.in/dataset/river-water-level-manual-hourly-cwc-subernarekha); Krishna resource IDs are retained in `data/river_stage/raw/source_manifest.json`.
- [CWC April 2025 SOP, page 46](https://cwc.gov.in/sites/default/files/sopapril2025.pdf#page=46), official warning/danger levels.
- [Full candidate and station results](experiments/river_stage_v1/MODEL_REPORT.md), [frozen numeric results](experiments/river_stage_v1/final_results.json), [episode/replay audit](experiments/river_stage_v1/verification.json).
- [Previous model report and annual-flow experiments](legacy/MODEL_REPORT_before_river_forecast.md) preserve earlier limitations and source attribution.
