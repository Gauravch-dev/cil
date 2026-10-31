# Maharashtra River-Level Forecast and Flood-Stage Warning

**Scope:** forecast the maximum observed level over the next 24 hours at 08:00 IST for three known CWC gauges. Apply official station warning/danger rules to numeric forecasts. Historical replay, not live alerts or district inundation.

Historical replay, not live alerts. Source records were inspected in feasibility research. Contiguous hourly flood records are not independent floods. Only three known gauges. Archive corrections/availability may differ from what was issued in real time. Official threshold datum/history assumed consistent for selected series; source QC cannot prove every value. No rainfall predictor: co-located gauge rain is sparse/nonzero-only; it is not filled with zeros. Availability favours high-frequency observation seasons; excluded/missing windows are not normal labels.

## Data and chronology

Manual CWC records only. Telemetry was rejected during feasibility inspection. Missing/ambiguous observations stay missing; no offset/decimal fixes, future filling or synthetic flood labels. All 24 future hourly bins must be observed. Only days with sufficient trailing observations can enter the training/evaluation table. This coverage constraint affects representativeness.
Fixed wide bounds [zero gauge RL-10, danger+30]m; unresolved conflicting duplicates missing; >10m/hour accepted-history jumps missing; unchanged run >72h stale from that point forward. No decimal/datum repairs. Features forward-fill at most2h; >=65 actually observed past72 hourly bins; future label bins never filled. Quality rules do not use model scores or future values.

Train 2014–2020; select 2021; separate error-range calibration 2022; final test 2023–2025. Future label windows crossing partition boundaries are purged. One origin per station/day prevents overlapping 24-hour labels.

## Selection and models

Minimum validation score: 0.5 macro-station MAE overall +0.5 macro-station MAE on target peaks at/above official warning. Ties overall MAE then fewer training-weight changes. Baselines participate in selection. Test/calibration never choose the predictor.
Compare Ridge, Random Forest, Extra Trees, HistGradientBoosting and XGBoost; training-only station-balanced weighting, optionally 4× weight for training target peaks above warning. All regressors predict future-peak change from current level, not the future absolute gauge offset. Numeric imputation, scaling and station categories fit training only. No random early-stopping holdout.
Frozen predictor: **xgboost_1_w1**. Best validation ML: xgboost_1_w1; best validation baseline: positive_6h_trend_damped_12h.

| Predictor | Validation selection score | Test MAE m | Test RMSE m | Relative R² | Warning-period MAE m |
|---|---:|---:|---:|---:|---:|
| xgboost_1_w1 | 0.2417 | 0.2578 | 0.5169 | 0.9718 | 0.2652888476978973 |
| ridge_0_w4 | 0.2718 | 0.3131 | 0.7025 | 0.9480 | 0.36085834866648836 |
| hist_gradient_boosting_0_w4 | 0.2745 | 0.2829 | 0.5608 | 0.9668 | 0.23575891253138365 |
| hist_gradient_boosting_0_w1 | 0.2783 | 0.2707 | 0.5369 | 0.9696 | 0.24232253598410977 |
| ridge_1_w4 | 0.2799 | 0.3087 | 0.6923 | 0.9495 | 0.33063661118890764 |
| xgboost_0_w1 | 0.2869 | 0.2605 | 0.5098 | 0.9726 | 0.21860005830228318 |
| ridge_0_w1 | 0.2957 | 0.2902 | 0.6570 | 0.9545 | 0.2653025344258168 |
| hist_gradient_boosting_1_w1 | 0.3010 | 0.2808 | 0.5400 | 0.9693 | 0.28918631448027554 |
| hist_gradient_boosting_1_w4 | 0.3065 | 0.2988 | 0.5823 | 0.9643 | 0.29368025655197416 |
| ridge_1_w1 | 0.3073 | 0.2890 | 0.6437 | 0.9563 | 0.2410804578303157 |
| ridge_2_w4 | 0.3251 | 0.3065 | 0.6514 | 0.9553 | 0.2750566520435626 |
| random_forest_1_w1 | 0.3348 | 0.2645 | 0.5107 | 0.9725 | 0.21048713932423765 |
| ridge_2_w1 | 0.3436 | 0.2875 | 0.6022 | 0.9618 | 0.2402350501127671 |
| random_forest_1_w4 | 0.3472 | 0.2746 | 0.5350 | 0.9698 | 0.21847582205610366 |
| random_forest_0_w1 | 0.3663 | 0.2665 | 0.5155 | 0.9720 | 0.21822516346930457 |
| extra_trees_1_w1 | 0.3768 | 0.2649 | 0.5084 | 0.9728 | 0.19962291401838 |
| extra_trees_0_w4 | 0.3787 | 0.2764 | 0.5357 | 0.9697 | 0.19795277552214258 |
| xgboost_1_w4 | 0.3811 | 0.2643 | 0.5336 | 0.9700 | 0.24793919721431337 |
| extra_trees_1_w4 | 0.3885 | 0.2822 | 0.5380 | 0.9695 | 0.18274769714716116 |
| random_forest_0_w4 | 0.3912 | 0.2698 | 0.5248 | 0.9710 | 0.2332809337135501 |
| xgboost_0_w4 | 0.3943 | 0.2705 | 0.5466 | 0.9685 | 0.2500252670496678 |
| extra_trees_0_w1 | 0.3985 | 0.2658 | 0.5186 | 0.9716 | 0.2177261089836861 |
| positive_6h_trend_damped_12h | 0.4359 | 0.2053 | 0.5543 | 0.9676 | 0.16650000000001058 |
| persistence | 0.5086 | 0.2062 | 0.5872 | 0.9637 | 0.30549999999999783 |
| previous_24h_max | 0.5976 | 0.3600 | 0.7073 | 0.9473 | 0.417500000000004 |

## Selected predictor: station results

| Gauge | Test forecasts | MAE m | RMSE m | Within-station R² | Warning-period MAE m |
|---|---:|---:|---:|---:|---:|
| Kurundwad | 401 | 0.2618 | 0.4621 | 0.9787221426731917 | 0.16000909530557314 |
| Narsingpur | 343 | 0.3036 | 0.6415 | 0.8779233674624716 | None |
| Takli | 412 | 0.2157 | 0.4464 | 0.9042252538932931 | 0.4232084762863835 |

R² is computed relative to each station warning level when pooled, and within each station separately. Pooling raw 400–540m offsets would inflate R² without proving forecasting skill. MAE/RMSE remain metres.

## Rule-derived warning evaluation

Danger onset restricts origins to current level below danger. This prevents ongoing floods from masquerading as advance prediction.
```json
{
  "warning": {
    "positive": 20,
    "tp": 19,
    "fp": 3,
    "fn": 1,
    "tn": 1133,
    "precision": 0.8636363636363636,
    "recall": 0.95,
    "f1": 0.9047619047619048
  },
  "danger": {
    "positive": 6,
    "tp": 5,
    "fp": 2,
    "fn": 1,
    "tn": 1148,
    "precision": 0.7142857142857143,
    "recall": 0.8333333333333334,
    "f1": 0.7692307692307693
  },
  "danger_onset": {
    "positive": 3,
    "tp": 2,
    "fp": 2,
    "fn": 1,
    "tn": 1148,
    "precision": 0.5,
    "recall": 0.6666666666666666,
    "f1": 0.5714285714285714
  }
}
```

## Forecast range

90% nominal empirical absolute-residual range, from separate 2022 calibration rows per station. No exchangeability or guaranteed future coverage claim; report test coverage.
```json
{
  "nominal": 0.9,
  "coverage": 0.9307958477508651,
  "mean_width_m": 1.455946472270677,
  "per_station": {
    "kurundwad": {
      "radius_m": 0.701526944637294,
      "calibration_n": 150,
      "test_coverage": 0.9301745635910225
    },
    "narsingpur": {
      "radius_m": 0.9853630352019991,
      "calibration_n": 135,
      "test_coverage": 0.9387755102040817
    },
    "takli": {
      "radius_m": 0.5394301822781813,
      "calibration_n": 150,
      "test_coverage": 0.9247572815533981
    }
  }
}
```

No target threshold is moved to improve classification metrics. Forecast ranges are estimated uncertainty, not flood probabilities. Few independent flood episodes limit conclusions; report missed onsets/false alarms alongside regression error. Historical source revisions and broad QC heuristics do not establish operational reliability.

## Files and run commands

Raw source manifest, quality audit, explicit split manifest, model dataset, all validation/test predictions, models, selected inference bundle and metadata are preserved. Previous annual-flow experiments remain unchanged.
```powershell
.\.venv\Scripts\python.exe scripts/fetch_river_stage.py
.\.venv\Scripts\python.exe scripts/train_river_stage.py prepare
.\.venv\Scripts\python.exe scripts/train_river_stage.py select
.\.venv\Scripts\python.exe scripts/train_river_stage.py evaluate
```

Stages refuse to overwrite frozen protocol/selection/test results. Use a new prespecified version for another study.

Sources: [CWC observations](https://nwdp.nwic.gov.in/dataset/river-water-level-manual-hourly-cwc-subernarekha), [CWC official thresholds, page 46](https://cwc.gov.in/sites/default/files/sopapril2025.pdf#page=46).

## Final interpretation and deployment review

The frozen ML winner does not establish superiority: trend/persistence have lower test MAE; trend also has better warning-period MAE and danger-onset recall. Only two observed danger episodes exist in the audited test archive, with one predicted in advance and one missed. Keep this as a historical demonstration. See the [current project report](../../MODEL_REPORT.md) for coverage, episode-level interpretation and completed dashboard verification.
