# Expanded features and station-aware validation

Previously inspected validation/test: exploratory follow-up, not fresh independent confirmation. Contemporary static catchment/climate proxies are not historically issued weather forecasts. The label is high annual peak discharge, not verified flood occurrence. Unseen stations can remain spatially related to training catchments.

## Task and partitions

High annual peak discharge above the full-training 95th percentile (1129.84, source units). This target is unchanged for final validation/test comparison.
Train 1900–1999: 37,184 rows / 1,859 positives. Validation 2000–2005: 3,387 / 162. Test 2006–2015: 5,339 / 320. All source rows and final labels match previous studies.

## Station-disjoint forward validation

SHA256(station code) first 4 bytes modulo 3, with group i held out in forward-time fold i. All records from those stations are excluded from that fold training.
Each fold derives the 95th percentile from its own training rows only. Final models use the unchanged full-training cutoff 1129.84.

| Fold | Train through | Validate years | Train rows | Validation rows | Validation positives | Fold cutoff |
|---|---:|---|---:|---:|---:|---:|
| 0 | 1959 | 1960–1979 | 6134 | 4253 | 162 | 1147.94 |
| 1 | 1979 | 1980–1989 | 14798 | 2427 | 105 | 1118.52 |
| 2 | 1989 | 1990–1999 | 19336 | 2627 | 174 | 1096.78 |

No station appears on both sides of a fold; every training year precedes its validation years. Station groups and year windows are fixed independently of outcomes. These folds live entirely inside the original training partition. They do not constitute a spatially separated basin benchmark.

## Features and model selection

Within each training fold: exclude additions >40% missing, constant, or Pearson correlation >0.98 with retained numeric features. Retain legacy13. For selected view retain top 12 additions by training-only, station-balanced Random Forest grouped importance. Refit selector only on full training for final models.
Three views: legacy13, expanded after training-only missingness/correlation filtering, and selected (legacy13 plus at most 12 training-ranked additions). Extra candidates include drainage area, basin length, sinuosity, stream order and seasonal climate attributes. Raw discharge, unit discharge, station identifiers, location identifiers and dam counts are excluded.
CatBoost uses native categorical inputs, ordered boosting and has_time=True on chronologically sorted training rows. Random Forest and HistGradientBoosting use training-fitted medians/scaling and one-hot categories. No random internal early stopping, SMOTE or held-out fitting is used.
Eighteen candidates × three folds = 54 CV model fits, plus four training-only feature selectors and nine final training-only pipelines. The two configurations per learner vary complexity and weighting; this is a bounded search, not exhaustive optimisation.
Maximum mean fold average precision on unseen stations; ties minimum fold AP then pooled OOF F2. Best configuration per learner/view and overall winner frozen before outer validation/test.
From pooled station-disjoint forward OOF training predictions: maximise F2 subject to precision >=0.50 and recall >=0.75. If infeasible, maximise F2 and disclose failure. Scores rounded to 12 decimals for numerical stability. No outer validation/test threshold tuning.
The unseen-station outer validation subgroup has only 222 rows / 18 positives. It is reported as a separate check; neither it nor final test is used to choose models or thresholds.

**Frozen winner: hist_gradient_boosting__expanded**, mean forward/station-disjoint AP 0.6080, OOF threshold 0.051432, OOF operating constraints met: False.
Final selected inputs (37): precipitation_risk_index, drainage_risk_index, annual_mean_temperature, temperature_seasonality, annual_precipitation, curve_number_amcii, drainage_density, basin_relief, ruggedness_number, infiltration_number, climate_type, landcover_type, soil_type, drainage_area, basin_length, sinuosity_index, stream_order, basin_perimeter, bifurcation_ratio, form_factor, relief_ratio, elongation_ratio, circularity_ratio, lemniscates_value, drainage_texture, compactness_coefficient, wandering_ratio, fitness_ratio, channel_frequency, drainage_intensity, precipitation_wettest_month, precipitation_driest_month, precipitation_seasonality, mean_diurnal_range, isothermality, temperature_annual_range, lithology_type.

## Chronological test: all stations

| Learner / feature view | CV mean AP | Precision | Recall | F1 | F2 | PR AP | ROC-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hist_gradient_boosting__expanded | 0.6080 | 0.4245 | 0.9313 | 0.5832 | 0.7518 | 0.8659 | 0.9770 | 404 | 22 |
| hist_gradient_boosting__selected | 0.6036 | 0.4797 | 0.8875 | 0.6228 | 0.7585 | 0.8326 | 0.9699 | 308 | 36 |
| catboost__expanded | 0.5994 | 0.3178 | 0.9406 | 0.4751 | 0.6758 | 0.7675 | 0.9668 | 646 | 19 |
| catboost__selected | 0.5961 | 0.3168 | 0.9406 | 0.4740 | 0.6749 | 0.7619 | 0.9702 | 649 | 19 |
| random_forest__expanded | 0.5890 | 0.6078 | 0.8812 | 0.7194 | 0.8085 | 0.8421 | 0.9668 | 182 | 38 |
| random_forest__selected | 0.5851 | 0.6074 | 0.8750 | 0.7170 | 0.8041 | 0.8370 | 0.9669 | 181 | 40 |
| hist_gradient_boosting__legacy13 | 0.4010 | 0.3261 | 0.9406 | 0.4843 | 0.6832 | 0.8574 | 0.9773 | 622 | 19 |
| random_forest__legacy13 | 0.3994 | 0.5766 | 0.8938 | 0.7010 | 0.8052 | 0.8418 | 0.9662 | 210 | 34 |
| catboost__legacy13 | 0.3613 | 0.3807 | 0.7625 | 0.5078 | 0.6351 | 0.6393 | 0.9477 | 397 | 76 |

## Chronological test: unseen stations

Same 494 test rows / 35 positives as the earlier failure. No test stations are removed.

| Learner / feature view | Precision | Recall | F1 | F2 | AP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hist_gradient_boosting__expanded | 0.4203 | 0.8286 | 0.5577 | 0.6938 | 0.6106 | 40 | 6 |
| hist_gradient_boosting__selected | 0.5000 | 0.6571 | 0.5679 | 0.6183 | 0.6179 | 23 | 12 |
| catboost__expanded | 0.3053 | 0.8286 | 0.4462 | 0.6170 | 0.6866 | 66 | 6 |
| catboost__selected | 0.3412 | 0.8286 | 0.4833 | 0.6444 | 0.7317 | 56 | 6 |
| random_forest__expanded | 0.5778 | 0.7429 | 0.6500 | 0.7027 | 0.6929 | 19 | 9 |
| random_forest__selected | 0.6000 | 0.6857 | 0.6400 | 0.6667 | 0.6627 | 16 | 11 |
| hist_gradient_boosting__legacy13 | 0.2871 | 0.8286 | 0.4265 | 0.6017 | 0.6732 | 72 | 6 |
| random_forest__legacy13 | 0.4833 | 0.8286 | 0.6105 | 0.7250 | 0.7177 | 31 | 6 |
| catboost__legacy13 | 0.4247 | 0.8857 | 0.5741 | 0.7277 | 0.6484 | 42 | 4 |

## Selected model versus the previous frozen winner

| Population / model | Precision | Recall | F1 | F2 | AP | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all / previous | 0.8894 | 0.6531 | 0.7532 | 0.6898 | 0.8431 | 209 | 26 | 111 |
| all / new | 0.4245 | 0.9313 | 0.5832 | 0.7518 | 0.8659 | 298 | 404 | 22 |
| unseen_stations / previous | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6883 | 0 | 0 | 35 |
| unseen_stations / new | 0.4203 | 0.8286 | 0.5577 | 0.6938 | 0.6106 | 29 | 40 | 6 |

Both the feature/model selection protocol and operating objective changed. Do not attribute the entire before/after difference to features or to CatBoost alone. Legacy13 controls in this experiment help distinguish feature effects under the same protocol.
**Assessment:** selected-model recall rises to 0.9313, but precision falls to 0.4245; false positives rise from 26 to 404 and all-station F1 falls from 0.7532 to 0.5832. This is not an overall fix. The precision operating target was not met in cross-validation; the candidate is retained as an experiment, not deployed.
Unseen-station average precision also falls from 0.6883 to 0.6106. Recovering more positives at a lower threshold does not establish better ranking or generalisation. Within the new protocol, expanded HistGradientBoosting reduces unseen-station false positives from 72 to 40 versus its legacy13 control at the same 29 detected positives, but its ranking AP is lower. The feature procedure and threshold selection have mixed effects.
Paired station-bootstrap 95% F1-difference intervals (new minus previous): {'all': [-0.23021770932482405, -0.10794255808813162], 'unseen_stations': [0.38505560704355885, 0.6987366407177728]}. These are exploratory intervals on previously examined data.

## Untuned outer validation check

| Selected model population | Rows | Positives | Precision | Recall | F1 | F2 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 3387 | 162 | 0.3734 | 0.9198 | 0.5312 | 0.7116 | 0.8543 |
| seen_stations | 3165 | 144 | 0.3668 | 0.9653 | 0.5315 | 0.7277 | 0.8966 |
| unseen_stations | 222 | 18 | 0.5000 | 0.5556 | 0.5263 | 0.5435 | 0.5767 |

## Limits and artifacts

Training-derived operating targets are not promises for future recall or precision. Minority and unseen-station samples are limited. Weighted scores are not established as calibrated flood probabilities. Static attributes cannot establish event timing. More useful catchment descriptors can improve high-discharge classification without turning these data into verified flood/non-flood observations.
All nine inference pipelines, native CatBoost models, metadata, fold manifests, selected inputs, fold feature audits, out-of-fold predictions, outer validation/test predictions and metrics are saved. Previous studies and the website model are unchanged.

## Reproduction

Install requirements-training.txt. Frozen stages refuse overwrite. Use a new prespecified version for another experiment.

```powershell
.\.venv\Scripts\python.exe scripts/compare_gloflo_station_models.py prepare
.\.venv\Scripts\python.exe scripts/compare_gloflo_station_models.py select
.\.venv\Scripts\python.exe scripts/compare_gloflo_station_models.py evaluate
.\.venv\Scripts\python.exe scripts/compare_gloflo_station_models.py verify
```

References: [CatBoost parameters](https://catboost.ai/docs/en/references/training-parameters/common), [group/time validation](https://scikit-learn.org/stable/modules/cross_validation.html).
