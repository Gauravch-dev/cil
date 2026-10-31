# Causal annual-history expert comparison

Previously examined data: exploratory follow-up, not new independent confirmation. Historical annual flows are retrospective covariates, not daily weather. Final history is stale by design (pre-2000 snapshot). High annual discharge is not verified flood occurrence or an operational warning.

## Protocol

Same station, strictly earlier years from an explicit fit-reference only. Same-year values are excluded; no current discharge/unit discharge as predictor. Final outer validation/test both reference training through 1999 only. No updating from validation/test outcomes, even earlier ones.
History count >=1: history expert. Count zero: static expert. All cases remain evaluated.
Use original three forward windows and exclude hashed station group i from fold fitting. Validate ALL stations in the later window: report history and no-history routes separately. Held-out stations have no history references. Fold cutoff fits fold training only.
Per route: prefer achieving precision>=0.60, recall>=0.75 and FPR<=0.03 in pooled OOF. Otherwise retain precision/FPR guards where feasible, maximise F1 and disclose recall failure. Model ties use AP. No model/threshold selection from outer validation/test.
Thresholds from route-specific pooled forward OOF scores, rounded to 12 decimals. Require >=20 predicted positives for a stable precision guard. No fallback to unconstrained F2 if the guard fails: retain FPR cap, maximise F1, report failure.
Same source rows, chronology and final high-discharge cutoff (1129.84). Three temporal folds use training-only labels/cutoffs, references, medians, vocabulary and fitting. Ten candidates × three folds = 30 CV fits; five final pipelines. History predictors are last/mean/median/max/std discharge, last-three-observation mean, prior exceedance rate, count and age. All are strictly earlier observations. They do not describe rainfall or flood occurrence.

## Frozen experts

| Route | Learner | OOF precision | Recall | F1 | FPR | Joint goal met | Threshold |
|---|---|---:|---:|---:|---:|---|---:|
| history | history__random_forest | 0.8010 | 0.7553 | 0.7774 | 0.0113 | True | 0.378649 |
| no_history | no_history__hist_gradient_boosting | 0.6020 | 0.4324 | 0.5033 | 0.0130 | False | 0.396904 |

## Final test comparison

| Model / population | Rows | Positives | Precision | Recall | F1 | F2 | AP | ROC-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| previous precision model / all | 5339 | 320 | 0.8894 | 0.6531 | 0.7532 | 0.6898 | 0.8431 | 0.9707 | 26 | 111 |
| previous recall model / all | 5339 | 320 | 0.4245 | 0.9313 | 0.5832 | 0.7518 | 0.8659 | 0.9770 | 404 | 22 |
| new experts / all | 5339 | 320 | 0.8172 | 0.7406 | 0.7770 | 0.7548 | 0.8792 | 0.9821 | 53 | 83 |
| new experts / with history | 4845 | 285 | 0.8168 | 0.7825 | 0.7993 | 0.7891 | 0.9057 | 0.9898 | 50 | 62 |
| new experts / no history (unseen stations) | 494 | 35 | 0.8235 | 0.4000 | 0.5385 | 0.4459 | 0.6548 | 0.8850 | 3 | 21 |
| historical-mean baseline / with history | 4845 | 285 | 0.8084 | 0.8140 | 0.8112 | 0.8129 | 0.8961 | 0.9899 | 55 | 53 |

Combined expert scores come from different models and need not share a calibrated scale. Combined AP/ROC are descriptive only; route-specific ranking is the clearer comparison. Combined precision/recall/F1 use actual route-specific binary decisions.

## Untuned outer validation

| Population | Precision | Recall | F1 | FPR | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| all | 0.7665 | 0.7901 | 0.7781 | 0.0121 | 39 | 34 |
| history | 0.7562 | 0.8403 | 0.7961 | 0.0129 | 39 | 23 |
| no_history | 1.0000 | 0.3889 | 0.5600 | 0.0000 | 0 | 11 |

## Interpretation and limits

The new combined operating point improves the precision/recall balance over the previous recall-oriented model: false positives fall from 404 to 53, precision rises to 0.8172, and recall falls to 0.7406. This is a partial improvement, not a universal fix. The no-history route still misses 21 of 35 positive test cases; its recall is 0.4000. The historical-mean baseline has history-route F1 0.8112 versus learned RF 0.7993, so extra ML benefit over a simple history baseline is not established.
Past discharge is additional information unavailable to earlier static models. Do not attribute changes solely to the learning algorithm or compare them as identical input settings. No current/future-year discharge is allowed. All 35 test positives at stations absent from training stay in the no-history route; their test outcomes never become later inputs. Final history is a pre-2000 snapshot; age and missing-history routing are explicit. The CV operating targets are not guarantees on outer periods. All failures are reported; no stations or cases are omitted for low confidence.
Learned history models must be compared with the historical-mean baseline. A strong history baseline indicates station-level persistence, not weather-forecast skill. These annual data cannot establish a daily flood warning model. This remains follow-up evidence on an inspected test set. The website is unchanged.

Supplementary paired station-bootstrap 95% F1-difference intervals: new minus previous precision model [-0.0091, 0.0583] (includes zero); new minus previous recall model [0.1453, 0.2418]. These describe exploratory repeated-test comparisons, not fresh independent confirmation.

## Artifacts and reproduction

Protocol, candidate/OOF results, reference provenance, five pipelines, training features, final predictions, metadata and verification are saved here. Dependencies: requirements-training.txt.

The self-contained `selected_experts.joblib` bundle contains the frozen training reference, two selected pipelines, feature lists and route thresholds. `discharge_history.predict_annual_discharge(query, bundle_path)` applies the correct route and threshold, without requiring current discharge or target values. Query years must follow 1999; the pre-2000 reference is preserved. Bundle predictions were replayed on every test row without its outcome columns.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_gloflo_history.py prepare
.\.venv\Scripts\python.exe scripts/evaluate_gloflo_history.py select
.\.venv\Scripts\python.exe scripts/evaluate_gloflo_history.py evaluate
.\.venv\Scripts\python.exe scripts/evaluate_gloflo_history.py verify
```

Frozen stages refuse overwrite. Use a new prespecified experiment version for another study.
