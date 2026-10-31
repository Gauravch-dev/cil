# Restored original RiskRadar predictor

The requested evaluation concerns the exact deployed `flood_model.pkl` XGBoost artifact. Inference-only scoring reproduces **96.89% accuracy, 65.64% positive precision, 79.08% positive recall and 71.74% positive F1** on the reconstructed original holdout. No fitting, tuning or artifact replacement occurs. Details and per-record predictions are in `evaluation/existing_model/EVALUATION.md` and `results.json`.

These are reconstructed original-evaluation results, not a verified independent estimate on new observations. The supplied files do not identify an independently labelled dataset known to be unseen by this artifact. The separate new-model training experiment is not an evaluation of the deployed model and must not be substituted for it.

## Current active model and target

The public Flood page again uses the original saved XGBoost predictor with original encoders and 13 ordered features. User authorization specifically requested original RiskRadar-style rainfall/climate/land inputs instead of the Kurundwad river-level form. The deployed model was not retrained or replaced, and its original artifacts remain unchanged. CPU prediction is selected in memory only; the model artifact hash is unchanged.

Existing forensic evidence strongly corroborates the supplied GloFlo dataset (45,932 rows, 93 columns) and a reconstructed binary target of annual peak discharge above the full-source 95th percentile (1145.198 in source units). This is a relative annual high-discharge proxy, not observed flood/no-flood occurrence, inundation or a next-day warning. Public wording uses an annual susceptibility indicator; internal scores are not calibrated flood probabilities.

## Preserved feature contract

Fourteen raw inputs create thirteen fitted features. Precipitation index = 0.4 min(annual rainfall,3000) + 0.3 min(wettest-month rainfall,1000) + 0.3 min(rainfall CV,100). Drainage index = 0.5 min(density,10) + 0.3 min(texture,50) + 0.2 min(relief,5000). Remaining features: mean temperature, temperature seasonality, raw annual precipitation, curve number AMCII, drainage density, basin relief, ruggedness, infiltration and original encoded climate/land-cover/soil classes. feature_columns.pkl controls order.

Original risk bands are retained: Low score <=0.4, Medium >0.4 to <=0.7, High >0.7. These are the supplied application's display bands, not newly validated physical warning thresholds. No thresholds were changed to make demonstrations succeed. The saved predictor is nonmonotonic; a larger annual rainfall value alone need not always increase its score, and values beyond learned tree split ranges can produce unchanged scores. No forced monotonicity or artificial output inflation is added.

## Original evaluation evidence and separation from retired studies

The reconstructed original random holdout has 9,187 rows and 459 positive high-discharge labels. Artifact replay gives accuracy 96.8869%, positive-class precision 65.6420%, recall 79.0850% and F1 71.7391%. This is a corroborated reconstruction on previously inspected data, not a new independent chronological evaluation. Repeated stations span the inferred split, and the label percentile was inferred from the full source. The public upstream F1 0.984 matches the negative class and must not be presented as positive-class F1.

The 99.29% accuracy / 89.66% F1 Kurundwad values belong to the retired city's different warning-stage target. They are not the current predictor's results. Its models and detailed evaluation remain in experiments/kurundwad_flood_risk_v1. No claim is made that the restored original model meets 75% on every positive-class metric.

## Behaviour verification

Default, drier and wetter examples exactly reproduce original scores 0.0779/Low, 0.0038/Low and 0.7840/High. A fixed example with annual rainfall 2200 and wettest-month rain 550, other default inputs, gives 0.6520/Medium. Forty additional seeded input combinations compare both all thirteen feature values and rounded outputs against the independent original api.py. All category codes, rainfall above old maxima, invalid inputs, band boundaries and page state changes are tested. These are software/behaviour checks, not new predictive accuracy evidence.

Detailed provenance: evaluation/upstream_forensics/PROVENANCE_FINDINGS.md and reconstructed_training_evidence.json. Original model SHA256: aad88a12f4645f3703c996641a15ce40603f6d5a608918fad4f37496d2f89395
