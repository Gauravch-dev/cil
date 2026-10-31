# Evaluation of the existing deployed model

The evaluated model is the exact XGBoost artifact used by the Flood interface: `flood_model.pkl`. Its SHA256 is `aad88a12f4645f3703c996641a15ce40603f6d5a608918fad4f37496d2f89395`. No training, calibration, threshold tuning, model selection or artifact replacement was performed.

| Metric | Existing model result |
|---|---:|
| Accuracy | 96.89% |
| Positive-class precision | 65.64% |
| Positive-class recall | 79.08% |
| Positive-class F1 | 71.74% |

The reconstructed original holdout contains 9,187 records and 459 positive high-discharge cases. The binary score threshold remains 0.5. Results reproduce the earlier original-model evidence exactly.

| Actual / Predicted | Other discharge | High discharge |
|---|---:|---:|
| Other discharge | 8,538 | 190 |
| High discharge | 96 | 363 |

The positive target is reconstructed as annual peak discharge above the full-source 95th percentile (1145.198 source units). It is not a supplied verified flood/no-flood label. The target and original split are inferred, and 9,029 holdout records share stations with the inferred training partition. The reconstruction reproduces source-wide numerical median handling. These are reproduction metrics, not an independently established unbiased estimate of future performance.

Evaluating this frozen artifact on genuinely unseen observations requires an independently labelled dataset compatible with its input schema and target. Creating new train/test partitions of its previously used source does not establish that those rows were unseen by this model. No such independent dataset has been identified among the supplied files.

The separate HistGradientBoosting and other retraining experiment evaluates different newly fitted models. Its results must not be attributed to this deployed XGBoost artifact.

Reproduce inference-only scoring: `.venv\Scripts\python.exe scripts/evaluate_existing_model.py`. Machine-readable results and per-record predictions are saved beside this report.
