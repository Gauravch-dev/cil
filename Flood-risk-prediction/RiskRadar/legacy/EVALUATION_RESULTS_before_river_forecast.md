# Current validation and evaluation results

This evaluates the current application without modifying the UI, model or API. Software behavior and predictive accuracy are reported separately.

## Existing regression suite

- Dashboard integration checks: 39 passed, 0 failed.
- Historical page integration checks: 15 passed, 0 failed.
- Data-handling unit tests: 10 passed, 0 failed, 0 skipped.

## Additional artifact and API validation

- Artifact/consistency checks: 6 passed, 0 failed.
- API input checks: 3 passed, 5 failed.

**Total: 73 passed, 5 failed. Overall strict software validation: FAIL.**

| API case | Expected HTTP status | Actual HTTP status | Result |
|---|---:|---:|---|
| Valid example | 200 | 200 | Pass |
| Missing required rainfall | 422 | 422 | Pass |
| Non-numeric rainfall | 422 | 422 | Pass |
| Negative annual rainfall | 422 | 200 | FAIL |
| Wettest month exceeds annual rainfall | 422 | 200 | FAIL |
| Negative drainage density | 422 | 200 | FAIL |
| Unknown climate code | 422 | 200 | FAIL |
| Negative soil category | 422 | 200 | FAIL |

Failed invalid-input checks mean the API returned a risk score for unsupported or physically inconsistent input. The dashboard rejects inconsistent annual/wettest-month rainfall and constrains widget ranges, but direct API callers can bypass these controls. The API schema needs non-negative numeric bounds, category-code bounds and cross-field rainfall validation. These failures were recorded rather than changed during evaluation.

## Data audit

- 45,932 records, 93 columns, 9,710 stations, 57 countries.
- 0 exact duplicate source rows; 22 unusable year entries; 10 entirely missing columns.
- The saved rainfall archive has 9,497 days and 0 missing rainfall values (100.0% coverage). Coverage is completeness, not measurement accuracy.
- Annual precipitation is a fixed station climate attribute in GloFlo, not an annual weather time series.

## Model accuracy evaluation

**Independent model accuracy cannot be established from the available data.**

| Metric | Independently measured result |
|---|---|
| Accuracy | Unavailable |
| Precision | Unavailable |
| Recall | Unavailable |
| F1 | Unavailable |
| ROC-AUC / PR-AUC | Unavailable |
| Calibration / operational thresholds | Unvalidated |

The original source lacks binary flood/no-flood labels, the target-generation pipeline and test-split membership. Saved risk scores are predictions, not ground truth. No synthetic labels or in-sample performance are substituted for a test. The existing binary classifier does not predict rainfall amounts, so rainfall MAE/RMSE/R² are not applicable to this model.

The upstream README claims accuracy 96.9%, F1 0.984 and recall 0.791. They have not been reproduced and must not be presented as this validation’s results. F1 0.984 with recall 0.791 is incompatible for the same positive class/test population; different averaging/evaluation conventions would need to be documented.

## Runtime examples, not accuracy measurements

| Scenario | Score | Category |
|---|---:|---|
| Typical conditions | 7.79% | LOW |
| Drier landscape | 0.38% | LOW |
| Wetter landscape | 78.40% | HIGH |

Repeated identical requests returned identical scores. These examples demonstrate execution only; they do not validate real-world flood prediction.

## Evaluation conclusion

The rainfall analysis and risk-scenario application pass the existing regression checks. Additional validation exposes missing API input safeguards. The supplied flood classifier remains an unvalidated demonstration until ground-truth labels and an independent evaluation split are available. No deployment-readiness or flood-warning accuracy claim is supported.
