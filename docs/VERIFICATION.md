# Verification record

Verified locally on 3 November 2026. These checks concern the web adapter and
bundled older model; they do not certify the application for public deployment.

## Automated checks

- **55 tests passed** across `test_web_api.py`, `test_readings.py`,
  `test_local_forecast.py`, `test_demo_evidence.py` and `test_hybrid_forecast.py`
  with the forecasting runtime and evaluated hybrid artifacts installed.
- The TypeScript and Vite production build passed.
- `git diff --check` passed.
- One upstream Starlette/httpx deprecation warning was reported; no test failed.

Reproduce from the repository root:

```powershell
.\.venv-web\Scripts\python.exe -m pip install -r requirements-forecast.txt -r requirements-web-dev.txt
.\.venv-web\Scripts\python.exe -B -m pytest tests/test_web_api.py tests/test_readings.py tests/test_local_forecast.py tests/test_demo_evidence.py tests/test_hybrid_forecast.py -q --basetemp ../work/pytest-local
Set-Location frontend
npm ci
npm run build
```

Without TensorFlow the real-model test module is skipped; report that limitation
rather than treating the reduced suite as complete inference verification.

## Evidence covered

| Area | Checks |
| --- | --- |
| Recorded results | Exact saved prediction values, original dates, matching explanations, unavailable explanations kept unavailable |
| Original rules/search | Essential ontology protection, occupancy/comfort behavior, valid schedules, infeasibility, independent tariff/cost recomputation |
| Input validation | Schema, size, encoding, delimiters, duplicate/future timestamps, timezone semantics, missing values, causal gap filling, eight-day coverage and freshness |
| Frozen inputs | History/context shapes and order, no future meter labels, scaler transforms without fitting |
| Real artifact fidelity | Five saved forecasts reproduce with maximum observed absolute difference about `5.3e-8 kW`; original input arrays and reconstructed preprocessing also agree |
| New inference | Real bundled model executes on an explicitly synthetic upload fixture in a temporary database; archived replay and fitting are forbidden in the check |
| Sample routing | Explicit sample flag required, synthetic samples blocked from current household inference, original dates and identity preserved in saved JSON |
| Persistence/export | Temporary SQLite databases, exact restore, JSON/CSV exports, formula-prefix neutralization, strict request/source selection and cross-origin write rejection |

Synthetic fixtures validate contracts; their predictions are not household
measurements or evidence of accuracy. The test suite does not reset user history.

## Browser observations

The local built frontend was exercised with recorded cases, household/tariff and
appliance settings, histories, rule evidence, upload checks and sample downloads.
Desktop/mobile layouts, keyboard interaction and error/loading states were
reviewed during implementation.

Before the new classical stage was trained, the sample flow loaded all 11,520 rows, selected UTC and executed the
bundled model. It displayed approximately **0.440 kW** for **9 January 2026,
00:45–01:00 UTC**, with an explicit sample label, original dates, rule checks,
appliance plan and saved history. This is a reproducibility example, not an
expected result for another file or a current household estimate.

## Limits

The newer multivariate/hybrid artifacts are absent, so that optional integration
has not been verified with real matching artifacts. No full Task 10 real-data
assessment, calibrated individual confidence interval, independent current meter
validation, measured utility savings, authentication or deployment was completed.
Original offline evaluation files were not rewritten as new results.

## Guided demo and household redesign

The redesigned UI separates ordinary household tasks from an eight-stage technical
walkthrough and retains shared history. New real-model checks verify 23-group
permutation SHAP reconstruction, actual activation dimensions, no training, exact
forecast identity, cache consistency, unavailable-runtime handling and explicit
SHAP failure without fabricated substitutes. The evidence endpoint preserves the
saved analysis and its original explanation. Browser checks cover recorded and
sample runs, ontology/rule inspection, SHAP, history, mobile layout and separated
household/demo preferences. No real household meter connection was introduced.

## Trained classical stage — 3 November 2026

The compatible older-LSTM hybrid is now trained and connected for new analyses.
It uses 16 frozen LSTM patterns and 17 context features; validation selected Random
Forest over Gradient Boosting. TreeSHAP explains the executed 33-feature regressor.
Old saved standalone forecasts remain unchanged. The newer multivariate experiment
remains distinct and absent. Earlier statements that the classical stage was
unavailable describe the pre-training state. See [the training report](HYBRID_TRAINING.md) for actual
metrics, limits, artifact restoration and the explicit offline preparation command.

Six additional hybrid checks verify validation-based selection, artifact integrity,
no runtime fitting, real 33-feature TreeSHAP reconstruction, sample-date boundaries,
legacy model preservation and rejection of mismatched heads. Browser verification
confirmed a new recorded hybrid forecast of about 1.220 kW for 24 April 2010,
18:00–18:15, an active ML tab and TreeSHAP reconstruction residual 1.33e-15 kW.
