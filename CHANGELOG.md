# Changelog

## 2026-11-03 — Trained classical prediction head

- Fitted Random Forest and Gradient Boosting on frozen older-LSTM features using
  the original training/validation partitions and Task 9 settings.
- Selected Random Forest by validation RMSE; evaluated all 18,889 original test cases.
- Connected the hybrid to new recorded, uploaded and sample analyses; preserved
  earlier standalone results and their matching explanation path.
- Added real TreeSHAP, feature-handoff evidence, load-only fingerprint checks,
  six hybrid integration tests and reproducible training documentation.
- Reported the tradeoff: slightly worse overall error and better high-demand error
  than the standalone LSTM. No claim of recreating the missing newer experiment.
- Included the evaluated serving artifact and a presenter walkthrough; pinned its
  scikit-learn version. Training caches and personal upload/history data stay ignored.

## 2026-11-03 — Household workspace and guided demonstration

- Replaced the crowded all-in-one dashboard with a three-step household workspace,
  eight-stage guided demo and shared history.
- Added isolated demo scenarios, inspectable ontology candidates, ordered fired rules,
  A* search evidence and explicit DL/classical-ML status.
- Added actual grouped permutation SHAP for the frozen bundled LSTM, prediction
  matching, reconstruction checks, real activations and downloadable evidence.
- Preserved original forecasting/rule/scheduling semantics and local history.

## 2026-11-02 — Local web application

### Added

- Responsive blue/white/orange React interface: overview, appliances, insights and history.
- FastAPI adapter over the unchanged original ontology, operational rules and A* scheduler.
- Recorded forecasts, exact saved explanation matching, rule evidence, schedules and job-cost comparisons.
- Local SQLite history and JSON/schedule/recommendation exports.
- CSV upload checks, timezone selection, template downloads and direct sample loading.
- Frozen inference through the bundled completed older Task 7 model, original preprocessing and scalers, with startup fidelity checks.
- Explicit sample-date forecasts using existing household settings, with original dates and sample identity preserved throughout results/history/exports.
- Setup, usage, API, architecture/integration and verification documentation.
- 45 adapter, validation and real-artifact tests; local production build verification.

### Preserved and clarified

- Original task sources, ontology, notebooks, trained artifacts and recorded metrics remain unchanged.
- Actual current use requires actual recent readings; synthetic examples are never relabelled as current household measurements.
- Newer hybrid artifacts, calibrated confidence and arbitrary-upload feature contributions are not fabricated.
- No deployment, utility account connection or hardware switching is included.
- Local readings/history, environments, dependencies, caches and generated builds are excluded from new Git commits.
