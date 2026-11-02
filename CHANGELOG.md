# Changelog

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
