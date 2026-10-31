# Verification of the corrected rainfall and flood analysis release

- **43 unit tests passed, 0 failed**, including nine new rainfall/flood semantic checks.
- **24 current dashboard checks passed, 0 failed**: navigation, filters, rainfall windows, event estimates, inventory and downloads.
- **15 preserved rainfall-history checks passed, 0 failed**: actual chart data, anomalies, leap-year/date controls, uncached-location outage handling and recovery. Obsolete Argentina/annual-risk steps were archived and updated to India/current navigation.
- **625 held-out events / 1250 model-score replays** match frozen results; maximum difference 2.91e-08. Three runtime outcome-mutation checks confirm realised levels/true labels are not predictors.
- Source/data/selection/prior-model checksums match. Verified non-flood examples: 0. Invented non-flood examples: 0.

Replay is runtime verification, not a new predictive evaluation. Actual model metrics and baseline limits are in MODEL_REPORT.md. The academic report omits software/data/API test counts as previously requested; this developer-facing file retains them.

The updated dashboard check includes visible Prediction, saved inputs, manual editing, separate outcome display, regional map, monsoon and anomaly views. The new rainfall benchmark replays all 2,189 test origins for the ML and baseline, with three future-mutation checks. This is rainfall-category verification, not flood-occurrence accuracy.
