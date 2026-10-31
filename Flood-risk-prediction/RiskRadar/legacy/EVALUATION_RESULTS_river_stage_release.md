# Current river-dashboard validation

This describes the river-level release, not the archived annual-risk API. Earlier findings are retained in `legacy/EVALUATION_RESULTS_before_river_forecast.md`.

- 24 unit tests passed, zero failed, using `python -m unittest discover -s tests -v`.
- 27 Streamlit interaction checks passed, 0 failed: all three gauges across 2023/2024/2025, all observation-based presets, recorded outcomes, exports, rainfall views/controls and navigation back to forecasts.
- Frozen source/protocol/selection/split/model checksums match; original annual model remains unchanged.
- All 1156 test predictions replay; largest difference 5.68e-14 m.
- 9 runtime checks confirm identical predictions with future observations removed.
- Actual browser inspection confirms the two-page dashboard and interactive forecast/actual chart render on localhost.

These are implementation checks, not measures of forecasting skill. Actual chronological metrics and baseline/episode limitations are in [MODEL_REPORT](MODEL_REPORT.md). No new validation of the archived API is claimed.
