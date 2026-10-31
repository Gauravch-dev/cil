# Evaluation and Application Verification

The requested predictive evaluation is [the exact existing-model evaluation](evaluation/existing_model/EVALUATION.md). It loads the deployed XGBoost, performs inference only, and verifies unchanged model/encoder/schema hashes. The reproduced metrics are **96.89% accuracy, 65.64% positive precision, 79.08% positive recall and 71.74% positive F1** on 9,187 inferred original holdout records. No new training or model selection is involved. The result is original-evaluation reconstruction; an independently labelled unseen evaluation set has not been identified. Metrics from separately retrained models do not evaluate this artifact.

The latest unit run passes all 70 tests, with output in `evaluation/grouped_evaluation_unit_tests.log`. The following historical restoration checks document the original interface release separately from predictive performance:

- 63 unit tests passed, 0 failed.
- 26 current two-page dashboard checks passed, 0 failed.
- Forty seeded valid scenarios match the original api.py feature arrays and rounded results exactly.
- Canonical original examples: Typical 0.0779/Low; Drier 0.0038/Low; Wetter 0.7840/High. A fixed medium case returns 0.6520/Medium.
- Full original category vocabularies verified: 6 climate, 11 land-cover, 25 soil choices.
- Invalid rainfall totals, missing/nonfinite values, unknown categories, runoff bounds, zero/fractional rain, original band boundaries, preset changes, input persistence and stale-result removal checked.
- Both public pages work without a separate API or network data fetch. No model/evaluation panels appear in the UI.
- The supplied model SHA256 is unchanged: aad88a12f4645f3703c996641a15ce40603f6d5a608918fad4f37496d2f89395.

Evidence: evaluation/riskradar_restore_verification.json and evaluation/two_page_dashboard_verification.json. These checks verify application behaviour and original-predictor parity, not new forecasting accuracy. The earlier Kurundwad and other experiment scores remain separate in their original folders.
