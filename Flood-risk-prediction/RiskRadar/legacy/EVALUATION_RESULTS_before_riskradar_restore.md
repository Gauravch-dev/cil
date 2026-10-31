# Manual Kurundwad classifier verification

- 52 unit tests passed, 0 failed, including nine new causal/manual-input checks.
- 19 current two-page dashboard checks passed, 0 failed, covering real rainfall chart values, all four rainfall views, navigation, remembered inputs, manual flood checks, optional rainfall, stale-result invalidation and current danger status. Evidence: evaluation/two_page_dashboard_verification.json. Former four-page UI checks are archived.
- All 1,766 model rows and physical warning labels reconstructed from the original CWC manual observations using the saved source quality rules.
- All 422 final test predictions replayed through the manual-input function, matching saved model decisions.
- Source, split, selected-model, frozen selection and test-prediction hashes verified. Partition label windows do not cross chronological boundaries. Future label bins are fully measured.

Predictive metrics are in MODEL_REPORT.md and experiments/kurundwad_flood_risk_v1/final_results.json. Software verification is distinct from predictive performance. The older nationwide and rainfall-category evidence is preserved in legacy and the prior experiment folders.
