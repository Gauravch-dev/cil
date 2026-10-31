# Current implementation review

The end-to-end historical river-stage forecasting lifecycle is implemented and the simple dashboard works. It addresses the missing dated-observations/labels problem by using actual CWC manual river-level observations and official station-stage thresholds. It narrows the claim to three gauges and a numeric next-24-hour observed peak.

## What improved

- Dated inputs and targets with chronological training, separate validation/calibration/test and boundary purging.
- One shared causal feature path for training and inference, right-closed hourly bins, no fabricated missing targets and conservative data-quality exclusions.
- Five regression algorithms, three baselines, frozen selection and saved artifacts/metadata.
- Directly interpretable metres and official stage rules, with uncertainty visible when thresholds are crossed.
- Only two user-facing pages, interactive charts, archived dates, optional actual-outcome comparison and downloads.
- Earlier annual-flow models/UI retained in history rather than presented as current flood evidence.

## Remaining limitations and blockers

1. ML does not consistently outperform simple baselines: lower RMSE, but higher average absolute error and weaker danger-onset results than damped trend on this test. Do not market this as a solved performance improvement.
2. Only two observed danger episodes in the test archive; one detected in advance, one missed. Station-day classification metrics are not counts of independent floods.
3. Missing observations and strict eligibility exclusions bias coverage toward high-frequency observation periods. Gaps are unknown rather than no flood.
4. Archive revisions and publication delay are not historical real-time availability. The website is explicitly historical replay, not live warning delivery.
5. Broad QC does not validate every observation. Historical datum/threshold consistency is assumed; the official station datum is not water depth.
6. No upstream inflow, reservoir releases, reliably continuous co-located rainfall or district inundation labels are model inputs. This cannot establish area-level damage, evacuation need or all-Maharashtra risk.
7. Empirical forecast ranges have measured coverage on this test, not guaranteed future coverage. Threshold-crossing status can remain uncertain.

The valid release is an experimental historical ML demonstration. Better operational performance requires independently verified continuity/datum, richer causal hydrological inputs, more independent events and fresh prospective evaluation. The final test has not been used to select a different model.

Full evidence: [MODEL_REPORT](MODEL_REPORT.md), [experiment](experiments/river_stage_v1/MODEL_REPORT.md), [verification](experiments/river_stage_v1/verification.json).
