# Review of the PS aligned release

The primary system now answers Rainfall & Flood Risk Analysis Using Historical Weather Data through matched historical rainfall/flood analysis. Numeric river-level forecasting is archived rather than presented as the main result.

Completed: explicit observational unit, validated event/gauge joins, supported rainfall coverage, dated antecedent features, genuine severity labels, EDA, chronological ML evidence, frozen model/threshold preservation, three simple user pages, downloads and a full report following the supplied checklist.

There are no verified non-flood observations. The optional model estimates severity among floods and does not estimate occurrence. Missing reports cannot supply negatives. The models do not beat always-severe F1 and transfer poorly to new gauges; this is a documented negative result, not dependable prediction.

Further work needs independently observed event/non-event periods, causal hydrology/rainfall, genuinely new geographic/temporal evaluation and calibration. Do not retune on the inspected test or change districts/months/labels to improve scores. Earlier review and source work remain preserved in legacy.

## Visible Prediction module and regional map

The top navigation now includes **Prediction**. It provides saved-observation replay and a manual 14-day rainfall editor, an explicit **Predict risk** button, Low/Medium/High ML and baseline indicators, an optional actual outcome, sensitivity examples and a CSV download.

The NEW indicator predicts the category of the **next three-day rainfall total** at the existing near-Mumbai ERA5 point. It is a **rainfall-based indicator**, not flood occurrence or a calibrated flood probability. Local training wet-window percentiles define Low below 70.40 mm, Medium from 70.40 to below 103.15 mm, and High at/above 103.15 mm. Rainfall is the only available dynamic weather predictor; no humidity or temperature observations are invented.

Existing data only are used. The rainfall prediction is limited to the saved near-Mumbai weather point. The regional map uses existing INDOFLOODS gauge records: colours show relative **observed severe shares among recorded floods**, using training-gauge terciles, with fewer than ten selected events shown gray. It does not extrapolate the Mumbai model across districts.

Rainfall history now includes a monsoon comparison, heavy-rain days above the baseline wet-day 95th percentile, and monthly departures from a fixed 2000–2015 calendar-month baseline. Months at least two baseline standard deviations wetter are flagged; incomplete periods remain unknown.

The previous event-severity estimates remain available in the historical explorer and retain their conditional meaning. Their Flood/Severe Flood labels are distinct from the new rainfall-category labels.

The weather benchmark has a stronger validation persistence baseline than ML. The UI shows both; model selection is not changed using test results.
