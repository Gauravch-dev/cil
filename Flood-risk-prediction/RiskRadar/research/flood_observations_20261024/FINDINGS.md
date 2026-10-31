# Flood observations and label feasibility — 24 October 2026

## Main finding

Found and retrieved actual timestamped CWC river-level observations for **Kurundwad, Kolhapur, Maharashtra, on the Krishna River**, together with matching published warning/danger thresholds and co-located rainfall records. These support a prospective station-level flood-stage prediction study after quality auditing. They do not establish district-wide inundation labels or guarantee model skill.

This is source research: no new model was trained, old experiments were not retuned and the website/model were not changed.

## Verified primary sources

1. **National Water Data Portal / NWIC / Central Water Commission — manual hourly river levels.**
   - [Official dataset](https://nwdp.nwic.gov.in/dataset/river-water-level-manual-hourly-cwc-subernarekha).
   - Historical Krishna resource: `7778f459-0e82-4681-9638-5e494c36fd42`.
   - Recent Krishna resource: `7a5386e8-bf7a-43ac-af78-04565db04e79`.
   - Station filter must match **KURUNDWAD** exactly. Case-sensitive searches for `Kurundwad` returned no rows; names elsewhere can use mixed case. Match station, river, coordinates and vertical reference, not names alone.
   - Official observed location: latitude 16.68444444, longitude 74.60222222; zero-gauge RL field 519.4550 m. Cross-check historical datum before labels.
   - Read-only official CKAN `datastore_search` was accessible without login. Basin CSVs are large, so only this station's rows were downloaded. SQL aggregate queries were not available; no access restrictions were bypassed.
2. **CWC Standard Operating Procedure for Flood Forecasting, April 2025.**
   - [Official PDF, printed page 46 / PDF page index 45, row 659](https://cwc.gov.in/sites/default/files/sopapril2025.pdf#page=46).
   - Kurundwad: warning **535.64 m**, danger **537.16 m**, listed historical highest flood level **541.35 m** dated 10 August 2019.
   - These warning/danger values also match the user's existing `metadata_indofloods.csv`, gauge `INDOFLOODS-gauge-678`.
   - A [CWC bulletin dated 1 August 2021](https://cwc.gov.in/sites/default/files/cfcrcwcdfb-01.08.2021-2.pdf) independently lists the same warning/danger thresholds. Full historical threshold/datum validity still needs review.
3. **CWC manual daily rainfall, Maharashtra.**
   - [Official dataset](https://nwdp.nwic.gov.in/dataset/rainfall-cwc-manual-daily).
   - Historical resource `f834e1fe-a0e0-4134-a23f-78e3e2eb73b3`; recent `1d7a5f00-64a9-484d-a1fa-f77a8ad3f23d`.
   - Co-located `KURUNDWAD` rainfall was successfully retrieved. Sparse, nonzero-only retrieved samples and duplicate records mean missing dates must not be silently interpreted as dry days.
4. **CAMELS-IND v2.2, optional richer meteorological/streamflow source.**
   - [Publisher dataset](https://zenodo.org/records/14999580), [peer-reviewed description](https://essd.copernicus.org/articles/17/461/2025/).
   - Daily meteorology and available observed streamflow, 1980–2020, across 472 catchments; the v2.2 publisher describes 242 with >30% observed streamflow coverage. The older article's 228 count refers to its earlier release.
   - Includes model-predicted LSTM streamflow as well as observations. Modelled flow must not be mistaken for ground truth. Official flood-stage labels are not provided automatically; catchment/gauge matching remains necessary. Archives were not downloaded in this research.

## What was actually retrieved

| Kurundwad source | Rows | Actual first timestamp | Actual last timestamp |
|---|---:|---|---|
| Manual river levels, historical resource | 34,704 | 2012-10-09 19:00 | 2020-12-31 18:00 |
| Manual river levels, recent resource | 21,921 | 2021-01-01 08:00 | 2025-12-31 18:00 |
| Manual rainfall, historical resource | 820 | 2013-06-06 08:30 | 2020-11-28 08:30 |
| Manual rainfall, recent resource | 638 | 2021-02-18 08:30 | 2025-11-01 17:30 |

Times are preserved as published; India/IST interpretation and rainfall accumulation interval must be confirmed before feature engineering. Filenames advertise broader date bands than some station records actually cover. **56,625 river-level readings do not mean 56,625 independent flood events.** Coverage is irregular and incomplete.

The historical rainfall subset contains 196 duplicate timestamps (624 unique). The recent rainfall subset has 638 unique timestamps. Neither retrieved subset contains a recorded zero; this is a coverage warning, not proof that every omitted day was dry. Local rainfall also does not represent the whole upstream catchment automatically.

## Concrete prospective label examples

Danger level: 537.16 m. Each example below has 24 distinct observed future hours and one-hour internal spacing. These demonstrate feasibility; they are not a cleaned training dataset.

| Forecast time | Current level (input) | Largest observed level in next 24h (label evidence only) | Future exceedance label |
|---|---:|---:|---:|
| 2019-08-04 12:00 | 536.505 | 537.205 | 1 |
| 2018-07-15 12:00 | 531.705 | 531.745 | 0 |
| 2021-07-23 15:00 | 535.375 | 537.385 | 1 |
| 2023-07-15 12:00 | 523.955 | 524.035 | 0 |

Future maximum is used only to construct the target. It must never become an input. Missing future coverage must produce an unknown label rather than a negative label. A river below danger can still coexist with other kinds of local flooding; this target is **river danger-level exceedance**, not every form of flooding.

Possible targets:

- Regression: next-day recorded water level, or maximum recorded water level in the next 24 hours, with documented observation coverage.
- Classification: at least one valid observed danger-level exceedance in the next 24 hours.
- Onset alert: the same future target restricted to times currently below danger; already-above-danger periods should be displayed as observed status separately.

Predictors can use the current observed level, distance to warning/danger, past level changes, and past rainfall only. Future observed rainfall/levels and retrospective event duration/peak must not be inputs.

## Quality problems and remaining blockers

1. **Telemetry is unreliable for this station.** Historical telemetry has gross spikes up to 5,224.856 m. Recent telemetry has 22,429 of 22,444 readings at exactly 522.395 m; its six raw danger exceedances are all >550 m. Do not use those as flood labels. The manual series is a stronger candidate.
2. **Manual historical records still need auditing.** The raw range is 2.4–1,139.455 m; some values conflict with the published zero-gauge reference or are implausible spikes. A broad diagnostic envelope flagged 2,823 rows; it is not an approved cleaning rule. Investigate datum changes and source corrections rather than automatically adding offsets or dividing values by ten.
3. **Recent manual readings look more plausible on initial checks**, ranging 521.855–540.165 m, with no duplicate timestamps found. This is not proof every value is correct. Plausible danger exceedances occur in 2019, 2021 and 2024. Repeated hourly readings during one flood are not independent events.
4. **Few independent flood seasons.** Kurundwad alone may be insufficient for robust training/validation/test event separation. Expand to a small set of matching gauges, such as Karad and other Krishna-basin stations, chosen by coverage/metadata rather than eventual model scores. Official API samples confirm manual records for Karad and Arjunwad too, but their full quality/coverage has not yet been audited.
5. **Rainfall completeness and spatial matching.** Obtain a complete, documented series or suitable gridded estimates; retain missingness and use only observations available by the issue time. Do not fill all missing gauge rainfall with zero. Rainfall at the gauge is not necessarily upstream rainfall.
6. **Threshold and vertical reference consistency.** Confirm the warning/danger level used matches the observation datum and applies to each period.
7. **Evaluation.** Split by time and flood episode, purge overlapping future-label windows at boundaries, preserve natural event frequency, report event detection/lead time and false alerts as well as row metrics. Compare with persistence and simple level/trend baselines. Reserve a fresh final period before modelling.

## Recommended scope

**Maharashtra River Flood-Stage Prediction — next 24 hours**, beginning with a few Krishna-basin gauges.

For a controlled dashboard, forecasting water level numerically and applying official warning/danger rules is a defensible option. The data now offers physical observations and an official event threshold; a quality-audited, leakage-safe modelling table still needs to be built. This cannot yet be called district-wide flood extent prediction or an operational warning service.

## Saved evidence

- `kurundwad_manual_historical.csv` and `kurundwad_manual_recent.csv`: retrieved station-level observations.
- `kurundwad_rain_historical.csv` and `kurundwad_rain_recent.csv`: retrieved rainfall records.
- `kurundwad_published_threshold_metadata.csv`: matched existing published metadata.
- `illustrative_future_labels.csv`: four provisional future-label examples; future fields are target evidence only.
- API metadata/preview JSON files and source-inspection scripts: provenance and reproducibility.

Underlying observations and thresholds belong to CWC/NWIC and the cited publishers; preserve attribution and applicable source terms.
