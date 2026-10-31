# INDOFLOODS XGBoost comparison

This experiment estimates severe versus other recorded flood events. It does not predict whether a flood will occur, and it does not validate the old application classifier.

## Requested alternatives

1. Exact reconstruction of the existing flood/no-flood model is unavailable: these files contain no monitored non-flood samples, and the old runoff curve-number input is absent. No labels were invented or missing inputs guessed.
2. With the permitted scope adjustment, two new XGBoost models were trained on the publisher’s Flood/Severe Flood labels. Static catchment/climate features form the first version. The second adds rainfall during the 1–10 days before event onset.

## Data and chronology

The supplied public files contain 4,548 events from 155 gauges: 2,919 Flood and 1,629 Severe Flood. Metadata also describe 59 restricted gauges without public event records; those gauges are not non-flood examples.
All events join one-to-one to rainfall and many-to-one to gauge/catchment attributes. The 155 catchment shapefiles are geographical context, not additional independent samples.

| Split | Events | Flood | Severe | Gauges | Onset range |
|---|---:|---:|---:|---:|---|
| train | 3250 | 2167 | 1083 | 88 | 1965-07-21 to 2009-11-12 |
| validation | 613 | 386 | 227 | 74 | 2010-06-04 to 2014-12-16 |
| test | 625 | 332 | 293 | 130 | 2015-05-18 to 2019-12-28 |

All 2020 events excluded before fitting: EM-Earth version 1 ends in 2019; 54/55 source 2020 events have T10d=0.
The original rainfall source covers 1950–2019: https://journals.ametsoc.org/view/journals/bams/103/4/BAMS-D-21-0106.1.xml
Events whose labels extend beyond a split cutoff are purged from that split. All partitions and exclusions were frozen before fitting. No district or month was selected using test outcomes.

## Selection and leakage controls

Highest validation average precision (PR-AUC/AP), then F1. Decision threshold maximises validation F1, with F2 as first tie-breaker.
Each XGBoost family used 16 prespecified configurations; 32 XGBoost fits in total. Preprocessing and class weights use training data only. The models remain fitted on training rows only; neither is refitted after final evaluation.
Event peaks, event duration, recession, flood volume and post-onset dates are outcomes and excluded from predictors. Gauge/Event IDs, the warning/danger thresholds, GDP/HDI/population vintages and district-impact/DFSI aggregates are also excluded. The seasonal features represent the day before known onset.
Twenty tabulated severe events equal the displayed danger level. All supplied labels match the inclusive >= relation. Publisher labels are preserved, and peak water level is never used to predict the label it defines.

**Model selected before test evaluation: xgb_static.**

| Model | Precision | Recall | F1 | F2 | PR-AUC/AP | ROC-AUC | Accuracy | False severe / other floods |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| xgb_static | 0.5429 | 0.6485 | 0.5910 | 0.6242 | 0.5528 | 0.6013 | 0.5792 | 0.4819 |
| xgb_rainfall | 0.5043 | 0.8020 | 0.6192 | 0.7173 | 0.6118 | 0.6241 | 0.5376 | 0.6958 |
| logistic_rainfall_baseline | 0.4844 | 0.8464 | 0.6161 | 0.7363 | 0.5225 | 0.5669 | 0.5056 | 0.7952 |
| training_prevalence_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4688 | 0.5000 | 0.5312 | 0.0000 |
| always_severe_baseline | 0.4688 | 1.0000 | 0.6383 | 0.8152 | 0.4688 | 0.5000 | 0.4688 | 1.0000 |

Frozen winner confusion matrix (Severe Flood is positive): TN=172, FP=160, FN=103, TP=190. These false positives mean a recorded Flood classified as Severe Flood, not a false flood-occurrence alert.

Validation results:

| Model | Threshold | F1 | PR-AUC/AP |
|---|---:|---:|---:|
| xgb_static | 0.300 | 0.6730 | 0.6873 |
| xgb_rainfall | 0.125 | 0.6203 | 0.6588 |
| logistic_rainfall_baseline | 0.250 | 0.5823 | 0.4785 |

## Critical result

Rainfall improves observed test F1 from 0.5910 to 0.6192 and average precision from 0.5528 to 0.6118. It raises recall from 0.6485 to 0.8020, but false severe classifications increase from 160 to 231. These are validation-selected operating thresholds, not thresholds changed after seeing test labels.
Both XGBoost F1 values are below the always-severe baseline (0.6383). They rank cases better than constant scores by average precision, but calibration and operating-point usefulness are weak. The constant training-prevalence baseline also has a slightly lower Brier error than either XGBoost model. This is not evidence of a dependable warning system.
The selected static model has ROC-AUC 0.7412 for 318 later events at gauges present in training, versus 0.4229 for 307 events at new gauges. This diagnostic exposes poor transfer; it is not used to remove gauges or change the final test population.
No model is promoted to the application. The validation-selected static version remains the frozen experimental selection. The rainfall version is retained for transparent comparison; choosing it as a new winner solely from test results would invalidate selection.

Paired 95% gauge-cluster bootstrap intervals for rainfall minus static (500 replicates):
{"pr_auc_ap": [-0.029447839531608722, 0.12109504208241392], "f1": [-0.027609479048722095, 0.08655627618936769], "replicates": 500}
An interval crossing zero does not establish a reliable improvement. Model choice is not changed using these test intervals.

## Additional inventory files

India Flood Inventory has 6,876 reported-event rows. District flooded area and impact files each contain 732 rows; DFSI has 744 rows. These are different observational units and labels from the gauge-event database. They are retained separately. District whole-period impacts cannot safely be joined as pre-event predictors, and missing inventory reports are not verified no-flood days.

## Interpretation and limits

Rows are sampled conditional on floods and aligned to their known onset. This is retrospective severity estimation, not a deployable flood-occurrence warning.
Modern static catchment/climate summaries are retrospective proxies, not a reconstruction of covariates issued in each historical year. Anthropogenic vintage fields excluded.
The chronological test contains both known and new gauges. Gauge transfer is poor in the reported subgroup analysis. Event rows within a gauge are dependent. Class probabilities have not been independently calibrated as warning probabilities.
Static climate/terrain proxies may be predictive of location-specific historical severity. The rainfall model is assessed against the same static inputs and seasonal variables, so the comparison isolates its added antecedent rainfall information.
Peak discharge is available for 3,993 events and could support a separate regression task. It is not a model input here. No rainfall-amount forecast is trained in this experiment.

## Artifacts and reproduction

Source files, merged data, partition manifest, validation candidates, frozen selection, fitted pipelines, XGBoost UBJ models, feature importance and final predictions are saved in this experiment directory. Gain importance describes fitted splits and is not causal attribution.
```powershell
.\.venv\Scripts\python.exe scripts/compare_indofloods_xgboost.py prepare --source C:/Users/Dell/Downloads
.\.venv\Scripts\python.exe scripts/compare_indofloods_xgboost.py select
.\.venv\Scripts\python.exe scripts/compare_indofloods_xgboost.py evaluate
```
The run refuses to overwrite a frozen selection or repeat final evaluation. Use a new versioned experiment for another prespecified study. The existing application model and API are preserved.

## Sources

- INDOFLOODS: https://zenodo.org/records/14584655
- INDOFLOODS paper: https://doi.org/10.1175/BAMS-D-24-0008.1
- Variable definitions: supplied variables_description_indofloods.pdf, especially pages 5 and 12.
- EM-Earth: https://doi.org/10.1175/BAMS-D-21-0106.1
- IFI-Impacts: https://zenodo.org/records/16994648
