# RiskRadar: model and data evaluation

## Causal annual-history experts with precision guards

The next experiment adds information available from earlier annual-flow records: last, mean, median, maximum and standard-deviation discharge, last-three-observation mean, past exceedance fraction, count and age. Reference values must come from the same station in strictly earlier years. Current-year discharge and unit discharge remain excluded. Every final validation/test query uses the original training reference through 1999 only; no validation/test outcomes become later inputs. Six causal-history tests cover current/future-value mutation, same-year exclusion, unknown years and absent history.

The three original forward windows now validate all stations in each later window, while hashed group i remains fully excluded from that fold fitting. Two routes are evaluated separately: stations with at least one prior training observation, and stations without any reference history. The held-out station group has no historical input, preventing exposure of its outcomes. Per-fold cutoffs, preprocessing, reference tables and fitting use fold training only. Ten candidates across Random Forest, HistGradientBoosting and history-route Logistic Regression produce 30 CV fits and five final pipelines. History models use 24 static plus 10 causal-history inputs; no-history models use 24 static inputs.

Operating-point selection prioritises precision ≥60%, recall ≥75% and false-positive rate ≤3%, with at least 20 predicted positives in pooled OOF. If the joint target is infeasible, precision/FPR guards take priority and F1 is maximised; recall failure is explicit. This replaces the prior unconstrained F2 fallback. Selection chooses **Random Forest with history** (OOF precision 80.10%, recall 75.53%, F1 0.7774) and **HistGradientBoosting without history** (OOF precision 60.20%, recall 43.24%, F1 0.5033). The no-history route does not meet the recall goal. The choices and thresholds are frozen before outer validation/test.

| Test result | Previous precision model | Previous recall model | New experts |
|---|---:|---:|---:|
| Precision | 0.8894 | 0.4245 | 0.8172 |
| Recall | 0.6531 | 0.9313 | 0.7406 |
| F1 | 0.7532 | 0.5832 | 0.7770 |
| F2 | 0.6898 | 0.7518 | 0.7548 |
| True positives | 209 | 298 | 237 |
| False positives | 26 | 404 | 53 |
| False negatives | 111 | 22 | 83 |

New history-route precision/recall/F1 are 0.8168 / 0.7825 / 0.7993 (4,845 rows; 285 positives). The no-history route has precision 0.8235 but recall 0.4000, detecting 14 of 35 positives with three false alerts. Thus aggregate improvement does not resolve missing-history detection. Combined scores are not on a proven shared calibrated probability scale; route-specific ranking is clearer than pooled AP. No calibrated flood probability is claimed.

The historical-mean baseline has history-route precision 0.8084, recall 0.8140 and F1 0.8112, exceeding the learned RF's F1. This shows strong station-level persistence and does not establish extra ML value over the simple baseline. Added information and operating criteria changed, so improvements cannot be attributed solely to a better algorithm. Paired station-bootstrap 95% F1-difference intervals are [-0.0091, 0.0583] versus the previous precision model and [0.1453, 0.2418] versus the previous recall model. Data/test reuse makes these exploratory comparisons. Untuned outer validation has combined precision 0.7665, recall 0.7901 and F1 0.7781.

The training snapshot yields median history age 17 years in test. These are retrospective annual-flow descriptors, not recent weather or an operational event forecast. Target remains high annual discharge relative to the same training-only cutoff 1,129.84. Original source rows, final labels and chronology are preserved.

All five pipelines and combined decisions replay exactly. The verified `selected_experts.joblib` bundle includes the two selected models, historical training reference and thresholds; `discharge_history.predict_annual_discharge` accepts station/year plus static inputs without any current outcome. The report, bundle manifest, causal features, OOF records, final predictions and supplementary uncertainty analysis are under [gloflo_history_v1](experiments/gloflo_history_v1/MODEL_REPORT.md). Previous studies and the website are unchanged. This is a partial balance improvement; cold-station recall and baseline superiority remain limitations.

## Expanded features with station-aware chronological validation

Implemented the proposed experiment using only existing GloFlo data. Added drainage area, basin length, sinuosity, catchment shape/drainage and climate descriptors; blocked annual/unit peak discharge, station/location identifiers and dam counts from model inputs. Compared the legacy 13 features, an expanded training-filtered set, and legacy13 plus 12 training-ranked additions. Missingness, correlation filtering, medians, category vocabularies and supervised feature selection fit within each training fold, then refit on the full training partition for final pipelines. Feature-selection forests balance station influence; feature candidates and year/station split rules are fixed before modelling.

Station groups use SHA256(station ID) modulo three, independently of outcomes. Forward folds train through 1959 / 1979 / 1989 and validate in 1960–1979 / 1980–1989 / 1990–1999, respectively. Each fold removes every record of its validation stations from training and derives its own discharge cutoff from its training rows. All folds lie within the original 1900–1999 training partition. Final target cutoff remains 1,129.84; outer validation and test rows/labels remain unchanged.

CatBoost (ordered boosting, native categories and chronological input order), HistGradientBoosting and Random Forest each compare two configurations under three feature procedures: 18 candidates, 54 CV fits and nine full-training pipelines. Model selection maximises mean held-out-station fold average precision, with minimum-fold AP and pooled F2 as tie-breakers. Thresholds maximise pooled out-of-fold F2 while seeking precision ≥0.50 and recall ≥0.75; if infeasible, the predefined fallback maximises F2 and records failure. No outer-validation/test tuning or refitting occurs. This uses previously examined data and is exploratory, not fresh independent confirmation.

**CV selects HistGradientBoosting with 37 inputs**, 400 iterations, 31 leaves, learning rate 0.05, L2 5, minimum leaf size 20 and positive weight sqrt(training imbalance). Mean station/time CV AP is 0.6080 versus 0.4010 for the legacy13 HistGradientBoosting control. Its frozen threshold is 0.051431943029. The pooled CV operating point has 82.31% recall but 37.73% precision: **the joint operating target was not met**.

| Final test metric | Previous frozen RF winner | New station-aware HistGradientBoosting |
|---|---:|---:|
| Precision | 0.8894 | 0.4245 |
| Recall | 0.6531 | 0.9313 |
| F1 | 0.7532 | 0.5832 |
| F2 | 0.6898 | 0.7518 |
| PR average precision | 0.8431 | 0.8659 |
| ROC-AUC | 0.9707 | 0.9770 |
| False positives | 26 | 404 |
| False negatives | 111 | 22 |
| Unseen-station true positives / positives | 0 / 35 | 29 / 35 |
| Unseen-station precision | 0.0000 | 0.4203 |
| Unseen-station recall | 0.0000 | 0.8286 |
| Unseen-station F1 | 0.0000 | 0.5577 |
| Unseen-station AP | 0.6883 | 0.6106 |

This improves detection at the chosen operating point, with substantially more false alerts. It does not establish an overall predictive fix: all-station F1 worsens, the precision requirement fails in CV, and unseen-station ranking AP decreases. Within the new protocol, expanded inputs reduce unseen-station false positives from 72 to 40 versus the legacy13 HistGradientBoosting control while both detect 29 positives, but AP decreases. Features, selection procedure and operating objective changed, so the full before/after gain cannot be attributed to added inputs or CatBoost alone. CatBoost did not win CV selection. None of these models is promoted to the website.

The paired station-bootstrap 95% F1-difference intervals versus the previous frozen winner are [-0.2302, -0.1079] for all test rows and [0.3851, 0.6987] for unseen stations. These describe an exploratory comparison of different operating points on previously examined data. The untouched-by-selection outer validation check has unseen-station precision 0.5000 and recall 0.5556 (18 positives), underscoring threshold transfer uncertainty. The high-discharge labels and static attributes still do not support verified flood-event prediction at a lead time.

All nine saved pipelines reproduce recorded test predictions. The three station/time folds, feature exclusions, training-only medians and 18-candidate selection were verified. Models, feature audits, manifests, merged data, predictions, metadata and figures are preserved under [gloflo_station_features_v1](experiments/gloflo_station_features_v1/MODEL_REPORT.md). Previous studies and the application model remain unchanged.

## Expanded learning algorithm and sampling comparison

The follow-up study freezes the same 13 inputs, high-discharge target (training-only cutoff 1,129.84), source rows and chronological partitions used below. Five learners—Logistic Regression, Random Forest, Extra Trees, HistGradientBoosting and XGBoost—each compare two configurations under seven treatments: untreated, class weights, random oversampling, random undersampling, SMOTENC, SMOTENC with majority-only Tomek cleaning, and SMOTENC with majority-only ENN cleaning. Balanced Random Forest, balanced bagging and EasyEnsemble each add two internally sampled configurations. Total: **76 candidate fits; 38 saved pipelines**.

All preprocessing, class weights, synthetic generation, neighbour cleaning and ensemble sampling use training rows only. Model inputs use numeric medians and scaling plus one-hot categories, fitted on original training rows. Validation and test preserve natural class distributions. Decision thresholds and model choice maximise validation positive-class F1, breaking ties by AP and precision. The same validation and test periods have already been examined in earlier studies; these results are an exploratory follow-up, not a new independent confirmation. The wider search can overfit validation.

**Validation-selected model:** Random Forest with 1:1 random oversampling, 300 trees, maximum 1,024 leaves per tree, minimum leaf size 2, and threshold 0.964885869955962. Validation F1 is 0.7972. Training duplication produces 70,650 rows, 35,325 per class. Selection was frozen before evaluating the new models on the test period.

Each row below uses that learner's validation-selected configuration and treatment:

| Learner / treatment | Test precision | Recall | F1 | F2 | PR AP | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest / oversampling (selected) | 0.8894 | 0.6531 | 0.7532 | 0.6898 | 0.8431 | 0.9707 |
| HistGradientBoosting / untreated | 0.9083 | 0.6813 | 0.7786 | 0.7171 | 0.8528 | 0.9746 |
| Extra Trees / untreated | 0.8773 | 0.6031 | 0.7148 | 0.6433 | 0.8328 | 0.9683 |
| XGBoost / class weights | 0.7879 | 0.7312 | 0.7585 | 0.7419 | 0.8199 | 0.9735 |
| Balanced Random Forest | 0.8140 | 0.6562 | 0.7266 | 0.6827 | 0.8105 | 0.9716 |
| Balanced bagging | 0.7137 | 0.5375 | 0.6132 | 0.5654 | 0.6926 | 0.9584 |
| Logistic Regression / oversampling | 0.3871 | 0.2625 | 0.3128 | 0.2806 | 0.2703 | 0.8629 |
| EasyEnsemble | 0.3230 | 0.4906 | 0.3896 | 0.4445 | 0.2774 | 0.8634 |

The selected model catches 209 of 320 positives, misses 111 and generates 26 false positives (accuracy 97.43%). Previous balanced XGBoost caught 230, missed 90 and generated 78 false positives (F1 0.7325). The new model trades recall for precision. Its paired station-cluster bootstrap 95% F1-difference interval versus that previous model is [-0.0220, 0.0599], spanning zero. The untreated Random Forest test F1 is 0.7744, exceeding the oversampled version despite lower validation F1; no reliable sampling benefit is established. HistGradientBoosting has the highest observed test F1, but it is not substituted for the validation-selected winner after inspecting test results.

**Important transfer failure:** at the frozen operating threshold, the selected model detects 0 of 35 positives at unseen stations (494 rows); all 209 detected positives belong to seen stations. Its seen-station F1 is 0.8038. Aggregate test scores therefore do not support reliable new-station deployment. Brier score is 0.0316 versus previous untreated XGBoost's 0.0250, and no calibrated flood probability is established. The high-discharge target and contemporary static attributes do not establish verified flood occurrence or a lead-time weather forecast.

The full 38-cell comparison, sampling audit, uncertainty intervals, frozen models and metadata are in [gloflo_learners_v1](experiments/gloflo_learners_v1/COMPARISON_REPORT.md). The source, partitions, previous studies and existing website model remain unchanged. No experimental model is deployed.

## SMOTE follow-up comparison

At the user's request, SMOTENC was applied to the same training partition only. This variant handles mixed numeric and categorical inputs without interpolating soil/climate category codes. Numeric medians and standardisation for neighbour distances are fitted on original training rows; generated numeric values are returned to the original scale before the original training-fitted model encoding. All 13 input features, labels, dates and source rows remain the same as in the imbalance study. Validation and test are not sampled.

The frozen follow-up protocol tests 18 candidates: three minority/majority ratios (0.25, 0.5, 1), two neighbour counts (3, 5), and the same three XGBoost configurations. Positive class weight is 1 to isolate synthetic sampling. Validation F1 selects ratio 0.5, k=3 and 300 trees of depth 5, learning rate 0.08, min child weight 5, L2 5 and L1 1. This produces 52,987 training rows, including 17,662 positives and 15,803 synthetic examples. Its validation-selected threshold is 0.7979285121 and validation F1 is 0.7086, below the previous selected model's 0.7337.

| Chronological test metric | Untreated | Previous balanced model | SMOTENC |
|---|---:|---:|---:|
| Positive precision | 0.7647 | 0.7468 | 0.8008 |
| Positive recall | 0.6906 | 0.7188 | 0.6281 |
| Positive F1 | 0.7258 | 0.7325 | 0.7040 |
| Positive F2 | 0.7043 | 0.7242 | 0.6564 |
| PR average precision | 0.8017 | 0.7823 | 0.7897 |
| ROC-AUC | 0.9693 | 0.9721 | 0.9694 |
| False positives | 68 | 78 | 50 |
| False negatives | 99 | 90 | 119 |

SMOTENC is more precise at the chosen operating point, but misses more positives and lowers F1. It does not solve the predictive limitations of the inputs. Its station-cluster bootstrap 95% F1-difference interval versus untreated is [-0.0590, 0.0120], including zero. Unseen-station F1 remains 0.4400. Brier score is 0.0400 versus untreated 0.0250; scores are not calibrated flood probabilities. At the default 0.5 threshold recall is 0.8500 with 256 false positives and F1 0.6415; this is not substituted for the frozen validation-selected operating point.

The 2006–2015 test was already examined in the previous experiment. This is explicitly a follow-up comparison on the same records, not new independent confirmation. Selection still uses validation only; the previous balanced model remains the validation winner, and no model is changed based on test results. Synthetic rows represent statistical interpolations, not measured catchments or physically validated flood events. Artifacts and the full comparison are frozen under [gloflo_smotenc_v1](experiments/gloflo_smotenc_v1/SMOTE_REPORT.md). The existing website model and previous experiments remain unchanged.

## Class imbalance re-evaluation

The GloFlo training partition contains 1,859 high-discharge records and 35,325 others, approximately 1:19. The new target is annual peak discharge above the training-only 95th percentile, 1,129.84 in source units. The fixed chronological partitions are 1900–1999 training (37,184 rows), 2000–2005 validation (3,387), and 2006–2015 test (5,339; 320 positives). Twenty-two invalid-year records are excluded. This target and split differ from the original reconstruction below, so its scores are not a matched before/after comparison.

Thirty-three XGBoost candidates compare no treatment, class weights, random oversampling, random undersampling, and balanced oversampling with positive weight 2. Sampling, imputation and category vocabularies use training rows only. Each candidate's decision threshold and the final winner are selected on validation positive-class F1; validation and test retain natural class proportions. The selected treatment is balanced oversampling plus positive weight 2 at threshold **0.9108189344**. Its validation F1 of 0.7337 is only slightly above the untreated candidate's 0.7333. Final selection was frozen before test evaluation and was not changed after observing test metrics.

| Chronological test metric | Untreated, validation-tuned threshold | Selected imbalance treatment |
|---|---:|---:|
| Positive precision | 0.7647 | 0.7468 |
| Positive recall | 0.6906 | 0.7188 |
| Positive F1 | 0.7258 | 0.7325 |
| Positive F2 | 0.7043 | 0.7242 |
| PR average precision | 0.8017 | 0.7823 |
| ROC-AUC | 0.9693 | 0.9721 |
| False positives | 68 | 78 |
| False negatives | 99 | 90 |

The selected model finds nine more positives with ten more false alerts. Its paired station-cluster bootstrap 95% interval for the F1 difference is [-0.0277, 0.0412], including zero. Ranking by AP and probability calibration worsen: Brier score increases from 0.0250 to 0.0711. Oversampling changes score scale; the score must not be displayed as a calibrated flood probability. At the default 0.5 threshold the combined treatment produces 492 false positives, versus 78 at its validation-selected threshold; threshold selection is a separate part of the intervention. Both columns above use validation-tuned thresholds for fairness.

On stations absent from training, the selected model has recall 0.3429 and F1 0.4364 (494 rows; 35 positives), so location transfer remains limited. GloFlo was previously inspected during source reconstruction; this is a held-out fitting/selection protocol, not pristine external validation. The task classifies high annual discharge from static catchment/climate proxies, not flood occurrence at a forecast lead time. The fitted pipelines and final evaluation are frozen under `experiments/gloflo_imbalance_v1/`; the application model remains unchanged. See [the full imbalance comparison](experiments/gloflo_imbalance_v1/IMBALANCE_REPORT.md).

## Original dataset and target reconstruction

The original raw-data connection is now strongly corroborated by artifact replay, beyond the README citation. Applying the saved encoders, 13-feature formulas and median imputation to the user's existing GloFlo `kuntla_flooddatabase.csv` reproduces all **2,538 saved high-risk map rows in the same order**, with maximum score difference below 0.000001. All four categorical encoder vocabularies match the source exactly.

A highly corroborated target hypothesis is `Annual peak discharge > full-source 95th percentile` (threshold **1,145.198 in source units**), followed by an 80/20 stratified random-row split with seed 42. Replaying the unchanged saved model on that inferred split reproduces accuracy **96.8869%**, positive-class recall **0.79085**, and negative-class F1 **0.98353**, which round to every advertised headline value. Positive-class precision is **0.65642** and positive-class F1 is **0.71739**. The reconstructed log-loss differs from the model's stored best validation loss by only **0.000000156**. The README therefore appears to quote F1 and recall from different class rows.

This is forensic reconstruction, **not a new independent evaluation**. The author's private training notebook and exact SMOTE interpolation rows are still unavailable. The target denotes unusually high annual peak discharge, not verified flood/no-flood observations. In the inferred random split, 9,029 of 9,187 held-out rows are at stations also in training; repeated static attributes cross partitions. These numbers must not be compared directly with the different INDOFLOODS severity task below or represented as real-world forecasting accuracy.

See [PROVENANCE_FINDINGS.md](evaluation/upstream_forensics/PROVENANCE_FINDINGS.md), the machine-readable evidence and reconstructed dataset in that directory. Reproduce without training using `scripts/reconstruct_original_dataset.py`. The earlier provenance/validation discussion below records the initial assessment; this section supplies the newly reconstructed target and metric interpretation while preserving the independent-evaluation caveats.

## New scoped XGBoost training comparison

On 24 October 2026, two new XGBoost pipelines were trained using the user-supplied INDOFLOODS files. The target is **Severe Flood versus Flood among recorded flood events**, not flood occurrence. The old model cannot be reconstructed from these files: monitored non-flood observations and its runoff curve-number input are absent. The India Flood Inventory and district impact/DFSI files were assessed and retained separately; historical whole-period impact summaries were not used as pre-event predictors.

The data contain 4,548 events from 155 gauges. A merged event table and fixed split manifest are in `experiments/indofloods_severity_v1/data/`. Training uses 3,250 events through 2009, validation uses 613 events in 2010–2014, and the final test uses 625 events in 2015–2019. All 55 events starting in 2020 were excluded before fitting because the documented antecedent rainfall source ends in 2019. Five events spanning split cutoffs were purged. No dates, districts or months were selected using test performance.

| Model | Test precision | Recall | F1 | F2 | PR-AUC/AP | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost static inputs | 0.543 | 0.648 | 0.591 | 0.624 | 0.553 | 0.601 |
| XGBoost with antecedent rainfall | 0.504 | 0.802 | 0.619 | 0.717 | 0.612 | 0.624 |
| Rainfall logistic baseline | 0.484 | 0.846 | 0.616 | 0.736 | 0.522 | 0.567 |
| Always Severe baseline | 0.469 | 1.000 | 0.638 | 0.815 | 0.469 | 0.500 |

The static model won **validation** selection (AP 0.687 versus 0.659). Its frozen threshold is 0.300; the rainfall model's is 0.125. Numeric imputation, category encoding and candidate imbalance weights use training data only. Both models were fitted on training rows only, and final test results were generated after selection. All fitted pipelines, preprocessing, parameters, feature lists and native XGBoost artifacts are saved in the experiment directory.

The result is not strong enough for operational use: both XGBoost F1 values fall below the always-severe baseline. Rainfall improves point estimates of ranking/recall, with more false severe classifications, but paired gauge-bootstrap intervals include no improvement. The selected static model transfers poorly to new gauges (test ROC-AUC 0.423). Static climate and 2015 land-cover attributes are retrospective proxies, not a reconstruction of historically available covariates. These experiments do not establish calibrated flood-warning probabilities or a strict operational forecasting backtest. No model is promoted to the website, and no old accuracy claim is treated as a comparator.

The full protocol, validation candidates, prediction records, uncertainty intervals and critical interpretation are in [COMPARISON_REPORT.md](experiments/indofloods_severity_v1/COMPARISON_REPORT.md). A five-page Word supplement with precision–recall curves and confusion matrices is saved as `reports/XGBoost_Training_Comparison_Report.docx`.

## Scope

Rainfall & Flood Risk Analysis Using Historical Weather Data. Global scope fits the supplied model and source data. The application has two views in top navigation: Rainfall history and Check flood risk. Rainfall history is the landing view. Model methodology and evaluation information are retained in this report, outside the public UI. The sidebar, technical panels and status badges were removed. Risk Map is archived under `legacy/`; its original CSV remains unchanged. The supplied classifier has not been retrained or replaced.

## Verified data

- **GloFlo:** Kuntla, S. K., Saharia, M., & Kirstetter, P. (2022), Global Flood characterization database, IIT Delhi / University of Oklahoma. https://zenodo.org/records/7158028. Article: https://doi.org/10.1016/j.jhydrol.2022.128668.
- Published file: `kuntla_flooddatabase.csv`, verified MD5 `4e0d207c21c54577bd5a1e176a96ed3e`.
- 45,932 source rows, 93 columns, 9,710 stations, 57 countries. These are records of high-streamflow extremes and catchment characteristics, not labelled flood/no-flood days.
- Years parse to 1900–2015. Twenty-two records have unusable years and are excluded from the dated peak-record extract. No exact duplicate source rows were found.
- Annual precipitation is constant within every station's records. It is a climate attribute, not that station's rainfall measurement in each event year. Repeating it against the event years would create a false historical rainfall series.
- Ten columns are entirely missing. `data/processed/data_audit.json` records their names and every column's missing-value count. Raw values are preserved; no invented labels or zero-filled weather values are created.
- `data/processed/stations.csv` is a unique station catalog. The dated peak extract merges metadata many-to-one on station ID, with a validation check against conflicting station metadata.

## Dated rainfall analysis

Daily liquid rainfall comes from Open-Meteo's historical API, explicitly selecting ERA5, `rain_sum`, millimetres and GMT/UTC day boundaries, 2000-01-01 through 2025-12-31. Documentation: https://open-meteo.com/en/docs/historical-weather-api. Attribution: Open-Meteo and Copernicus/ECMWF ERA5. This is a gridded reanalysis estimate near the station, not an observed rain-gauge record or a catchment average.

The default station is selected geographically near Mumbai from the original catalog, independently of model scores or event outcomes. Its 9,497 daily values are cached with the exact request, returned grid coordinates and retrieval time. Other source stations can be loaded on demand. The weather cache is read before any network request, so downloaded locations work offline. Dates, units, array lengths, duplicates and numeric values are checked. Negative/non-finite/invalid values remain missing. Missing dates are reindexed as missing. Annual and monthly totals require complete calendar periods; missing days never become zero rainfall.

The page offers annual totals, deviations from the selected-period mean, monthly rainfall heatmaps, daily zoom and a CSV download. Exact acquisition metadata remains in the saved weather JSON on disk; source links are displayed as a short attribution. These are descriptive analyses; the selected-period average is not a held-out model baseline or a causal climate-change conclusion. Weather history is not silently substituted into model inputs. Historical daily rainfall is not joined to later flood outcomes: GloFlo peak records end in 2015, while users may explore rainfall through 2025.

## Supplied prediction model

The existing XGBoost binary classifier accepts 13 engineered features built from 14 manually entered climate, terrain and land attributes. Existing feature formulas, artifacts and API behavior remain unchanged. Model SHA256: `aad88a12f4645f3703c996641a15ce40603f6d5a608918fad4f37496d2f89395`.

Its output is shown as a model score. Existing display rules are Low at scores ≤0.4, Medium at >0.4 through 0.7, and High above 0.7. These thresholds were supplied by the repository and have not been validated as operational warning thresholds. The score is not established here as a calibrated probability of a flood at a stated time. The rainfall explorer changes one annual climate input and evaluates the same model; it is not a rainfall forecast or feature-importance analysis.

## What cannot be reproduced

The repository advertises accuracy 96.9%, F1 0.984 and recall 0.791. These are **upstream claims, not independently measured results**. Their averaging conventions, class definition and evaluation population are not supplied. If F1 and recall referred to the same positive class and evaluation set, those two reported numbers would not be compatible; different averaging conventions would need to be documented.

The raw GloFlo file contains annual/unit peak discharge but no binary `flooded` target, nor a representative set of observed non-flood days. The repository supplies no original label-generation code, split membership, fitted imputers or full training notebook. Its map CSV has only latitude, longitude and predicted risk; predictions are not ground-truth labels. `flood_fast.py` expects feature columns that the map CSV does not contain and therefore is not a reproducible training route for these artifacts. It was not executed or used to overwrite the model.

**Independent accuracy, F1, precision, recall, ROC-AUC and PR-AUC for the supplied classifier remain unavailable.** No synthetic data, guessed flood labels or in-sample scores are presented as a genuine test. Downloading the cited source establishes provenance but does not reconstruct the missing target definition.

To validate/retrain a flood-occurrence model, obtain the original labels and preprocessing/split pipeline, or use an event definition supported by observed flood thresholds plus monitored non-event periods. Separate stations/catchments across train and test because the climate/terrain attributes repeat within stations. Use chronological splits for event-date prediction, fit preprocessing only on training data and report natural class distributions, precision/recall/F1/PR-AUC, calibration and a baseline. Do not treat unrecorded events as verified negatives.

## Result of this pass

Historical rainfall analysis, source acquisition, station merging and data-quality auditing are implemented. Existing flood-risk scenarios remain functional. Saved map work is preserved in an archive and is no longer part of the user interface. The application's historical-analysis scope now matches its problem statement more closely; independently validated flood prediction remains a documented limitation. Software checks establish correct execution and data handling, not real-world forecast skill.

Software validation: **64 checks passed, 0 failed**: 39 dashboard integration checks, 15 historical-page integration checks and 10 data-handling unit tests. They cover two-choice navigation, an empty sidebar, absence of technical panels, preservation of the map archive and model, real cached rainfall totals, yearly filtering, anomalies, leap years, unavailable weather/model services, preserved assessments, missing values, duplicate dates and invalid units. No classifier training/test metrics were generated by this pass.

## Extended validation findings

The latest extended evaluation is in [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md), with machine-readable evidence in `evaluation/current_evaluation.json`. In addition to the 64 existing checks, six artifact/consistency checks passed and eight direct API input tests produced three passes and five failures. **Overall: 73 checks passed, five failed.** The API accepts negative annual rainfall, contradictory annual/wettest-month totals, negative drainage density, unknown climate codes and negative soil codes. The web form constrains these cases, but direct API requests bypass its protections. Server-side validation is therefore incomplete. The API/model were not modified during evaluation. Independent predictive accuracy remains unavailable for the reasons above.
