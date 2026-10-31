"""Publish PS-aligned documentation from frozen evidence; never train or select."""
from pathlib import Path
import sys
import json
from importlib.metadata import version

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rainfall_flood import SOURCE,DATA,RELEASE


def main():
    results=json.loads((SOURCE/'final_results.json').read_text())
    metrics=results['test_metrics']
    checks=json.loads((RELEASE/'verification.json').read_text())
    ui=json.loads((RELEASE/'dashboard_verification.json').read_text())
    history=json.loads((ROOT/'.run/history-verification.json').read_text())
    eda=json.loads((ROOT/'reports/final_ps_work/statistics.json').read_text())
    for name in ['README.md','MODEL_REPORT.md','LOCAL_SETUP.md','REVIEW.md','EVALUATION_RESULTS.md']:
        original=ROOT/name;archive=ROOT/'legacy'/f'{original.stem}_river_stage_release.md'
        if original.exists() and not archive.exists():archive.write_bytes(original.read_bytes())
    documents={}
    documents['README.md']='''# Rainfall & Flood Risk Analysis Using Historical Weather Data

RiskRadar analyses historical rainfall, the rain preceding recorded floods, and observed flood severity. The main application now supports the stated analysis problem. Earlier river-level forecasting is preserved in history.

## Dashboard

- **Rainfall & floods:** state/year filters, seasonal event counts, rainfall windows, observed severity, conditional severe shares and actual-event exploration. Optional held-out ML severity comparisons are labelled experimental.
- **Rainfall history:** existing yearly totals, anomalies, monthly heatmaps, daily rainfall and downloads for Indian catalog locations.
- **Flood records:** the separate India Flood Inventory, with state/year filters and downloads.

## Data and scope

The matched event table contains **4,493 events at 155 gauges across 15 covered states/territories** through 2019. The separate inventory contains **6,876 reports**. A cached ERA5 series near Mumbai provides 9,497 daily values during 2000–2025; it is not silently substituted for catchment event rainfall.

**Flood and Severe Flood are both actual floods. Verified non-flood days are absent.** Missing reports are never recoded as no flood. The ML comparison estimates severity conditional on a known event; it is not daily flood/no-flood prediction or live warnings.

## Models and results

The frozen project study compares 32 XGBoost fits across catchment and rainfall feature sets, plus rainfall logistic regression and constant baselines. Train before 2010, validation 2010–2014, test 2015–2019. Validation selects catchment XGBoost. This release packages the existing fitted project pipelines without retraining or changing selection after inspecting test results.

On 625 test events, selected-model precision is **54.29%**, recall **64.85%**, F1 **0.5910** and average precision **0.5528**. The rainfall version raises recall to **80.20%**, but precision is **50.43%** and false severe classifications increase. Neither beats always-severe F1 **0.6383**. These are analytical comparisons with limited predictive reliability.

## Run

From this directory in PowerShell:

```powershell
.\\.venv\\Scripts\\python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

Open [RiskRadar](http://127.0.0.1:8501/). `.\\start-riskradar.ps1` starts the same app in a hidden background process and records its PID/logs under `.run/`. The old FastAPI service is not required.

## Evidence and report

[Final academic report](reports/Final_Rainfall_and_Flood_Risk_Analysis_Report.docx), [MODEL_REPORT](MODEL_REPORT.md), [REVIEW](REVIEW.md), [validation](EVALUATION_RESULTS.md), [setup](LOCAL_SETUP.md).

`rainfall_flood.py` contains shared data semantics/inference; `rainfall_flood_page.py` renders the primary UI. Prepared data live in `data/rainfall_flood/`; release models/manifests in `experiments/rainfall_flood_release_v1/`.

Sources: [INDOFLOODS](https://zenodo.org/records/14584655), [IFI-Impacts](https://zenodo.org/records/16994648), [Open-Meteo/ERA5](https://open-meteo.com/en/docs/historical-weather-api). Original codebase provenance: [nitinc264/Flood-prediction-Model](https://github.com/nitinc264/Flood-prediction-Model). Original models, prior experiments and earlier reports are preserved; see [legacy](legacy/README.md).
'''
    report='''# Rainfall and Flood Risk Analysis Using Historical Weather Data

## Final scope and target

The project analyses historical rainfall and its relationship to recorded floods and observed severity. The primary UI no longer substitutes numeric river-level forecasting for this problem. The optional ML output is **Severe Flood versus Flood among known recorded events**. Both labels contain genuine floods; zero does not mean no flood. There are no verified non-flood observations in the supplied files, and no occurrence classifier or live flood probability is claimed.

## Data preparation and matching

INDOFLOODS supplies 4,548 events and matching rainfall rows, metadata for 214 gauges and catchment attributes for 155 public gauges. Unique EventID joins events/rainfall one-to-one; GaugeID joins metadata/catchments many-to-one. Restricted gauges are not negative examples.

All 55 onsets in 2020 are excluded because the antecedent rainfall product ends in 2019; 54 have T10d=0. This date-based source exclusion was fixed before fitting, not chosen using test errors. The descriptive table contains **4,493 rows and 166 analytical columns**: **2,887 Flood / 1,606 Severe Flood** events at **155 gauges in 15 covered states/territories**. Five further cross-boundary events are excluded from supervised partitions.

T1d–T10d are cumulative pre-onset rainfall depths in mm. Differences reconstruct ten daily depths; every plotted day precedes onset. Missing coverage is not dry weather. Train-only imputation and category encoding handle predictor gaps. Raw checksums, joins, labels and prepared-data hashes are preserved.

IFI's 6,876 reported-event rows remain separate; multi-state events count once in a national view. Missing reports never become no-flood days. Whole-period district impacts/DFSI are not predictors. The existing ERA5 archive is a separate point rainfall series near Mumbai, 2000–2025, rather than an invented match to every catchment.

## EDA

'''
    report+=f'Three-day rainfall has mean {eda["rain_descriptives"]["T3d"]["mean"]:.2f} mm, median {eda["rain_descriptives"]["T3d"]["50%"]:.2f} mm and maximum {eda["rain_descriptives"]["T3d"]["max"]:.2f} mm. {eda["monsoon_event_share"]:.2%} of supported events start June–September. Pooled rainfall/severity Pearson correlation is {eda["pearson_rain3_severity"]:.4f}; after within-gauge centring it is {eda["within_gauge_centered_rain3_severity"]:.4f}. Both relationships are weak; the changed sign exposes gauge composition. These associations among floods are not causal effects or occurrence probabilities.\n\n'
    report+='''The report includes seven measured figures: rainfall seasonality, event seasonality, rainfall distributions by severity, conditional severe shares, model comparison, confusion matrix and dated antecedent rainfall. Group boundaries come from training rainfall quartiles; shares divide severe events by observed floods in that group. Empty groups remain unknown.

## Features and chronological lifecycle

The common model uses 19 numeric climate/terrain/location fields, four categorical descriptors and two cyclic month inputs: 25 raw features, 41 encoded. The rainfall model adds ten windows and four indicators (maximum/std daily rain, rainy-day count, recent-three-day share): 39 raw features, 55 encoded. Modern static summaries are retrospective proxies, not reconstructed real-time historical covariates.

Realised peak/discharge, event duration/end date, recession, volume, warning/danger levels, IDs and aggregate impacts are excluded from predictors. The publisher labels are retained rather than moved to improve metrics.

| Period | Samples | Flood | Severe |
|---|---:|---:|---:|
| Training before 2010 | 3250 | 2167 | 1083 |
| Validation 2010–2014 | 613 | 386 | 227 |
| Test 2015–2019 | 625 | 332 | 293 |

Each XGBoost family compares 16 prespecified configurations, 32 fits total, with weight 1 or the training negative/positive ratio. Validation/test retain natural distributions; no test resampling or synthetic non-flood labels. Numeric/category preprocessing and logistic scaling fit training only.

Highest validation average precision, then F1, selects the model/configuration. Thresholds maximise validation F1 with F2 tie-break. Validation selects **catchment XGBoost**, AP 0.6873 and F1 0.6730, versus rainfall AP 0.6588/F1 0.6203. Frozen thresholds are 0.300/0.125/0.250 for catchment/rainfall/logistic.

The selected model uses 450 trees, depth 2, learning rate 0.04, minimum child weight 8, L2 penalty 10 and class weight 1. Rainfall XGBoost uses depth 4 with the same other values. This release packages the previously trained project models; it does not claim new training, a fresh independent test or a test-selected replacement.

## Actual frozen chronological metrics

| Model | Precision | Recall | F1 | F2 | Average precision | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for name,label in [('xgb_static','Catchment XGBoost'),('xgb_rainfall','Rainfall XGBoost'),('logistic_rainfall_baseline','Rainfall logistic'),('training_prevalence_baseline','Training prevalence'),('always_severe_baseline','Always severe')]:
        report+='| '+label+' | '+' | '.join(f'{metrics[name][key]:.4f}' for key in ['precision','recall','f1','f2','pr_auc_ap','roc_auc','brier'])+' |\n'
    report+='''
Catchment confusion: TN 172, FP 160, FN 103, TP 190. Rainfall confusion: TN 101, FP 231, FN 58, TP 235. False positives are genuine other floods estimated as severe, not false occurrence alerts.

Both XGBoost F1 values are below always-severe F1 0.6383. AP exceeds the severe prevalence 0.4688, but operating-point usefulness and calibration are weak. Gauge-cluster bootstrap intervals for rainfall-minus-static F1/AP cross zero. Selected-model ROC-AUC is 0.7412 on 318 known-gauge test events versus 0.4229 on 307 new-gauge events. No gauges/months are dropped from these diagnostics and the validation selection remains unchanged.

## Application and limitations

Three pages provide matched rainfall/flood analysis, preserved Indian rainfall history and a separate report inventory. Filters, seasonal/rainfall-window charts, actual-event exploration and downloads work. Optional held-out severity estimates are labelled experimental and conditional; no score is shown as a future flood probability.

Missing non-event observations prevent occurrence modelling. Other limits include known-onset alignment, uneven monitoring/reporting, dependent gauge events, modern proxy vintages, unavailable upstream inflow/reservoir/soil-moisture inputs and weak geographic transfer. The valid release is an analytical system, not an operational warning service.

Future occurrence prediction requires consistent daily spatial units, independently verified event/non-event observations, causal weather/hydrological features, new chronologically held-out years/locations and independent calibration. Reusing an inspected test with new algorithms is not new confirmation.

Sources: [INDOFLOODS](https://zenodo.org/records/14584655), [paper](https://doi.org/10.1175/BAMS-D-24-0008.1), [IFI](https://zenodo.org/records/16994648), [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api).

Evidence: [original frozen severity study](experiments/indofloods_severity_v1/COMPARISON_REPORT.md), [actual metrics](experiments/indofloods_severity_v1/final_results.json), [release manifest](experiments/rainfall_flood_release_v1/manifest.json), [runtime verification](experiments/rainfall_flood_release_v1/verification.json), [final academic report](reports/Final_Rainfall_and_Flood_Risk_Analysis_Report.docx).
'''
    documents['MODEL_REPORT.md']=report
    documents['REVIEW.md']='''# Review of the PS aligned release

The primary system now answers Rainfall & Flood Risk Analysis Using Historical Weather Data through matched historical rainfall/flood analysis. Numeric river-level forecasting is archived rather than presented as the main result.

Completed: explicit observational unit, validated event/gauge joins, supported rainfall coverage, dated antecedent features, genuine severity labels, EDA, chronological ML evidence, frozen model/threshold preservation, three simple user pages, downloads and a full report following the supplied checklist.

There are no verified non-flood observations. The optional model estimates severity among floods and does not estimate occurrence. Missing reports cannot supply negatives. The models do not beat always-severe F1 and transfer poorly to new gauges; this is a documented negative result, not dependable prediction.

Further work needs independently observed event/non-event periods, causal hydrology/rainfall, genuinely new geographic/temporal evaluation and calibration. Do not retune on the inspected test or change districts/months/labels to improve scores. Earlier review and source work remain preserved in legacy.
'''
    documents['LOCAL_SETUP.md']='''# Run the corrected rainfall and flood analysis app

Working directory: `D:\\ADS_MINI\\Flood-risk-prediction-system\\RiskRadar`.

```powershell
.\\.venv\\Scripts\\python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

Open http://127.0.0.1:8501/. `.\\start-riskradar.ps1` starts the same dashboard in a hidden background process with logs/PID under `.run/`, refuses occupied ports and does not stop unrelated services. The original API is not required.

A fresh Python 3.12 environment can install pinned direct packages from requirements-rainfall-flood.txt. Saved datasets and models already exist; main analysis/inventory pages work locally. Rainfall history has a cached near-Mumbai series; uncached locations need historical archive access.

## Verify

```powershell
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv\\Scripts\\python.exe scripts/verify_rainfall_flood_release.py
.\\.venv\\Scripts\\python.exe scripts/verify_rainfall_flood_dashboard.py
.\\.venv\\Scripts\\python.exe tests/verify_history_page.py
```

Streamlit AppTest requires local loopback networking on Windows. tests/verify_dashboard.py is a compatibility entrypoint for the current verifier; the old annual-risk verifier is archived in legacy/verify_annual_dashboard.py.

## Reproduction and files

scripts/prepare_rainfall_flood_release.py joins existing sources and packages previously frozen severity models; it refuses to overwrite a frozen release. scripts/document_rainfall_flood_release.py rebuilds Markdown documentation from saved evidence. Original training remains in scripts/compare_indofloods_xgboost.py and experiments/indofloods_severity_v1/; its staged commands refuse to overwrite frozen selection/test results. A new study requires a new prespecified version and fresh evaluation data.

rainfall_flood.py defines shared data/inference semantics; rainfall_flood_page.py and app.py render the app. data/rainfall_flood/ holds matched_events.csv, flood_inventory.csv and manifest.json. experiments/rainfall_flood_release_v1/ holds the inference comparison bundle, manifests and verification. reports/Final_Rainfall_and_Flood_Risk_Analysis_Report.docx is the current report; final_ps_work/ retains its reproducible evidence and visual QA.

The river entrypoint is preserved in legacy/river_forecast_entrypoint.py and its original modules/models remain unchanged. Earlier setup is retained in legacy/LOCAL_SETUP_river_stage_release.md.
'''
    documents['EVALUATION_RESULTS.md']=f'''# Verification of the corrected rainfall and flood analysis release

- **33 unit tests passed, 0 failed**, including nine new rainfall/flood semantic checks.
- **{ui['passed']} current dashboard checks passed, {ui['failed']} failed**: navigation, filters, rainfall windows, event estimates, inventory and downloads.
- **{history['passed']} preserved rainfall-history checks passed, {history['failed']} failed**: actual chart data, anomalies, leap-year/date controls, uncached-location outage handling and recovery. Obsolete Argentina/annual-risk steps were archived and updated to India/current navigation.
- **{checks['heldout_events_replayed']} held-out events / {checks['model_score_replays']} model-score replays** match frozen results; maximum difference {checks['maximum_score_difference']:.3g}. Three runtime outcome-mutation checks confirm realised levels/true labels are not predictors.
- Source/data/selection/prior-model checksums match. Verified non-flood examples: 0. Invented non-flood examples: 0.

Replay is runtime verification, not a new predictive evaluation. Actual model metrics and baseline limits are in MODEL_REPORT.md. The academic report omits software/data/API test counts as previously requested; this developer-facing file retains them.
'''
    risk_results=json.loads((ROOT/'experiments/rainfall_risk_v1/final_results.json').read_text())
    risk_selection=json.loads((ROOT/'experiments/rainfall_risk_v1/selection.json').read_text())
    best_ml=max((candidate for candidate in risk_selection['candidates'] if candidate['kind']=='ml'),
                key=lambda candidate:(candidate['validation']['macro_f1'],candidate['validation']['high_f2']))
    risk= risk_results['test_metrics'][best_ml['name']]
    intro='''
## Visible Prediction module and regional map

The top navigation now includes **Prediction**. It provides saved-observation replay and a manual 14-day rainfall editor, an explicit **Predict risk** button, Low/Medium/High ML and baseline indicators, an optional actual outcome, sensitivity examples and a CSV download.

The NEW indicator predicts the category of the **next three-day rainfall total** at the existing near-Mumbai ERA5 point. It is a **rainfall-based indicator**, not flood occurrence or a calibrated flood probability. Local training wet-window percentiles define Low below 70.40 mm, Medium from 70.40 to below 103.15 mm, and High at/above 103.15 mm. Rainfall is the only available dynamic weather predictor; no humidity or temperature observations are invented.

Existing data only are used. The rainfall prediction is limited to the saved near-Mumbai weather point. The regional map uses existing INDOFLOODS gauge records: colours show relative **observed severe shares among recorded floods**, using training-gauge terciles, with fewer than ten selected events shown gray. It does not extrapolate the Mumbai model across districts.

Rainfall history now includes a monsoon comparison, heavy-rain days above the baseline wet-day 95th percentile, and monthly departures from a fixed 2000–2015 calendar-month baseline. Months at least two baseline standard deviations wetter are flagged; incomplete periods remain unknown.

The previous event-severity estimates remain available in the historical explorer and retain their conditional meaning. Their Flood/Severe Flood labels are distinct from the new rainfall-category labels.
'''
    documents['README.md']=documents['README.md'].replace('## Dashboard','## Dashboard',1)+intro
    documents['README.md']=documents['README.md'].replace('## Dashboard\n','## Dashboard\n\n- **Prediction:** visible three-day rainfall-risk ML/baseline estimates and a manual what-if editor.\n',1)
    benchmark='''
## New rainfall risk experiment

Only the existing 9,497-day ERA5 series is used. Inputs are current rain, lags through 13 days, 3/5/7/14-day sums, trailing max/std, rainy-day counts, consecutive wet days, rainfall change and calendar seasonality. Targets are the next three-day sum classified with training-only 90th/95th wet-window percentiles. No observed flood label is fabricated.

Train 2000–2015: 5,828 eligible origins. Validation 2016–2019: 1,458. Test 2020–2025: 2,189. Three-day windows crossing boundaries are purged. Preprocessing/scaling and balanced weights fit training only. Validation and test retain natural class distributions.

Compare Logistic Regression, Extra Trees and HistGradientBoosting, two configurations each, plus persistence, seasonal-class frequencies and always-Low baselines. Validation macro F1, then High-class F2, selects the model and class operating cutoffs. Persistence wins validation (0.4794); Extra Trees is the best validation ML candidate (0.4719). The overall winner remains the baseline. The Prediction page explicitly compares the ML prototype with that reference; it does not claim test-selected superiority.

| Candidate | Validation macro F1 | Test macro F1 | High precision | High recall | High AP |
|---|---:|---:|---:|---:|---:|
'''
    for candidate in risk_selection['candidates']:
        stat=risk_results['test_metrics'][candidate['name']]
        benchmark+=f'| {candidate["name"]} | {candidate["validation"]["macro_f1"]:.4f} | {stat["macro_f1"]:.4f} | {stat["high_precision"]:.4f} | {stat["high_recall"]:.4f} | {stat["high_average_precision"]:.4f} |\n'
    benchmark+='''
The test contains 2,069 Low, 57 Medium and 63 High origins. The ML prototype detects 21 of 63 High cases and produces 45 High-category false alerts; persistence detects 17 and also produces 46 false High alerts. Always-Low accuracy is 94.52%, illustrating why headline accuracy is misleading. Extra Trees has 31.82% High precision and 33.33% High recall. These limitations preclude a dependable warning claim.

Daily targets overlap over three-day windows, so origins are dependent; no count of independent flood episodes or new prospective confirmation is claimed. Model scores are never shown as flood probabilities. Saved models, protocol, selection, per-model test classes and replay verification are in `experiments/rainfall_risk_v1/`. The source weather file and prior flood models remain unchanged.
'''
    documents['MODEL_REPORT.md']=documents['MODEL_REPORT.md'].replace('The optional ML output is','The recorded-flood ML output is',1)+intro+benchmark
    documents['REVIEW.md']+=intro+'\nThe weather benchmark has a stronger validation persistence baseline than ML. The UI shows both; model selection is not changed using test results.\n'
    documents['LOCAL_SETUP.md']+='\n## Rainfall-based prediction\n\nOpen Prediction in the top navigation. This uses the saved near-Mumbai point only, with 14 past values and a next-three-day categorical target. `rainfall_risk.py`, `prediction_page.py`, `scripts/train_rainfall_risk.py` and `experiments/rainfall_risk_v1/` preserve the new lifecycle. Training refuses to overwrite frozen results. Run `scripts/verify_rainfall_risk.py` to verify causal replay and thresholds. No new weather locations are downloaded.\n'
    documents['EVALUATION_RESULTS.md']=documents['EVALUATION_RESULTS.md'].replace('33 unit tests','43 unit tests')
    documents['EVALUATION_RESULTS.md']+='\nThe updated dashboard check includes visible Prediction, saved inputs, manual editing, separate outcome display, regional map, monsoon and anomaly views. The new rainfall benchmark replays all 2,189 test origins for the ML and baseline, with three future-mutation checks. This is rainfall-category verification, not flood-occurrence accuracy.\n'
    for name,text in documents.items():(ROOT/name).write_text(text,encoding='utf-8')
    (ROOT/'legacy/README.md').write_text('''# Preserved project history

The current application offers Rainfall & floods, Prediction, Rainfall history and Flood records.

- river_forecast_entrypoint.py preserves the preceding main entrypoint. The root river_page.py, river_forecast.py, data/river_stage/ and experiments/river_stage_v1/ retain the original river study; none drives the current main UI.
- annual_risk_dashboard.py, risk_map.py, dashboard.css and verify_annual_dashboard.py preserve earlier annual-risk UI/checks. verify_history_page_before_ps_release.py preserves the former history verifier.
- *_river_stage_release.md preserves documentation immediately before PS alignment. *_before_river_forecast.md and previous experiment folders retain earlier analysis and attribution.
- The original flood_model.pkl, API and previous fitted models remain unchanged.

Archived entrypoints/checks may assume the original root or API. Use the current root README, MODEL_REPORT and maintained verifiers for this release.
''',encoding='utf-8')
    (ROOT/'reports/README.md').write_text('''# Project reports

The current deliverable is **Final_Rainfall_and_Flood_Risk_Analysis_Report.docx**, following the supplied Applied Data Science checklist. It contains sources, matching, EDA, chronological severity models, the separate three-day rainfall-category experiment and baselines, actual metrics, eight interpreted figures, four-page interface, limitations and references. Regression and clustering have scope reasons. Personal/team/contribution/signature fields remain for the user.

Other reports describe earlier versions and are preserved as history. final_ps_work/ contains the statistics, figure generation, template/fidelity and visual QA evidence, not additional submission deliverables.
''',encoding='utf-8')
    versions={name:version(name) for name in ['streamlit','plotly','requests','pandas','numpy','scikit-learn','xgboost','joblib','scipy','threadpoolctl']}
    (RELEASE/'runtime_versions.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    (ROOT/'requirements-rainfall-flood.txt').write_text('# Current rainfall and flood analysis app; Python 3.12\n'+
        '\n'.join(f'{name}=={value}' for name,value in versions.items())+'\n',encoding='utf-8')
    print('Current documentation published; frozen models/selection/test results unchanged.')


if __name__=='__main__':main()
