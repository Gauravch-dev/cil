"""Publish documentation from frozen results without fitting or selecting models."""
from pathlib import Path
import sys
import json
from importlib.metadata import version

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from river_forecast import EXPERIMENT,DATA,STATIONS


def main():
    results=json.loads((EXPERIMENT/'final_results.json').read_text())
    verified=json.loads((EXPERIMENT/'verification.json').read_text())
    ui=json.loads((EXPERIMENT/'dashboard_verification.json').read_text())
    source=json.loads((DATA/'raw/source_manifest.json').read_text())
    splits=pd.read_csv(EXPERIMENT/'split_manifest.csv',parse_dates=['time'])
    selected=results['selected_test'];baseline=results['best_validation_baseline_test']
    for name in ['README.md','MODEL_REPORT.md','LOCAL_SETUP.md','EVALUATION_RESULTS.md']:
        archive=ROOT/'legacy'/f'{Path(name).stem}_before_river_forecast.md'
        if not archive.exists():archive.write_bytes((ROOT/name).read_bytes())

    report=f'''# Maharashtra River-Level Forecast and Flood-Stage Warning

RiskRadar now demonstrates numeric next-24-hour river-peak forecasting at three Maharashtra CWC gauges. A regressor predicts level change; fixed official thresholds derive Below warning / Warning / Danger. This is historical replay, not live warning delivery or district inundation prediction.

## Observations, target and features

- Source: {sum(x['rows'] for x in source['resources']):,} manual river-level records from CWC/NWDP, with original files, timestamps, resource IDs and SHA-256 checksums retained.
- Gauges: Kurundwad (Krishna, Kolhapur), Narsingpur and Takli (Bhima, Solapur).
- Target: maximum **observed** level in the next 24 hourly bins after 08:00 IST, excluding the current reading. Every label bin must contain an actual observation. This does not establish the unsampled continuous physical peak.
- Inputs: station, current level relative to warning, warning–danger gap, lag changes through 72 hours, trailing means/maxima/minima/std, positive rises, acceleration and calendar seasonality. No future rainfall or future river readings enter inference.
- Rainfall remains a separate historical-analysis module. Sparse co-located gauge rainfall is not filled with zero or used as a river-model predictor. Reservoir releases and upstream inflows are unavailable as features.
- Rule labels represent **gauge threshold exceedance**, not independently verified inundation, damage or district flood occurrence.

## Data preparation and coverage

Same-time conflicting readings remain unknown. Fixed broad physical bounds, gross jump checks and a causal stale-reading rule flag questionable data; no guessed decimal/datum repairs. Hourly bins are right-closed and right-labelled, so an 08:30 reading cannot enter an 08:00 forecast. Features permit at most two hours of forward filling and require at least 65 actually observed past-72-hour bins. Future labels are never filled.

The merged model dataset contains **{len(splits):,}** eligible station-days. Coverage is selective and favours high-frequency observation seasons. Missing/excluded windows are not labelled normal. The quality audit contains overlapping exclusion reasons; they must not be added as disjoint counts. This archive cannot establish source availability or reporting delays at historical issue time. Current published official threshold datum/history is assumed consistent with the selected series.

| Period | Purpose | Samples |
|---|---|---:|
| 2014–2020 | Training | {int(splits.split.eq('train').sum())} |
| 2021 | Model selection | {int(splits.split.eq('validation').sum())} |
| 2022 | Separate forecast-range calibration | {int(splits.split.eq('calibration').sum())} |
| 2023–2025 | Final chronological test | {int(splits.split.eq('test').sum())} |

Windows crossing partitions are purged. One origin per day avoids overlapping 24-hour labels, but consecutive days and gauges remain correlated. The final test was evaluated only after model selection and range calibration. Test results have not triggered retuning or replacement of the frozen selection.

## Compared models and selection

Twenty-two candidates compare Ridge, Random Forest, Extra Trees, HistGradientBoosting and XGBoost. Training-only station-balanced weights are compared with optional 4× emphasis on training peaks above official warning. Three baselines also participate: persistence, damped positive six-hour trend and previous-day maximum.

The prespecified 2021 objective equally weights macro-station MAE overall and macro-station MAE during observed warning-level target windows. **XGBoost (`xgboost_1_w1`)** wins with validation score 0.2417, versus 0.4359 for the best baseline. It uses 400 trees, depth 4, learning rate 0.04, minimum child weight 5 and L2 penalty 20. It predicts the numeric change from current level; adding the current level produces the forecast peak. It is not a classifier and does not output a flood probability.

## Actual chronological test results

| Predictor | MAE (m) | RMSE (m) | Warning-period MAE (m) |
|---|---:|---:|---:|
| Frozen XGBoost | {selected['mae']:.4f} | {selected['rmse']:.4f} | {selected['warning_period_mae']:.4f} |
| Damped positive trend | {baseline['mae']:.4f} | {baseline['rmse']:.4f} | {baseline['warning_period_mae']:.4f} |
| Persistence | {results['test_metrics']['persistence']['mae']:.4f} | {results['test_metrics']['persistence']['rmse']:.4f} | {results['test_metrics']['persistence']['warning_period_mae']:.4f} |
| Previous-day maximum | {results['test_metrics']['previous_24h_max']['mae']:.4f} | {results['test_metrics']['previous_24h_max']['rmse']:.4f} | {results['test_metrics']['previous_24h_max']['warning_period_mae']:.4f} |

**ML superiority is not established.** XGBoost has lower RMSE than these baselines but worse overall MAE than persistence and damped trend. The trend baseline also has better warning-period MAE and danger-onset recall. The validation winner remains frozen to avoid choosing a replacement from the final test. This is an experimental ML lifecycle demonstration, not an operational flood warning system.

Pooled R² is {selected['r2_relative_to_station_warning']:.4f}, computed after subtracting each station's official warning level. Raw gauge offsets around 400–540 m would inflate pooled R². R² is not classification accuracy. Per-station results appear in the detailed experiment report.

| Rule applied to predicted peak | Actual positive station-days | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Warning or above | 20 | 19 | 3 | 1 | 86.36% | 95.00% | 0.9048 |
| Danger or above | 6 | 5 | 2 | 1 | 71.43% | 83.33% | 0.7692 |
| Danger, current reading below danger | 3 | 2 | 2 | 1 | 50.00% | 66.67% | 0.5714 |

The last row removes origins already above danger, but dips within a continuing episode can still produce multiple positive days. A separate audit groups observed danger readings separated by at most 24 hours: **{verified['observed_danger_episodes']} observed episodes, {verified['episodes_with_eligible_advance_origin']} with an eligible advance origin, {verified['episodes_predicted_in_advance']} predicted in advance**. Kurundwad July 2024 was detected 23 hours before the first observed crossing; Takli September 2025 was missed. These two episodes cannot establish reliable generalisation, and unobserved gaps can hide or fragment episodes. The trend baseline detects all three positive danger-onset station-days with one false alert; this does not prove reliable episode-level performance either.

The separate 2022 calibration produces 90%-nominal empirical ranges. Actual test coverage is {results['error_range']['coverage']:.2%}, with mean full width {results['error_range']['mean_width_m']:.3f} m. Coverage is not guaranteed in future weather regimes. The range can span warning/danger thresholds; the UI explicitly flags uncertain status.

## Dashboard and saved artifacts

The main app has only River forecast and Rainfall history. Gauge/year/date controls, past-observation example presets, numeric peak and status, a forecast range, interactive level chart, optional recorded outcomes and CSV export work locally. The forecast line represents a window maximum, not an invented hourly trajectory. Archived dates are explicit. The former annual flood-risk UI/model and prior experiments remain preserved outside the primary navigation.

Saved artifacts include the merged dataset, fixed split manifest, acquisition/quality audits, all 22 pipelines, validation/test predictions, selection, model metadata and an inference bundle. Frozen source, split, original-model and bundle hashes verify; all 1,156 held-out predictions replay within numerical precision. Nine truncated-history runtime checks confirm future observations are unnecessary.

Technical verification: **24 unit tests passed; {ui['passed']} dashboard interaction checks passed; {ui['failed']} failed**. These checks establish runtime/causality behaviour, not predictive reliability. Earlier annual-model/API evaluations are separate historical work and do not describe this app.

## Sources and detailed evidence

- [CWC manual hourly observations via NWDP](https://nwdp.nwic.gov.in/dataset/river-water-level-manual-hourly-cwc-subernarekha); Krishna resource IDs are retained in `data/river_stage/raw/source_manifest.json`.
- [CWC April 2025 SOP, page 46](https://cwc.gov.in/sites/default/files/sopapril2025.pdf#page=46), official warning/danger levels.
- [Full candidate and station results](experiments/river_stage_v1/MODEL_REPORT.md), [frozen numeric results](experiments/river_stage_v1/final_results.json), [episode/replay audit](experiments/river_stage_v1/verification.json).
- [Previous model report and annual-flow experiments](legacy/MODEL_REPORT_before_river_forecast.md) preserve earlier limitations and source attribution.
'''
    (ROOT/'MODEL_REPORT.md').write_text(report,encoding='utf-8')
    detailed=EXPERIMENT/'MODEL_REPORT.md'
    marker='## Final interpretation and deployment review'
    existing=detailed.read_text(encoding='utf-8').split(marker)[0].rstrip()
    detailed.write_text(existing+'\n\n'+marker+'\n\n'
        'The frozen ML winner does not establish superiority: trend/persistence have lower test MAE; '
        'trend also has better warning-period MAE and danger-onset recall. Only two observed danger '
        'episodes exist in the audited test archive, with one predicted in advance and one missed. '
        'Keep this as a historical demonstration. See the [current project report](../../MODEL_REPORT.md) '
        'for coverage, episode-level interpretation and completed dashboard verification.\n',encoding='utf-8')
    (ROOT/'README.md').write_text('''# RiskRadar · Maharashtra River-Level Forecast

A controlled dashboard for next-24-hour river-peak forecasting and historical rainfall analysis. Three gauges: Kurundwad, Narsingpur and Takli. Archived CWC observations drive a newly trained numeric XGBoost regressor; official station thresholds derive warning/danger status.

**This version replays historical forecasts; it does not fetch live 2026 readings or issue emergency alerts.** It predicts gauge stage, not district inundation or flood damage.

## Run locally

From this directory in PowerShell:

```powershell
.\\.venv\\Scripts\\python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

Open [RiskRadar](http://127.0.0.1:8501/). Alternatively, `.\\start-riskradar.ps1` launches the dashboard in a hidden background process and records logs/PID under `.run/`. The primary app does not require the old FastAPI service.

## Use

- **River forecast:** choose a gauge, year and available date. Inspect the forecast peak, stage, range and past readings. Example buttons use only observations available at the issue time. Optionally reveal the recorded outcome or download the forecast.
- **Rainfall history:** explore yearly totals/anomalies, monthly heatmaps and daily rainfall. Existing cached history works offline; an uncached location requires downloading its historical archive.

## ML lifecycle and honest results

Train 2014–2020, select 2021, calibrate ranges 2022, test 2023–2025. Compare 22 ML candidates across five algorithms and three simple baselines. Freeze XGBoost using validation alone. Final test: 1,156 forecasts; MAE **0.258 m**, RMSE **0.517 m**. **Trend baseline MAE is better (0.205 m)**; ML reduces RMSE but does not demonstrate overall superiority. There are only two observed danger episodes, of which one was predicted in advance. This is insufficient for an operational warning claim.

Read [MODEL_REPORT](MODEL_REPORT.md) for full results, [REVIEW](REVIEW.md) for limitations, and [LOCAL_SETUP](LOCAL_SETUP.md) for commands and files.

## Architecture

`CWC archive → conservative QC/hourly bins → causal trailing features → frozen regressor → numeric 24-hour peak → official stage rules → Streamlit/Plotly`

`river_forecast.py` shares feature and inference logic with training. `river_page.py` contains the user interface; `historical_page.py` retains rainfall analysis. Source manifests, merged data, splits and model artifacts are on disk under `data/river_stage/` and `experiments/river_stage_v1/`.

## History and provenance

The original repository is [nitinc264/Flood-prediction-Model](https://github.com/nitinc264/Flood-prediction-Model). Its former annual-risk dashboard is preserved in `legacy/annual_risk_dashboard.py`, and its original model remains unchanged. Prior nationwide/annual-flow studies and original documentation are retained under `experiments/` and [legacy](legacy/README.md); their claims do not describe the new river-level model.
''',encoding='utf-8')
    (ROOT/'LOCAL_SETUP.md').write_text('''# Local setup · current river dashboard

Run from `D:\\ADS_MINI\\Flood-risk-prediction-system\\RiskRadar`.

```powershell
.\\.venv\\Scripts\\python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

Dashboard: http://127.0.0.1:8501/. Only Streamlit is required; the former API on port 8000 is not used. `.\\start-riskradar.ps1` starts a hidden local process, refuses an occupied port and saves `.run/processes.json` and logs. Do not stop unrelated processes to free a port.

For a fresh Python 3.12 environment, install `requirements-river.txt`. It pins the direct packages used by this version. `requirements-local.lock.txt` preserves the earlier complete environment resolution; sampling/CatBoost dependencies in `requirements-training.txt` belong to older studies and are not required by the new regressor.

The existing saved artifacts load without downloading or retraining. Forecast dates are historical, selected from eligible 2023–2025 windows. Rainfall history uses its existing catalog and weather cache; other locations may need a historical archive download. New river forecasts never call the old annual-risk API.

## Verify

```powershell
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv\\Scripts\\python.exe scripts/verify_river_stage.py
.\\.venv\\Scripts\\python.exe scripts/verify_river_dashboard.py
```

Streamlit interaction checks need local loopback networking on Windows. The first command currently passes 24 tests, the third 27 interaction checks. `tests/verify_dashboard.py` targets the archived annual-risk app and is retained as historical verification, not a verifier for this entrypoint.

## Reproduce the experiment

`scripts/fetch_river_stage.py` acquires only the selected three gauges; manifests and checksums preserve provenance. `scripts/train_river_stage.py prepare`, `select`, and `evaluate` are sequential stages. They refuse to overwrite existing frozen protocol, selection or final results. Do not remove the guard or select/tune again using final test results. A future study needs a new prespecified version and fresh evaluation period.

`scripts/document_river_stage.py` rebuilds current Markdown documentation from frozen results without fitting or selecting models. Runtime package versions are saved under `experiments/river_stage_v1/runtime_versions.json`.

## Important files

- `river_forecast.py`: conservative QC, causal features, numeric prediction and stage rules.
- `river_page.py`, `app.py`, `assets/dashboard.css`: current two-page dashboard.
- `data/river_stage/processed/model_dataset.csv`: the merged, eligible modelling table.
- `experiments/river_stage_v1/forecast_bundle.joblib`: selected pipeline, feature list, station thresholds and calibrated error ranges.
- `experiments/river_stage_v1/selection.json`, `final_results.json`, `test_predictions.csv`: frozen evidence.
- `legacy/annual_risk_dashboard.py`: source snapshot of the preceding app. Its paths assume the original root; restore to a separate root before attempting to run it.

Original setup instructions and old API details are retained in `legacy/LOCAL_SETUP_before_river_forecast.md`.
''',encoding='utf-8')
    (ROOT/'REVIEW.md').write_text('''# Current implementation review

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
''',encoding='utf-8')
    (ROOT/'EVALUATION_RESULTS.md').write_text(f'''# Current river-dashboard validation

This describes the river-level release, not the archived annual-risk API. Earlier findings are retained in `legacy/EVALUATION_RESULTS_before_river_forecast.md`.

- 24 unit tests passed, zero failed, using `python -m unittest discover -s tests -v`.
- {ui['passed']} Streamlit interaction checks passed, {ui['failed']} failed: all three gauges across 2023/2024/2025, all observation-based presets, recorded outcomes, exports, rainfall views/controls and navigation back to forecasts.
- Frozen source/protocol/selection/split/model checksums match; original annual model remains unchanged.
- All {verified['test_rows_replayed']} test predictions replay; largest difference {verified['largest_prediction_difference_m']:.3g} m.
- {verified['runtime_prefix_replays']} runtime checks confirm identical predictions with future observations removed.
- Actual browser inspection confirms the two-page dashboard and interactive forecast/actual chart render on localhost.

These are implementation checks, not measures of forecasting skill. Actual chronological metrics and baseline/episode limitations are in [MODEL_REPORT](MODEL_REPORT.md). No new validation of the archived API is claimed.
''',encoding='utf-8')
    (ROOT/'legacy/README.md').write_text('''# Preserved project history

The current app offers **River forecast** and **Rainfall history**. These files preserve preceding source and evidence; they are not imported or routed by the new forecast page.

- `annual_risk_dashboard.py`: exact source snapshot of the preceding annual-risk app. It expects root-relative assets, so restore it into a separate checkout root to run it.
- `risk_map.py`, `dashboard.css`: previous map renderer/style. `../high_risk_locations.csv` preserves its unchanged unverified predictions.
- `README_before_river_forecast.md`, `MODEL_REPORT_before_river_forecast.md`, `LOCAL_SETUP_before_river_forecast.md`, `EVALUATION_RESULTS_before_river_forecast.md`: previous documentation and experimental findings.
- The unchanged original `../flood_model.pkl`, `../api.py` and earlier `../experiments/` remain on disk for provenance and research history. The primary application does not call the original model/API.

See the root README and MODEL_REPORT for current behaviour and evidence.
''',encoding='utf-8')
    packages=['streamlit','plotly','requests','pandas','numpy','scikit-learn','xgboost','joblib','scipy','threadpoolctl']
    versions={package:version(package) for package in packages}
    (EXPERIMENT/'runtime_versions.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    (ROOT/'requirements-river.txt').write_text('# Current river dashboard and regression experiment; Python 3.12\n'+
        '\n'.join(f'{package}=={value}' for package,value in versions.items())+'\n',encoding='utf-8')
    print('Published current documentation; frozen model, selection and test files unchanged.')


if __name__=='__main__':main()
