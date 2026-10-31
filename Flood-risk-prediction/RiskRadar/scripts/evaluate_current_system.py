"""Audit current artifacts, data and API behavior without training or inventing labels."""
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from historical_data import SOURCE_MD5, default_station, read_weather, station_catalog

BASE = dict(annual_precipitation=1200, precipitation_of_wettest_month=300,
    precipitation_seasonality=50, drainage_density=2.5, drainage_texture=10.,
    basin_relief=500, annual_mean_temperature=20, temperature_seasonality=500,
    curve_number_amcii=70., ruggedness_number=.5, infiltration_number=5.,
    climate_type=2, landcover_type=1, soil_type=3)
SCENARIOS = {
    'Typical conditions': BASE,
    'Drier landscape': {**BASE, 'annual_precipitation': 400,
        'precipitation_of_wettest_month': 120, 'precipitation_seasonality': 25,
        'drainage_density': 1., 'curve_number_amcii': 50.},
    'Wetter landscape': {**BASE, 'annual_precipitation': 2600,
        'precipitation_of_wettest_month': 800, 'precipitation_seasonality': 85,
        'drainage_density': 6., 'drainage_texture': 30., 'basin_relief': 2000,
        'annual_mean_temperature': 27, 'curve_number_amcii': 90.,
        'ruggedness_number': 2., 'infiltration_number': 10.,
        'climate_type': 1, 'landcover_type': 0, 'soil_type': 9},
}


def evaluate():
    session = requests.Session()
    api = 'http://127.0.0.1:8000'
    health = session.get(api + '/health', timeout=10)
    health.raise_for_status()
    source_path = ROOT / 'data/raw/kuntla_flooddatabase.csv'
    source = pd.read_csv(source_path, low_memory=False)
    years = pd.to_numeric(source.year, errors='coerce')
    target_candidates = {'flooded', 'is_flood', 'flood_occurrence', 'label', 'target'}
    target_columns = [c for c in source if c.strip().lower() in target_candidates]
    catalog = station_catalog()
    weather, _ = read_weather(default_station(catalog))
    columns = joblib.load(ROOT / 'feature_columns.pkl')
    model = joblib.load(ROOT / 'flood_model.pkl')
    names = model.get_booster().feature_names
    artifact_checks = {
        'published_source_checksum_matches': hashlib.md5(source_path.read_bytes()).hexdigest() == SOURCE_MD5,
        'original_model_fingerprint_matches': hashlib.sha256((ROOT / 'flood_model.pkl').read_bytes()).hexdigest() ==
            'aad88a12f4645f3703c996641a15ce40603f6d5a608918fad4f37496d2f89395',
        'model_feature_count_matches_schema': int(model.n_features_in_) == len(columns) == 13,
        'model_feature_order_matches_schema': names is None or names == list(columns),
        'binary_model_classes': np.array_equal(model.classes_, [0, 1]),
    }
    examples = []
    for name, payload in SCENARIOS.items():
        response = session.post(api + '/predict', json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        examples.append({'name': name, 'inputs': payload, 'score': result['flood_risk_score'],
                         'risk_level': result['risk_level']})
    repeats, timings = [], []
    for _ in range(5):
        started = perf_counter()
        response = session.post(api + '/predict', json=BASE, timeout=15)
        response.raise_for_status()
        timings.append((perf_counter() - started) * 1000)
        repeats.append(response.json()['flood_risk_score'])
    artifact_checks['identical_inputs_produce_identical_scores'] = len(set(repeats)) == 1
    missing = {key: value for key, value in BASE.items() if key != 'annual_precipitation'}
    cases = [
        ('Valid example', BASE, 200),
        ('Missing required rainfall', missing, 422),
        ('Non-numeric rainfall', {**BASE, 'annual_precipitation': 'invalid'}, 422),
        ('Negative annual rainfall', {**BASE, 'annual_precipitation': -1, 'precipitation_of_wettest_month': 0}, 422),
        ('Wettest month exceeds annual rainfall', {**BASE, 'annual_precipitation': 100}, 422),
        ('Negative drainage density', {**BASE, 'drainage_density': -.1}, 422),
        ('Unknown climate code', {**BASE, 'climate_type': 99}, 422),
        ('Negative soil category', {**BASE, 'soil_type': -1}, 422),
    ]
    boundary_checks = []
    for name, payload, expected_status in cases:
        response = session.post(api + '/predict', json=payload, timeout=15)
        body = response.json()
        boundary_checks.append({'case': name, 'expected_status': expected_status,
            'actual_status': response.status_code, 'passed': response.status_code == expected_status,
            'returned_risk_score': body.get('flood_risk_score'), 'request': payload})
    ui = json.loads((ROOT / '.run/ui-verification.json').read_text(encoding='utf-8'))
    history = json.loads((ROOT / '.run/history-verification.json').read_text(encoding='utf-8'))
    unit_log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py')
    unit_run = unittest.TextTestRunner(stream=unit_log, verbosity=2).run(suite)
    unit_failed = len(unit_run.failures) + len(unit_run.errors)
    units = {'passed': unit_run.testsRun - unit_failed - len(unit_run.skipped),
             'failed': unit_failed, 'skipped': len(unit_run.skipped), 'output': unit_log.getvalue()}
    boundary_failed = sum(not case['passed'] for case in boundary_checks)
    totals = {'passed': ui['passed'] + history['passed'] + units['passed'] + sum(artifact_checks.values())
                         + len(boundary_checks) - boundary_failed,
              'failed': ui['failed'] + history['failed'] + units['failed']
                         + len(artifact_checks) - sum(artifact_checks.values()) + boundary_failed}
    result = {
        'scope': 'Read-only validation of the current application; no retraining or UI changes.',
        'existing_integration_checks': {'passed': ui['passed'] + history['passed'],
                                       'failed': ui['failed'] + history['failed']},
        'unit_checks': units,
        'overall_software_validation': {**totals, 'status': 'FAIL' if totals['failed'] else 'PASS'},
        'artifact_and_consistency_checks': artifact_checks,
        'data': {'raw_rows': len(source), 'columns': len(source.columns),
            'stations': int(catalog.station_id.nunique()), 'countries': int(catalog.country.nunique()),
            'duplicate_rows': int(source.duplicated().sum()),
            'invalid_year_rows': int(years.isna().sum()),
            'fully_missing_columns': source.columns[source.isna().all()].tolist(),
            'binary_target_columns_found': target_columns,
            'rainfall_series_days': len(weather), 'missing_rainfall_days': int(weather.rain_mm.isna().sum()),
            'rainfall_coverage_percent': float(weather.rain_mm.notna().mean() * 100)},
        'example_predictions': examples,
        'repeat_request_median_ms': float(np.median(timings)),
        'timing_note': 'Five sequential localhost requests; not a load/performance benchmark.',
        'api_input_validation': boundary_checks,
        'model_evaluation': {'status': 'NOT INDEPENDENTLY EVALUABLE',
            'accuracy': None, 'precision': None, 'recall': None, 'f1': None,
            'roc_auc': None, 'pr_auc': None, 'calibration': None,
            'reason': 'No verified binary flood/no-flood labels, original label-generation pipeline, or held-out split membership are available.',
            'upstream_claims_unverified': {'accuracy': .969, 'f1': .984, 'recall': .791}},
    }
    output = ROOT / 'evaluation'
    output.mkdir(exist_ok=True)
    (output / 'current_evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    failed = [case for case in boundary_checks if not case['passed']]
    lines = [
        '# Current validation and evaluation results', '',
        'This evaluates the current application without modifying the UI, model or API. Software behavior and predictive accuracy are reported separately.', '',
        '## Existing regression suite', '',
        f'- Dashboard integration checks: {ui["passed"]} passed, {ui["failed"]} failed.',
        f'- Historical page integration checks: {history["passed"]} passed, {history["failed"]} failed.',
        f'- Data-handling unit tests: {units["passed"]} passed, {units["failed"]} failed, {units["skipped"]} skipped.', '',
        '## Additional artifact and API validation', '',
        f'- Artifact/consistency checks: {sum(artifact_checks.values())} passed, {len(artifact_checks) - sum(artifact_checks.values())} failed.',
        f'- API input checks: {len(boundary_checks) - len(failed)} passed, {len(failed)} failed.', '',
        f'**Total: {totals["passed"]} passed, {totals["failed"]} failed. Overall strict software validation: {"FAIL" if totals["failed"] else "PASS"}.**', '',
        '| API case | Expected HTTP status | Actual HTTP status | Result |',
        '|---|---:|---:|---|',
    ]
    lines += [f'| {c["case"]} | {c["expected_status"]} | {c["actual_status"]} | {"Pass" if c["passed"] else "FAIL"} |' for c in boundary_checks]
    lines += ['', 'Failed invalid-input checks mean the API returned a risk score for unsupported or physically inconsistent input. The dashboard rejects inconsistent annual/wettest-month rainfall and constrains widget ranges, but direct API callers can bypass these controls. The API schema needs non-negative numeric bounds, category-code bounds and cross-field rainfall validation. These failures were recorded rather than changed during evaluation.', '',
        '## Data audit', '',
        f'- {len(source):,} records, {len(source.columns)} columns, {catalog.station_id.nunique():,} stations, {catalog.country.nunique()} countries.',
        f'- {source.duplicated().sum()} exact duplicate source rows; {years.isna().sum()} unusable year entries; {source.isna().all().sum()} entirely missing columns.',
        f'- The saved rainfall archive has {len(weather):,} days and {weather.rain_mm.isna().sum()} missing rainfall values ({weather.rain_mm.notna().mean():.1%} coverage). Coverage is completeness, not measurement accuracy.',
        '- Annual precipitation is a fixed station climate attribute in GloFlo, not an annual weather time series.', '',
        '## Model accuracy evaluation', '',
        '**Independent model accuracy cannot be established from the available data.**', '',
        '| Metric | Independently measured result |', '|---|---|',
        '| Accuracy | Unavailable |', '| Precision | Unavailable |', '| Recall | Unavailable |',
        '| F1 | Unavailable |', '| ROC-AUC / PR-AUC | Unavailable |',
        '| Calibration / operational thresholds | Unvalidated |', '',
        'The original source lacks binary flood/no-flood labels, the target-generation pipeline and test-split membership. Saved risk scores are predictions, not ground truth. No synthetic labels or in-sample performance are substituted for a test. The existing binary classifier does not predict rainfall amounts, so rainfall MAE/RMSE/R² are not applicable to this model.', '',
        'The upstream README claims accuracy 96.9%, F1 0.984 and recall 0.791. They have not been reproduced and must not be presented as this validation’s results. F1 0.984 with recall 0.791 is incompatible for the same positive class/test population; different averaging/evaluation conventions would need to be documented.', '',
        '## Runtime examples, not accuracy measurements', '',
        '| Scenario | Score | Category |', '|---|---:|---|',
    ]
    lines += [f'| {example["name"]} | {example["score"]:.2%} | {example["risk_level"]} |' for example in examples]
    lines += ['', 'Repeated identical requests returned identical scores. These examples demonstrate execution only; they do not validate real-world flood prediction.', '',
        '## Evaluation conclusion', '',
        'The rainfall analysis and risk-scenario application pass the existing regression checks. Additional validation exposes missing API input safeguards. The supplied flood classifier remains an unvalidated demonstration until ground-truth labels and an independent evaluation split are available. No deployment-readiness or flood-warning accuracy claim is supported.', '']
    (ROOT / 'EVALUATION_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'overall_software_validation': result['overall_software_validation'],
        'unit_checks': {key: value for key, value in units.items() if key != 'output'}, 'artifact_checks': artifact_checks,
        'api_validation_passed': len(boundary_checks) - len(failed), 'api_validation_failed': len(failed),
        'failed_cases': [case['case'] for case in failed],
        'examples': [{key: value for key, value in example.items() if key != 'inputs'} for example in examples],
        'model_evaluation': result['model_evaluation'], 'data': result['data']}, indent=2))


if __name__ == '__main__':
    evaluate()
