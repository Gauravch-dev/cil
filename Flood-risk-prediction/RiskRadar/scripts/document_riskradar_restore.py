"""Record exactly which original predictor is now connected to the public UI."""
from pathlib import Path
from hashlib import sha256
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    checks=json.loads((ROOT/'evaluation/two_page_dashboard_verification.json').read_text())
    original='aad88a12f4645f3703c996641a15ce40603f6d5a608918fad4f37496d2f89395'
    assert sha256((ROOT/'flood_model.pkl').read_bytes()).hexdigest()==original
    for name in ['README.md','MODEL_REPORT.md','REVIEW.md','LOCAL_SETUP.md','EVALUATION_RESULTS.md']:
        source=ROOT/name;archive=ROOT/'legacy'/f'{source.stem}_before_riskradar_restore.md'
        if source.exists() and not archive.exists():archive.write_bytes(source.read_bytes())
    readme='''# Rainfall and Flood Risk Analysis Using Historical Weather Data

The public application has exactly **two pages: Rainfall and Flood**.

- **Rainfall:** saved historical rainfall near Mumbai, a period selector and one chart at a time with Daily, Monthly, Yearly and Monsoon views; unusually wet days/months and CSV download.
- **Flood:** the original RiskRadar-style annual rainfall, climate and catchment/land/soil inputs. Typical, Drier and Wetter examples; editable area details; Low/Medium/High result; a visual category gauge and CSV download.

The Flood page uses the original saved RiskRadar predictor locally, with the original 13-feature equations, categorical encoder order and risk bands. There are no Kurundwad river-elevation or next-24-hour inputs on this page. The former city classifier is preserved in experiments/kurundwad_flood_risk_v1 but is no longer the public predictor.

The screen is an **annual area-condition susceptibility indicator**, not a time-specific flood alert. It does not show model/evaluation panels or label its internal score as a calibrated flood probability. Technical evidence stays in MODEL_REPORT.md and EVALUATION_RESULTS.md.

## Inputs and behaviour

Three main inputs are annual rainfall, wettest-month rainfall and rainfall variation. Land & weather details contains mean temperature, temperature variation, runoff potential, drainage density/texture, elevation difference, ruggedness, infiltration, climate type, land cover and soil class. All six climate, eleven land-cover and twenty-five soil categories are readable and available.

Rainfall fields accept nonnegative values without the old arbitrary slider maxima. Validation rejects missing/nonfinite/negative measurements, unknown categories, runoff outside 0–100 and inconsistent rainfall totals. A wettest month cannot exceed its year, and annual total cannot exceed twelve wettest months. Original feature-engineering caps are preserved where the fitted predictor requires them; raw annual rainfall is not clipped. Invalid or changed inputs never show a stale assessment. Presets and manual inputs persist across page navigation.

## Run

```powershell
.\\.venv\\Scripts\\python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

Open [RiskRadar](http://127.0.0.1:8501/). The optional start-riskradar.ps1 launcher starts the same app in the background. No separate API service or live data feed is required for either public page. The original api.py is preserved for reference.

## Main files

monitor_page.py is the two-page shell. rainfall_page.py renders historical analysis. riskradar_page.py renders the restored assessment. riskradar_predictor.py validates inputs and calls the original flood_model.pkl, encoders.pkl and feature_columns.pkl. tests/test_riskradar_predictor.py compares the new path against the independent original api.py. scripts/verify_two_page_dashboard.py checks real chart values, all risk categories, edits, invalid inputs and navigation.

Original source: [nitinc264/Flood-prediction-Model](https://github.com/nitinc264/Flood-prediction-Model). Dataset/target reconstruction: evaluation/upstream_forensics/PROVENANCE_FINDINGS.md. Older models, sources, reports and interfaces are preserved in history. The Word report under reports describes an older release and has not been updated for this restoration.
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    (ROOT/'MODEL_REPORT.md').write_text('''# Restored original RiskRadar predictor

## Current active model and target

The public Flood page again uses the original saved XGBoost predictor with original encoders and 13 ordered features. User authorization specifically requested original RiskRadar-style rainfall/climate/land inputs instead of the Kurundwad river-level form. No model was retrained and no frozen experiment was overwritten. CPU prediction is selected in memory only; the model artifact hash is unchanged.

Existing forensic evidence strongly corroborates the supplied GloFlo dataset (45,932 rows, 93 columns) and a reconstructed binary target of annual peak discharge above the full-source 95th percentile (1145.198 in source units). This is a relative annual high-discharge proxy, not observed flood/no-flood occurrence, inundation or a next-day warning. Public wording uses an annual susceptibility indicator; internal scores are not calibrated flood probabilities.

## Preserved feature contract

Fourteen raw inputs create thirteen fitted features. Precipitation index = 0.4 min(annual rainfall,3000) + 0.3 min(wettest-month rainfall,1000) + 0.3 min(rainfall CV,100). Drainage index = 0.5 min(density,10) + 0.3 min(texture,50) + 0.2 min(relief,5000). Remaining features: mean temperature, temperature seasonality, raw annual precipitation, curve number AMCII, drainage density, basin relief, ruggedness, infiltration and original encoded climate/land-cover/soil classes. feature_columns.pkl controls order.

Original risk bands are retained: Low score <=0.4, Medium >0.4 to <=0.7, High >0.7. These are the supplied application's display bands, not newly validated physical warning thresholds. No thresholds were changed to make demonstrations succeed. The saved predictor is nonmonotonic; a larger annual rainfall value alone need not always increase its score, and values beyond learned tree split ranges can produce unchanged scores. No forced monotonicity or artificial output inflation is added.

## Original evaluation evidence and separation from retired studies

The reconstructed original random holdout has 9,187 rows and 459 positive high-discharge labels. Artifact replay gives accuracy 96.8869%, positive-class precision 65.6420%, recall 79.0850% and F1 71.7391%. This is a corroborated reconstruction on previously inspected data, not a new independent chronological evaluation. Repeated stations span the inferred split, and the label percentile was inferred from the full source. The public upstream F1 0.984 matches the negative class and must not be presented as positive-class F1.

The 99.29% accuracy / 89.66% F1 Kurundwad values belong to the retired city's different warning-stage target. They are not the current predictor's results. Its models and detailed evaluation remain in experiments/kurundwad_flood_risk_v1. No claim is made that the restored original model meets 75% on every positive-class metric.

## Behaviour verification

Default, drier and wetter examples exactly reproduce original scores 0.0779/Low, 0.0038/Low and 0.7840/High. A fixed example with annual rainfall 2200 and wettest-month rain 550, other default inputs, gives 0.6520/Medium. Forty additional seeded input combinations compare both all thirteen feature values and rounded outputs against the independent original api.py. All category codes, rainfall above old maxima, invalid inputs, band boundaries and page state changes are tested. These are software/behaviour checks, not new predictive accuracy evidence.

Detailed provenance: evaluation/upstream_forensics/PROVENANCE_FINDINGS.md and reconstructed_training_evidence.json. Original model SHA256: '''+original+'\n',encoding='utf-8')
    (ROOT/'LOCAL_SETUP.md').write_text('''# Run the two-page RiskRadar dashboard

```powershell
.\\.venv\\Scripts\\python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

Open http://127.0.0.1:8501/. Rainfall opens by default. Flood accepts original annual rainfall, climate and land/soil conditions and calls the supplied saved predictor locally; it does not need a second API service. Use Typical/Drier/Wetter examples or edit measured area inputs. No river-gauge elevation or observation date is required. The saved model and source data are preserved. Runtime inference selects CPU without rewriting model files.
''',encoding='utf-8')
    (ROOT/'REVIEW.md').write_text('''# Current interface correction

Restored original RiskRadar inputs and inference after explicit user clarification. The previous station-level river-warning form solved a different task and is removed from public navigation, with its code/models retained in history. A tested scenario with river level 537.2 m and rainfall 500 mm returned Lower from that former classifier, demonstrating unhelpful behaviour outside its sparse measured rainfall coverage. No retired thresholds or test outcomes were altered to hide that result.

The current two-page interface preserves historical rainfall analysis and provides the original Low/Medium/High annual-condition workflow. Validation, complete category choices, default/preset parity, invalid-input rejection and stale-result handling are verified. The saved original model is reused honestly; current documentation does not attribute the retired city's metrics to it. See MODEL_REPORT.md for actual target and original evaluation evidence. No ML metrics are displayed on the public interface.
''',encoding='utf-8')
    evaluation={'unit_tests_passed':63,'unit_tests_failed':0,'dashboard_checks':checks,
        'original_model_sha256':original,'model_unchanged':True,'training_performed':False,
        'default':{'score':.0779,'risk':'LOW'},'drier':{'score':.0038,'risk':'LOW'},
        'medium_example':{'score':.652,'risk':'MEDIUM'},'wetter':{'score':.784,'risk':'HIGH'},
        'independent_original_api_feature_and_score_parity_cases':40,
        'new_predictive_accuracy_evaluation':False}
    (ROOT/'evaluation/riskradar_restore_verification.json').write_text(json.dumps(evaluation,indent=2),encoding='utf-8')
    (ROOT/'EVALUATION_RESULTS.md').write_text(f'''# Original RiskRadar restoration verification

- 63 unit tests passed, 0 failed.
- {checks['passed']} current two-page dashboard checks passed, {checks['failed']} failed.
- Forty seeded valid scenarios match the original api.py feature arrays and rounded results exactly.
- Canonical original examples: Typical 0.0779/Low; Drier 0.0038/Low; Wetter 0.7840/High. A fixed medium case returns 0.6520/Medium.
- Full original category vocabularies verified: 6 climate, 11 land-cover, 25 soil choices.
- Invalid rainfall totals, missing/nonfinite values, unknown categories, runoff bounds, zero/fractional rain, original band boundaries, preset changes, input persistence and stale-result removal checked.
- Both public pages work without a separate API or network data fetch. No model/evaluation panels appear in the UI.
- The supplied model SHA256 is unchanged: {original}.

Evidence: evaluation/riskradar_restore_verification.json and evaluation/two_page_dashboard_verification.json. These checks verify application behaviour and original-predictor parity, not new forecasting accuracy. The earlier Kurundwad and other experiment scores remain separate in their original folders.
''',encoding='utf-8')
    print('Original RiskRadar restoration documented; model and past experiments preserved.')


if __name__=='__main__':main()
