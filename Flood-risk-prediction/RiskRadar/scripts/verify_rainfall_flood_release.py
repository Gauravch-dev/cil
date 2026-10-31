"""Verify joins, past rainfall, frozen predictions and the scope of this release."""
from pathlib import Path
import sys
import json

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from rainfall_flood import (ROOT,SOURCE,DATA,RELEASE,RAIN,MODELS,digest,load_events,load_bundle,
                            daily_rainfall,severity_estimates)


def main():
    manifest=json.loads((RELEASE/'manifest.json').read_text())
    analysis=json.loads((DATA/'manifest.json').read_text())
    assert digest(DATA/'manifest.json')==manifest['analysis_manifest_sha256']
    assert digest(DATA/'matched_events.csv')==analysis['dataset_sha256']
    assert digest(DATA/'flood_inventory.csv')==analysis['inventory_sha256']
    assert digest(RELEASE/'severity_comparison.joblib')==manifest['bundle_sha256']
    assert digest(SOURCE/'selection.json')==manifest['source_selection_sha256']
    assert digest(SOURCE/'final_results.json')==manifest['source_results_sha256']
    assert digest(ROOT/'flood_model.pkl')==manifest['original_model_sha256']
    assert digest(ROOT/'experiments/river_stage_v1/forecast_bundle.joblib')==manifest['previous_river_bundle_sha256']
    for name,value in analysis['raw_source_sha256'].items():assert digest(SOURCE/'raw'/name)==value
    frame=load_events();bundle=load_bundle()
    events=pd.read_csv(SOURCE/'raw/floodevents_indofloods.csv').set_index('EventID')
    rain=pd.read_csv(SOURCE/'raw/precipitation_variables_indofloods.csv').set_index('EventID')
    assert frame.EventID.is_unique and len(frame)==4493
    assert frame['Start Date'].dt.year.max()==2019
    assert not analysis['no_flood_labels_available'] and not analysis['unknown_days_relabelled_as_non_flood']
    assert frame['Flood Type'].eq(frame.EventID.map(events['Flood Type'])).all()
    assert frame.target.eq(frame['Flood Type'].eq('Severe Flood').astype(int)).all()
    for key in RAIN:np.testing.assert_allclose(frame[key],frame.EventID.map(rain[key]),rtol=0,atol=1e-10)
    daily=frame[[f'rain_day_minus_{i}_mm' for i in range(1,11)]].to_numpy()
    np.testing.assert_allclose(np.cumsum(daily,axis=1),frame[RAIN].to_numpy(),atol=1e-9)
    forbidden={'EventID','GaugeID','target','Flood Type','Peak Flood Level (m)','Peak FL Date','End Date',
               'Peak Discharge Q (cumec)','Warning Level','Danger Level','Event Duration (days)','split'}
    for features in bundle['features'].values():assert not set(features)&forbidden
    test=frame.loc[frame.split.eq('test')].copy()
    recorded=pd.read_csv(SOURCE/'test_predictions.csv').set_index('EventID').loc[test.EventID]
    maximum=0.;runtime=0
    for name in MODELS:
        scores=bundle['models'][name].predict_proba(test[bundle['features'][name]])[:,1]
        error=float(np.max(np.abs(scores-recorded[f'{name}_score'].to_numpy())))
        assert error<1e-6,(name,error)
        maximum=max(maximum,error)
        np.testing.assert_array_equal(scores>=bundle['thresholds'][name],recorded[f'{name}_prediction'])
    # Replaying an event does not need the realised peak or true severity.
    for position in [0,len(test)//2,len(test)-1]:
        row=test.iloc[position].copy();first=severity_estimates(row,bundle)
        row['Peak Flood Level (m)']=99999;row['Flood Type']='CHANGED';row['target']=-1
        assert first==severity_estimates(row,bundle)
        assert daily_rainfall(row).date.max()<row['Start Date'];runtime+=1
    report={'artifact_and_source_checks':'passed','matched_event_rows':len(frame),
            'heldout_events_replayed':len(test),'model_score_replays':len(test)*len(MODELS),
            'maximum_score_difference':maximum,'outcome_mutation_runtime_checks':runtime,
            'verified_non_flood_examples':0,'non_flood_examples_invented':0,
            'model_selection_changed':False,'new_independent_test_claimed':False}
    (RELEASE/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
