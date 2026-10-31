"""Replay frozen model artifacts and audit held-out danger-episode coverage."""
from pathlib import Path
import sys
import json
from hashlib import sha256

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from river_forecast import (ROOT,DATA,EXPERIMENT,STATIONS,FEATURES,causal_features,
                            load_hourly,load_bundle,forecast_at)

def digest(path):return sha256(path.read_bytes()).hexdigest()

def main():
    p=json.loads((EXPERIMENT/'protocol.json').read_text())
    s=json.loads((EXPERIMENT/'selection.json').read_text())
    r=json.loads((EXPERIMENT/'final_results.json').read_text())
    assert digest(EXPERIMENT/'protocol.json')==s['protocol_sha256']
    assert digest(EXPERIMENT/'selection.json')==r['selection_sha256']
    assert digest(EXPERIMENT/'forecast_bundle.joblib')==r['bundle_sha256']
    assert digest(ROOT/'flood_model.pkl')==p['original_model_sha256']
    assert digest(DATA/'raw/source_manifest.json')==p['source_manifest_sha256']
    manifest=json.loads((DATA/'raw/source_manifest.json').read_text())
    for resource in manifest['resources']:
        assert digest(DATA/'raw'/resource['file'])==resource['sha256']
    for split,checksum in p['split_hashes'].items():assert digest(EXPERIMENT/f'{split}.csv')==checksum
    for station,checksum in p['hourly_hashes'].items():assert digest(DATA/f'processed/{station}_hourly.csv')==checksum
    frames={split:pd.read_csv(EXPERIMENT/f'{split}.csv',parse_dates=['time','label_end'])
            for split in ['train','validation','calibration','test']}
    periods=list(frames)
    for left,right in zip(periods,periods[1:]):
        assert frames[left].label_end.max()<frames[right].time.min()
    assert not set(FEATURES)&{'target_peak','target_change','label_end','observed_max','future_observed_hours'}
    recorded=pd.read_csv(EXPERIMENT/'test_predictions.csv',parse_dates=['time','label_end'])
    test=frames['test'];bundle=load_bundle();selected=s['selected']['name']
    max_error=0.;episodes=[];replay_rows=0
    for station in STATIONS:
        hourly=load_hourly(station)
        subset=test.loc[test.station.eq(station)]
        bank=causal_features(hourly,station).loc[subset.time]
        if bundle['predictor_type']!='ml':raise RuntimeError('Adjust verification explicitly for selected baseline.')
        batch=bank.current_level.to_numpy()+bundle['pipeline'].predict(bank[FEATURES])
        expected=recorded.loc[recorded.station.eq(station),selected].to_numpy()
        error=float(np.max(np.abs(batch-expected)));max_error=max(max_error,error)
        assert error<1e-5,(station,error)
        for pos in [0,len(subset)//2,len(subset)-1]:
            time=subset.iloc[pos].time
            forecast=forecast_at(station,time,bundle=bundle,hourly=hourly)
            assert abs(forecast['predicted_peak']-batch[pos])<1e-5
            # Future input fields can be removed without affecting the result.
            only_past=hourly.loc[hourly.index<=time].copy()
            again=forecast_at(station,time,bundle=bundle,hourly=only_past)
            assert forecast['predicted_peak']==again['predicted_peak']
            replay_rows+=1
        threshold=STATIONS[station]['danger']
        danger=hourly.loc[(hourly.index>=pd.Timestamp('2023-01-01'))&
                          (hourly.index<=pd.Timestamp('2025-12-31 23:59:59'))&hourly.observed_max.ge(threshold)]
        group=(danger.index.to_series().diff()>pd.Timedelta(hours=24)).cumsum()
        for _,event in danger.groupby(group.to_numpy()):
            first=event.index.min();last=event.index.max()
            eligible=subset.loc[(subset.time<first)&(subset.label_end>=first)]
            chosen=recorded.loc[recorded.station.eq(station)&recorded.time.isin(eligible.time)]
            alerts=chosen.loc[chosen[selected]>=threshold]
            lead=float((first-alerts.time.min()).total_seconds()/3600) if len(alerts) else None
            episodes.append({'station':station,'first_observed_danger':first.isoformat(),
                             'last_observed_danger':last.isoformat(),'above_danger_readings':len(event),
                             'eligible_pre_onset_origins':len(eligible),'advance_prediction':bool(len(alerts)),
                             'lead_hours':lead})
    pd.DataFrame(episodes).to_csv(EXPERIMENT/'danger_episode_audit.csv',index=False)
    findings={'frozen_artifact_checks':'passed','test_rows_replayed':len(test),'runtime_prefix_replays':replay_rows,
              'largest_prediction_difference_m':max_error,'future_observations_not_required':True,
              'observed_danger_episodes':len(episodes),
              'episodes_with_eligible_advance_origin':sum(e['eligible_pre_onset_origins']>0 for e in episodes),
              'episodes_predicted_in_advance':sum(e['advance_prediction'] for e in episodes),
              'episode_definition':'Observed danger readings grouped when gaps <=24h; source gaps can fragment episodes.',
              'episodes':episodes}
    (EXPERIMENT/'verification.json').write_text(json.dumps(findings,indent=2),encoding='utf-8')
    print(json.dumps(findings,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
