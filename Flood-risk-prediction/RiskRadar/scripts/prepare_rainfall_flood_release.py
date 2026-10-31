"""Build the PS-aligned release from existing dated events and frozen models.

This does not retrain, retune or replace the previous severity-study selection.
"""
from pathlib import Path
import sys
import json
from datetime import datetime,timezone

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from rainfall_flood import ROOT,SOURCE,DATA,RELEASE,RAIN,MODELS,digest,daily_rainfall
from scripts.compare_indofloods_xgboost import feature_frame


def main():
    if (RELEASE/'manifest.json').exists():
        raise RuntimeError('Release is frozen. Verify it rather than overwrite it.')
    protocol=json.loads((SOURCE/'protocol.json').read_text())
    selection=json.loads((SOURCE/'selection.json').read_text())
    results=json.loads((SOURCE/'final_results.json').read_text())
    assert digest(SOURCE/'protocol.json')==selection['protocol_sha256']
    assert digest(SOURCE/'selection.json')==results['selection_sha256']
    for name,meta in protocol['source_manifest'].items():
        assert digest(SOURCE/'raw'/name)==meta['sha256'],name
    e=pd.read_csv(SOURCE/'raw/floodevents_indofloods.csv')
    r=pd.read_csv(SOURCE/'raw/precipitation_variables_indofloods.csv')
    m=pd.read_csv(SOURCE/'raw/metadata_indofloods.csv')
    c=pd.read_csv(SOURCE/'raw/catchment_characteristics_indofloods.csv')
    assert e.EventID.is_unique and r.EventID.is_unique and m.GaugeID.is_unique and c.GaugeID.is_unique
    e['GaugeID']=e.EventID.str.rsplit('-',n=1).str[0]
    merged=e.merge(r,on='EventID',validate='one_to_one').merge(m,on='GaugeID',validate='many_to_one')
    merged=merged.merge(c,on='GaugeID',validate='many_to_one',suffixes=('','_catchment'))
    assert len(merged)==len(e)==4548 and merged.Privacy.eq('Open').all()
    merged=feature_frame(merged)
    merged['Start Date']=pd.to_datetime(merged['Start Date'])
    merged['End Date']=pd.to_datetime(merged['End Date'])
    merged['target']=merged['Flood Type'].eq('Severe Flood').astype(int)
    splits=pd.read_csv(SOURCE/'data/split_manifest.csv')[['EventID','split']]
    merged=merged.merge(splits,on='EventID',validate='one_to_one')
    # Source rainfall ends in 2019. Exclusions are determined by documented
    # source coverage, never by test performance or the amount of rainfall.
    analysis=merged.loc[merged['Start Date'].dt.year.le(2019)].sort_values(['Start Date','EventID']).copy()
    days=np.diff(np.column_stack([np.zeros(len(analysis)),analysis[RAIN].to_numpy(float)]),axis=1)
    assert np.isfinite(days).all() and (days>=-1e-6).all()
    assert analysis['Flood Type'].isin(['Flood','Severe Flood']).all()
    for day in range(1,11):analysis[f'rain_day_minus_{day}_mm']=days[:,day-1]
    DATA.mkdir(parents=True,exist_ok=True);RELEASE.mkdir(parents=True,exist_ok=True)
    analysis.to_csv(DATA/'matched_events.csv',index=False)
    inventory=pd.read_csv(SOURCE/'raw/India_Flood_Inventory_v3.csv')
    for date in ['Start Date','End Date']:
        inventory[date]=pd.to_datetime(inventory[date],format='%d-%m-%Y %H:%M',errors='coerce')
    assert inventory.UEI.is_unique
    inventory.to_csv(DATA/'flood_inventory.csv',index=False)
    train=analysis.loc[analysis.split.eq('train')]
    meta={'problem_statement':'Rainfall & Flood Risk Analysis Using Historical Weather Data',
          'observational_unit':'Recorded flood event, not monitored daily flood/non-flood observations',
          'source_events':len(e),'source_event_classes':e['Flood Type'].value_counts().to_dict(),
          'matched_rainfall_events':len(analysis),'gauges':int(analysis.GaugeID.nunique()),
          'states':int(analysis.State.nunique()),'rainfall_source_coverage_excluded':len(merged)-len(analysis),
          'period':[analysis['Start Date'].min().isoformat(),analysis['Start Date'].max().isoformat()],
          'model_rows':int(analysis.split.isin(['train','validation','test']).sum()),
          'training_rainfall_quartiles':{key:train[key].quantile([.25,.5,.75]).tolist() for key in RAIN},
          'inventory_records':len(inventory),'inventory_invalid_start_dates':int(inventory['Start Date'].isna().sum()),
          'raw_source_sha256':{name:digest(SOURCE/'raw'/name) for name in
                             ['floodevents_indofloods.csv','precipitation_variables_indofloods.csv',
                              'metadata_indofloods.csv','catchment_characteristics_indofloods.csv','India_Flood_Inventory_v3.csv']},
          'dataset_sha256':digest(DATA/'matched_events.csv'),'inventory_sha256':digest(DATA/'flood_inventory.csv'),
          'no_flood_labels_available':False,'unknown_days_relabelled_as_non_flood':False,
          'label_definition':'Publisher Flood versus Severe Flood; conditional on a recorded flood',
          'missing_2020_rain_not_interpreted_as_zero':True,
          'built_utc':datetime.now(timezone.utc).isoformat()}
    (DATA/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    bundles={}
    for name in MODELS:
        assert digest(SOURCE/f'models/{name}.joblib')==selection['families'][name]['artifact_sha256']
        bundles[name]=joblib.load(SOURCE/f'models/{name}.joblib')
    joblib.dump({'models':bundles,'features':{key:protocol['features'][key] for key in MODELS},
                 'thresholds':{key:selection['families'][key]['validation']['threshold'] for key in MODELS},
                 'validation_winner':selection['winner_by_validation'],
                 'target':protocol['target'],'mode':'Retrospective severity comparison for known floods'},
                RELEASE/'severity_comparison.joblib',compress=3)
    release={'scope':'Historical rainfall analysis, rainfall–flood relationships, observed flood severity and experimental severity comparison',
             'models_retrained_for_this_release':False,'frozen_source_experiment':str(SOURCE.relative_to(ROOT)),
             'model_selection_changed':False,'new_independent_test_claimed':False,
             'bundle_sha256':digest(RELEASE/'severity_comparison.joblib'),
             'analysis_manifest_sha256':digest(DATA/'manifest.json'),
             'source_selection_sha256':digest(SOURCE/'selection.json'),
             'source_results_sha256':digest(SOURCE/'final_results.json'),
             'previous_river_bundle_sha256':digest(ROOT/'experiments/river_stage_v1/forecast_bundle.joblib'),
             'original_model_sha256':digest(ROOT/'flood_model.pkl'),
             'test_metrics':results['test_metrics'],'validation_winner':selection['winner_by_validation'],
             'limits':['No verified non-flood observations; flood-occurrence classifier not claimed.',
                       'Event-aligned rainfall and static climate proxies support retrospective analysis.',
                       'Fitted severity models are weak and do not beat always-severe F1.',
                       'Archive coverage and gauge dependence limit generalisation.']}
    (RELEASE/'manifest.json').write_text(json.dumps(release,indent=2),encoding='utf-8')
    print(json.dumps({key:meta[key] for key in ['source_events','matched_rainfall_events','model_rows','gauges','states','inventory_records']},indent=2))


if __name__=='__main__':main()
