"""Prepare, select, calibrate and freeze a 24-hour peak river-level regressor."""
from pathlib import Path
import sys
import argparse
import json
from hashlib import sha256
from datetime import datetime,timezone

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from river_forecast import (ROOT,DATA,EXPERIMENT,STATIONS,FEATURES,NUMERIC_FEATURES,
                            clean_station,supervised_rows,forecast_at)

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=str),encoding='utf-8')
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return sha256(path.read_bytes()).hexdigest()

CONFIGS={
 'ridge':[dict(alpha=1.),dict(alpha=10.),dict(alpha=100.)],
 'hist_gradient_boosting':[dict(max_iter=300,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=10.),
                          dict(max_iter=400,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=20.)],
 'random_forest':[dict(n_estimators=250,max_depth=12,min_samples_leaf=3),
                   dict(n_estimators=250,max_depth=12,min_samples_leaf=8)],
 'extra_trees':[dict(n_estimators=300,max_depth=16,min_samples_leaf=3),
                dict(n_estimators=300,max_depth=16,min_samples_leaf=8)],
 'xgboost':[dict(n_estimators=300,max_depth=3,learning_rate=.04,min_child_weight=10,reg_lambda=10),
             dict(n_estimators=400,max_depth=4,learning_rate=.04,min_child_weight=5,reg_lambda=20)],
}

def prepare():
    if (EXPERIMENT/'protocol.json').exists():raise RuntimeError('Protocol frozen. Use a new version.')
    manifest=load(DATA/'raw/source_manifest.json')
    for r in manifest['resources']:assert digest(DATA/'raw'/r['file'])==r['sha256']
    (DATA/'processed').mkdir(parents=True,exist_ok=True)
    all_rows=[];audits={}
    for key in STATIONS:
        raw=pd.concat([pd.read_csv(DATA/f'raw/{key}_{period}.csv') for period in ['historical','recent']],ignore_index=True)
        hourly,audit=clean_station(raw,key)
        hourly.to_csv(DATA/f'processed/{key}_hourly.csv')
        daily=supervised_rows(hourly,key)
        daily=daily.loc[(daily.index.year>=2014)&(daily.index.year<=2025)]
        audit.update(candidate_origins=len(daily),insufficient_history=int((~daily.input_eligible).sum()),
                     incomplete_future=int(daily.target_peak.isna().sum()))
        eligible=daily.loc[daily.input_eligible&daily.target_peak.notna()].reset_index()
        eligible['split']=np.select([eligible.time.dt.year.le(2020),eligible.time.dt.year.eq(2021),
                                      eligible.time.dt.year.eq(2022)],['train','validation','calibration'],default='test')
        # Purge any future target window that crosses a partition boundary.
        ends={'train':pd.Timestamp('2020-12-31 23:59:59'),'validation':pd.Timestamp('2021-12-31 23:59:59'),
              'calibration':pd.Timestamp('2022-12-31 23:59:59'),'test':pd.Timestamp('2025-12-31 23:59:59')}
        eligible=eligible.loc[[end<=ends[split] for end,split in zip(eligible.label_end,eligible.split)]]
        audit['eligible_origins']=len(eligible)
        audits[key]=audit;all_rows.append(eligible)
    master=pd.concat(all_rows,ignore_index=True).sort_values(['time','station']).reset_index(drop=True)
    master.to_csv(DATA/'processed/model_dataset.csv',index=False)
    EXPERIMENT.mkdir(parents=True,exist_ok=True)
    for split in ['train','validation','calibration','test']:
        master.loc[master.split.eq(split)].to_csv(EXPERIMENT/f'{split}.csv',index=False)
    master[['time','label_end','station','split']].to_csv(EXPERIMENT/'split_manifest.csv',index=False)
    dump(EXPERIMENT/'data_quality.json',audits)
    protocol={
      'title':'Maharashtra River-Level Forecast and Flood-Stage Warning',
      'target':'Maximum valid observed river level over (issue time, issue time +24h], metres; predict change from current.',
      'issue_time':'08:00 IST once per day; 24 hourly future bins must have actual observations.',
      'features':FEATURES,'stations':STATIONS,'seed':20261024,'configs':CONFIGS,'training_weights':[1.,4.],
      'splits':{'train':'2014–2020','validation':'2021','calibration':'2022','test':'2023–2025'},
      'selection':'Minimum validation score: 0.5 macro-station MAE overall +0.5 macro-station MAE on '
                  'target peaks at/above official warning. Ties overall MAE then fewer training-weight changes. '
                  'Baselines participate in selection. Test/calibration never choose the predictor.',
      'calibration':'90% nominal empirical absolute-residual range, from separate 2022 calibration rows per station. '
                    'No exchangeability or guaranteed future coverage claim; report test coverage.',
      'baselines':['persistence','positive_6h_trend_damped_12h','previous_24h_max'],
      'quality':'Fixed wide bounds [zero gauge RL-10, danger+30]m; unresolved conflicting duplicates missing; '
                '>10m/hour accepted-history jumps missing; unchanged run >72h stale from that point forward. '
                'No decimal/datum repairs. Features forward-fill at most2h; >=65 actually observed past72 hourly bins; '
                'future label bins never filled. Quality rules do not use model scores or future values.',
      'source_manifest_sha256':digest(DATA/'raw/source_manifest.json'),
      'split_hashes':{s:digest(EXPERIMENT/f'{s}.csv') for s in ['train','validation','calibration','test']},
      'hourly_hashes':{s:digest(DATA/f'processed/{s}_hourly.csv') for s in STATIONS},
      'original_model_sha256':digest(ROOT/'flood_model.pkl'),
      'limits':'Historical replay, not live alerts. Source records were inspected in feasibility research. '
               'Contiguous hourly flood records are not independent floods. Only three known gauges. '
               'Archive corrections/availability may differ from what was issued in real time. '
               'Official threshold datum/history assumed consistent for selected series; source QC cannot prove every value. '
               'No rainfall predictor: co-located gauge rain is sparse/nonzero-only; it is not filled with zeros. '
               'Availability favours high-frequency observation seasons; excluded/missing windows are not normal labels.',
      'versions':{'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__},
      'created_utc':datetime.now(timezone.utc).isoformat()}
    dump(EXPERIMENT/'protocol.json',protocol)
    print(master.groupby(['split','station']).size().to_string(),flush=True)

def read_split(name,p):
    path=EXPERIMENT/f'{name}.csv';assert digest(path)==p['split_hashes'][name]
    return pd.read_csv(path,parse_dates=['time','label_end'])

def make_model(name,config):
    prep=ColumnTransformer([('numeric',Pipeline([('median',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUMERIC_FEATURES),
                            ('station',OneHotEncoder(handle_unknown='error',sparse_output=False),['station'])],sparse_threshold=0)
    if name=='ridge':model=Ridge(**config)
    elif name=='hist_gradient_boosting':model=HistGradientBoostingRegressor(**config,learning_rate=.05,early_stopping=False,random_state=20261024)
    elif name=='random_forest':model=RandomForestRegressor(**config,n_jobs=2,random_state=20261024)
    elif name=='extra_trees':model=ExtraTreesRegressor(**config,n_jobs=2,random_state=20261024)
    else:model=XGBRegressor(**config,objective='reg:squarederror',tree_method='hist',n_jobs=2,random_state=20261024)
    return Pipeline([('preprocessing',prep),('model',model)])

def binary_metrics(y,pred):
    y=np.asarray(y,dtype=bool);pred=np.asarray(pred,dtype=bool)
    tp=int(np.sum(y&pred));fp=int(np.sum(~y&pred));fn=int(np.sum(y&~pred));tn=int(np.sum(~y&~pred))
    precision=tp/(tp+fp) if tp+fp else None;recall=tp/(tp+fn) if tp+fn else None
    return {'positive':int(y.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':precision,'recall':recall,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}

def evaluate_predictions(frame,pred):
    actual=frame.target_peak.to_numpy();pred=np.asarray(pred)
    warning=frame.warning_level.to_numpy();danger=frame.danger_level.to_numpy()
    errors=np.abs(actual-pred);high=actual>=warning
    current=frame.current_level.to_numpy()
    station_stats={};macro=[];high_macro=[]
    for key in STATIONS:
        mask=frame.station.eq(key).to_numpy()
        if not mask.any():continue
        hmask=mask&high
        m={'n':int(mask.sum()),'mae':float(errors[mask].mean()),'rmse':float(np.sqrt(mean_squared_error(actual[mask],pred[mask]))),
           'r2':float(r2_score(actual[mask],pred[mask])) if mask.sum()>1 else None,
           'warning_period_mae':float(errors[hmask].mean()) if hmask.any() else None,
           'warning_period_n':int(hmask.sum())}
        station_stats[key]=m;macro.append(m['mae'])
        if hmask.any():high_macro.append(m['warning_period_mae'])
    onset=current<danger
    result={'n':len(frame),'mae':float(errors.mean()),'rmse':float(np.sqrt(mean_squared_error(actual,pred))),
            'r2_relative_to_station_warning':float(r2_score(actual-warning,pred-warning)),
            'macro_station_mae':float(np.mean(macro)),
            'warning_period_mae':float(errors[high].mean()) if high.any() else None,
            'macro_warning_period_mae':float(np.mean(high_macro)) if high_macro else None,
            'warning_period_n':int(high.sum()),'per_station':station_stats,
            'warning':binary_metrics(actual>=warning,pred>=warning),
            'danger':binary_metrics(actual>=danger,pred>=danger),
            'danger_onset':binary_metrics((actual>=danger)[onset],(pred>=danger)[onset])}
    result['selection_score']=.5*result['macro_station_mae']+.5*(result['macro_warning_period_mae']
                                                                if high_macro else result['macro_station_mae'])
    return result

def baseline_predictions(frame,name):
    current=frame.current_level.to_numpy()
    if name=='persistence':return current
    if name=='positive_6h_trend_damped_12h':return current+np.maximum(frame.change_6h.to_numpy(),0)*2
    if name=='previous_24h_max':return current+frame.max_change_24h.to_numpy()
    raise ValueError(name)

def select():
    if (EXPERIMENT/'selection.json').exists():raise RuntimeError('Selection frozen.')
    p=load(EXPERIMENT/'protocol.json');train=read_split('train',p);val=read_split('validation',p)
    (EXPERIMENT/'models').mkdir(exist_ok=True)
    candidates=[]
    for name in p['baselines']:
        pred=baseline_predictions(val,name)
        candidates.append({'name':name,'type':'baseline','validation':evaluate_predictions(val,pred)})
        pd.DataFrame({'time':val.time,'station':val.station,'prediction':pred}).to_csv(EXPERIMENT/f'{name}_validation.csv',index=False)
    for name,configs in p['configs'].items():
        for i,config in enumerate(configs):
            for weight in p['training_weights']:
                identity=f'{name}_{i}_w{int(weight)}'
                estimator=make_model(name,config)
                counts=train.station.value_counts()
                weights=train.station.map(lambda s:1/counts[s]).to_numpy(copy=True)
                weights/=weights.mean()
                weights*=np.where(train.target_peak>=train.warning_level,weight,1.)
                estimator.fit(train[FEATURES],train.target_change,model__sample_weight=weights)
                pred=val.current_level.to_numpy()+estimator.predict(val[FEATURES])
                metrics=evaluate_predictions(val,pred)
                path=EXPERIMENT/f'models/{identity}.joblib'
                joblib.dump(estimator,path,compress=3)
                candidates.append({'name':identity,'type':'ml','learner':name,'config':config,'high_water_weight':weight,
                                   'artifact_sha256':digest(path),'validation':metrics})
                pd.DataFrame({'time':val.time,'station':val.station,'prediction':pred}).to_csv(EXPERIMENT/f'{identity}_validation.csv',index=False)
                print(identity,'validation score',round(metrics['selection_score'],4),'MAE',round(metrics['mae'],4),flush=True)
    rank=lambda c:(c['validation']['selection_score'],c['validation']['macro_station_mae'],c.get('high_water_weight',1.))
    winner=min(candidates,key=rank)
    best_ml=min((c for c in candidates if c['type']=='ml'),key=rank)
    best_baseline=min((c for c in candidates if c['type']=='baseline'),key=rank)
    selection={'selected':winner,'best_ml':best_ml,'best_baseline':best_baseline,'candidates':candidates,
               'protocol_sha256':digest(EXPERIMENT/'protocol.json'),'test_or_calibration_used_for_selection':False,
               'created_utc':datetime.now(timezone.utc).isoformat()}
    dump(EXPERIMENT/'selection.json',selection)
    print('FROZEN WINNER',winner['name'],winner['validation'],flush=True)

def predict(frame,item):
    if item['type']=='baseline':return baseline_predictions(frame,item['name'])
    path=EXPERIMENT/f"models/{item['name']}.joblib";assert digest(path)==item['artifact_sha256']
    return frame.current_level.to_numpy()+joblib.load(path).predict(frame[FEATURES])

def evaluate():
    if (EXPERIMENT/'final_results.json').exists():raise RuntimeError('Test evaluated and frozen; do not retune.')
    p=load(EXPERIMENT/'protocol.json');s=load(EXPERIMENT/'selection.json')
    assert digest(EXPERIMENT/'protocol.json')==s['protocol_sha256']
    calibration=read_split('calibration',p);test=read_split('test',p)
    selected=s['selected'];cal_pred=predict(calibration,selected)
    error_ranges={};cal_n={}
    for key in STATIONS:
        mask=calibration.station.eq(key).to_numpy();errors=np.abs(calibration.target_peak.to_numpy()[mask]-cal_pred[mask])
        if len(errors)<30:raise RuntimeError(f'Insufficient independent calibration rows for {key}')
        q=min(1.,np.ceil((len(errors)+1)*.9)/len(errors))
        error_ranges[key]=float(np.quantile(errors,q,method='higher'));cal_n[key]=len(errors)
    results={};predictions=test[['time','label_end','station','current_level','target_peak','warning_level','danger_level']].copy()
    # All candidates frozen on validation, never select among these using test.
    for item in s['candidates']:
        pred=predict(test,item);results[item['name']]=evaluate_predictions(test,pred)
        predictions[item['name']]=pred
    pred=predictions[selected['name']].to_numpy()
    radii=test.station.map(error_ranges).to_numpy()
    interval={'nominal':.90,'coverage':float(np.mean(np.abs(test.target_peak-pred)<=radii)),
              'mean_width_m':float(np.mean(2*radii)),
              'per_station':{k:{'radius_m':error_ranges[k],'calibration_n':cal_n[k],
                'test_coverage':float(np.mean(np.abs(test.loc[test.station.eq(k),'target_peak']-pred[test.station.eq(k)])<=error_ranges[k]))}
                for k in STATIONS}}
    if selected['type']=='ml':kind='ml';pipeline=joblib.load(EXPERIMENT/f"models/{selected['name']}.joblib")
    else:kind='persistence' if selected['name']=='persistence' else 'trend' if selected['name']=='positive_6h_trend_damped_12h' else 'previous_max';pipeline=None
    bundle={'pipeline':pipeline,'predictor_type':kind,'selected_name':selected['name'],'features':FEATURES,
            'error_ranges':error_ranges,'training_end':'2020-12-31 23:59:59','stations':STATIONS,
            'target':p['target'],'mode':'Historical forecast replay; not a live alert'}
    joblib.dump(bundle,EXPERIMENT/'forecast_bundle.joblib',compress=3)
    result={'selected_name':selected['name'],'selected_type':selected['type'],'test_metrics':results,
            'selected_test':results[selected['name']],'best_validation_ml_test':results[s['best_ml']['name']],
            'best_validation_baseline_test':results[s['best_baseline']['name']],
            'error_range':interval,'selection_sha256':digest(EXPERIMENT/'selection.json'),
            'bundle_sha256':digest(EXPERIMENT/'forecast_bundle.joblib'),
            'original_model_unchanged':digest(ROOT/'flood_model.pkl')==p['original_model_sha256'],
            'test_used_for_selection':False,'evaluated_utc':datetime.now(timezone.utc).isoformat()}
    predictions.to_csv(EXPERIMENT/'test_predictions.csv',index=False)
    dump(EXPERIMENT/'final_results.json',result)
    dump(EXPERIMENT/'model_metadata.json',{'protocol':p,'selection':selected,'calibration':interval,
                                         'test':result['selected_test'],'bundle_sha256':result['bundle_sha256']})
    write_report(p,s,result)
    print('FINAL SELECTED',result['selected_name'],result['selected_test'],interval,flush=True)

def write_report(p,s,r):
    lines=['# Maharashtra River-Level Forecast and Flood-Stage Warning','',
           '**Scope:** forecast the maximum observed level over the next 24 hours at 08:00 IST for three known CWC gauges. '
           'Apply official station warning/danger rules to numeric forecasts. Historical replay, not live alerts or district inundation.',
           '',p['limits'],'','## Data and chronology','',
           'Manual CWC records only. Telemetry was rejected during feasibility inspection. Missing/ambiguous observations stay missing; '
           'no offset/decimal fixes, future filling or synthetic flood labels. All 24 future hourly bins must be observed. '
           'Only days with sufficient trailing observations can enter the training/evaluation table. This coverage constraint affects representativeness.',
           p['quality'],'', 'Train 2014–2020; select 2021; separate error-range calibration 2022; final test 2023–2025. '
           'Future label windows crossing partition boundaries are purged. One origin per station/day prevents overlapping 24-hour labels.',
           '', '## Selection and models','',p['selection'],
           'Compare Ridge, Random Forest, Extra Trees, HistGradientBoosting and XGBoost; '
           'training-only station-balanced weighting, optionally 4× weight for training target peaks above warning. '
           'All regressors predict future-peak change from current level, not the future absolute gauge offset. '
           'Numeric imputation, scaling and station categories fit training only. No random early-stopping holdout.',
           f"Frozen predictor: **{s['selected']['name']}**. Best validation ML: {s['best_ml']['name']}; "
           f"best validation baseline: {s['best_baseline']['name']}.",
           '', '| Predictor | Validation selection score | Test MAE m | Test RMSE m | Relative R² | Warning-period MAE m |',
           '|---|---:|---:|---:|---:|---:|']
    for item in sorted(s['candidates'],key=lambda c:c['validation']['selection_score']):
        m=r['test_metrics'][item['name']]
        lines.append(f"| {item['name']} | {item['validation']['selection_score']:.4f} | {m['mae']:.4f} | "
                     f"{m['rmse']:.4f} | {m['r2_relative_to_station_warning']:.4f} | {m['warning_period_mae']} |")
    m=r['selected_test']
    lines+=['','## Selected predictor: station results','',
            '| Gauge | Test forecasts | MAE m | RMSE m | Within-station R² | Warning-period MAE m |',
            '|---|---:|---:|---:|---:|---:|']
    for key,stat in m['per_station'].items():
        lines.append(f"| {STATIONS[key]['name']} | {stat['n']} | {stat['mae']:.4f} | {stat['rmse']:.4f} | "
                     f"{stat['r2']} | {stat['warning_period_mae']} |")
    lines+=['','R² is computed relative to each station warning level when pooled, and within each station separately. '
            'Pooling raw 400–540m offsets would inflate R² without proving forecasting skill. MAE/RMSE remain metres.',
            '', '## Rule-derived warning evaluation','',
            'Danger onset restricts origins to current level below danger. This prevents ongoing floods from masquerading as advance prediction.',
            '```json',json.dumps({k:m[k] for k in ['warning','danger','danger_onset']},indent=2),'```','',
            '## Forecast range','',p['calibration'],'```json',json.dumps(r['error_range'],indent=2),'```','',
            'No target threshold is moved to improve classification metrics. Forecast ranges are estimated uncertainty, not flood probabilities. '
            'Few independent flood episodes limit conclusions; report missed onsets/false alarms alongside regression error. '
            'Historical source revisions and broad QC heuristics do not establish operational reliability.',
            '', '## Files and run commands','',
            'Raw source manifest, quality audit, explicit split manifest, model dataset, all validation/test predictions, '
            'models, selected inference bundle and metadata are preserved. Previous annual-flow experiments remain unchanged.',
            '```powershell','.\\.venv\\Scripts\\python.exe scripts/fetch_river_stage.py',
            '.\\.venv\\Scripts\\python.exe scripts/train_river_stage.py prepare',
            '.\\.venv\\Scripts\\python.exe scripts/train_river_stage.py select',
            '.\\.venv\\Scripts\\python.exe scripts/train_river_stage.py evaluate','```','',
            'Stages refuse to overwrite frozen protocol/selection/test results. Use a new prespecified version for another study.',
            '', 'Sources: [CWC observations](https://nwdp.nwic.gov.in/dataset/river-water-level-manual-hourly-cwc-subernarekha), '
            '[CWC official thresholds, page 46](https://cwc.gov.in/sites/default/files/sopapril2025.pdf#page=46).','']
    (EXPERIMENT/'MODEL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','select','evaluate'])
    args=parser.parse_args()
    with threadpool_limits(limits=2):{'prepare':prepare,'select':select,'evaluate':evaluate}[args.stage]()
