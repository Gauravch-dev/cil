"""Prepare, select and independently evaluate two real-label XGBoost experiments.

The application model is never overwritten. Run prepare, select, evaluate in order.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256, md5
import json
from pathlib import Path
import platform
import shutil
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, fbeta_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/indofloods_severity_v1'
SEED=20261024
FILES={
 'floodevents_indofloods.csv':'729f3f857e4d4dce33fbe8b67c492172',
 'precipitation_variables_indofloods.csv':'dfd1491d22fff46e6086eebda8f758c9',
 'metadata_indofloods.csv':'8228295ef7133fb46c92dd7ff0066c14',
 'catchment_characteristics_indofloods.csv':'480e5f8604f1232b69b73ec25dd74a18',
 'variables_description_indofloods.pdf':'d8cf91fe8731922d7195bfb3394d0e2f',
 'catchments_shapefiles_indofloods.zip':'87021353b066248023a62e492d766839',
}
EXTRA=['India_Flood_Inventory_v3.csv','District_FloodedArea.csv','District_FloodImpact.csv','DFSI.csv']
STATIC_NUM=['Drainage Area','Catchment Relief','Stream Order','Drainage Density','Drainage Texture',
 'Catchment Length','Maximal Flow Length','Sinuosity Index','Form Factor','Relief Ratio',
 'Ruggedness Number','Infiltration Number','Annual Mean Temperature','Temperature Seasonality',
 'Annual Precipitation','Precipitation of Wettest Month','Precipitation Seasonality','Latitude','Longitude']
STATIC_CAT=['KoppenGeiger Climate Type','Land cover','Soil type','lithology type']
SEASON=['month_sin','month_cos']
RAIN=[f'T{i}d' for i in range(1,11)]
RAIN_DERIVED=['rain_max_10d','rain_std_10d','rainy_days_10d','recent_3d_share']
FEATURES={
 'xgb_static':STATIC_NUM+STATIC_CAT+SEASON,
 'xgb_rainfall':STATIC_NUM+STATIC_CAT+SEASON+RAIN+RAIN_DERIVED,
 'logistic_rainfall_baseline':RAIN+SEASON,
}
FORBIDDEN={'EventID','GaugeID','Flood Type','target','Peak Flood Level (m)','Peak FL Date',
 'Num Peak FL','Peak Discharge Q (cumec)','Peak Discharge Date','Flood Volume (cumec)',
 'Event Duration (days)','Time to Peak (days)','Recession Time (day)','End Date',
 'Warning Level','Danger Level','DFSI','Human_fatality','Human_injured','Mean_Flood_Duration'}
ORIGINAL_MODEL_SHA='aad88a12f4645f3703c996641a15ce40603f6d5a608918fad4f37496d2f89395'

def dump(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str),encoding='utf-8')

def load(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return sha256(path.read_bytes()).hexdigest()

def feature_frame(frame):
    frame=frame.copy()
    # Model issue time is the day before known flood onset, conditional on an event.
    asof=pd.to_datetime(frame['Start Date'])-pd.Timedelta(days=1)
    month=asof.dt.month
    frame['month_sin']=np.sin(2*np.pi*month/12)
    frame['month_cos']=np.cos(2*np.pi*month/12)
    cumulative=frame[RAIN].to_numpy(float)
    days=np.diff(np.column_stack([np.zeros(len(frame)),cumulative]),axis=1)
    assert (days>=-1e-6).all(), 'Cumulative rainfall is not monotonic'
    frame['rain_max_10d']=days.max(axis=1)
    frame['rain_std_10d']=days.std(axis=1)
    frame['rainy_days_10d']=(days>0).sum(axis=1)
    frame['recent_3d_share']=np.divide(frame['T3d'],frame['T10d'],out=np.zeros(len(frame)),where=frame['T10d'].gt(0))
    return frame

def prepare(source):
    if (RUN/'selection.json').exists():raise RuntimeError('Selection is frozen; do not overwrite this experiment.')
    raw=RUN/'raw';raw.mkdir(parents=True,exist_ok=True)
    manifest={}
    for name,expected in FILES.items():
        incoming=source/name
        assert md5(incoming.read_bytes()).hexdigest()==expected,name
        dest=raw/name
        if dest.exists():assert digest(dest)==digest(incoming)
        else:shutil.copy2(incoming,dest)
        manifest[name]={'md5':expected,'sha256':digest(dest),'bytes':dest.stat().st_size}
    extra_summary={}
    for name in EXTRA:
        incoming=source/name
        if incoming.exists():
            dest=raw/name
            if not dest.exists():shutil.copy2(incoming,dest)
            d=pd.read_csv(dest,low_memory=False)
            extra_summary[name]={'rows':len(d),'columns':list(d.columns),'sha256':digest(dest)}
    e=pd.read_csv(raw/'floodevents_indofloods.csv')
    rain=pd.read_csv(raw/'precipitation_variables_indofloods.csv')
    meta=pd.read_csv(raw/'metadata_indofloods.csv')
    catch=pd.read_csv(raw/'catchment_characteristics_indofloods.csv')
    assert e.EventID.is_unique and rain.EventID.is_unique and meta.GaugeID.is_unique and catch.GaugeID.is_unique
    assert set(e.EventID)==set(rain.EventID)
    assert set(e['Flood Type'])=={'Flood','Severe Flood'}
    e['GaugeID']=e.EventID.str.rsplit('-',n=1).str[0]
    warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)
    master=e.merge(rain,on='EventID',validate='one_to_one').copy()
    master=master.merge(meta,on='GaugeID',validate='many_to_one').copy()
    master=master.merge(catch,on='GaugeID',validate='many_to_one',suffixes=('','_catchment')).copy()
    assert len(master)==len(e)==4548
    assert set(e.GaugeID)==set(catch.GaugeID)
    assert master.Privacy.eq('Open').all()
    start=pd.to_datetime(master['Start Date'],errors='raise')
    end=pd.to_datetime(master['End Date'],errors='raise')
    assert start.le(end).all()
    assert master[RAIN].notna().all().all() and master[RAIN].ge(0).all().all()
    master['target']=master['Flood Type'].eq('Severe Flood').astype(int)
    # Preserve publisher labels. The tabulated boundary is >=, despite prose 'above'.
    assert master.target.eq(master['Peak Flood Level (m)'].ge(master['Danger Level']).astype(int)).all()
    master['split']='excluded_rainfall_source_after_2019'
    train=start.lt('2010-01-01')&end.lt('2010-01-01')
    val=start.ge('2010-01-01')&start.lt('2015-01-01')&end.lt('2015-01-01')
    test=start.ge('2015-01-01')&start.lt('2020-01-01')&end.lt('2020-01-01')
    master.loc[start.lt('2020-01-01'),'split']='excluded_event_crosses_period_boundary'
    for mask,name in [(train,'train'),(val,'validation'),(test,'test')]:master.loc[mask,'split']=name
    master=feature_frame(master).sort_values(['Start Date','GaugeID','EventID']).reset_index(drop=True)
    (RUN/'data').mkdir(exist_ok=True)
    master.to_csv(RUN/'data/event_master.csv',index=False)
    for name in ['train','validation','test']:
        part=master.loc[master.split.eq(name)].copy()
        assert len(part)>0
        part.to_csv(RUN/f'data/{name}.csv',index=False)
    master.loc[~master.split.isin(['train','validation','test'])].to_csv(RUN/'data/excluded.csv',index=False)
    split_manifest=master[['EventID','GaugeID','Start Date','End Date','split']]
    split_manifest.to_csv(RUN/'data/split_manifest.csv',index=False)
    for cols in FEATURES.values():assert not set(cols)&FORBIDDEN
    old_schema=list(joblib.load(ROOT/'feature_columns.pkl'))
    protocol={
      'target':'Severe Flood=1, Flood=0; conditional on an observed flood event',
      'not_a_target':'Flood versus no flood; future daily/weekly occurrence is not measurable here',
      'original_task_compatibility':{'exact_dataset':False,'original_schema':old_schema,
        'missing_required_input':'curve_number_amcii; source class vocabulary and feature definitions also differ',
        'reason':'No non-flood observations; do not reinterpret class Flood as a no-flood negative.'},
      'comparison':{'xgb_static':'Static catchment, climate, location and common seasonal inputs',
                    'xgb_rainfall':'Same inputs plus antecedent 1–10-day rainfall and derived rainfall indicators'},
      'split':{'train':'events starting and ending before 2010',
               'validation':'onset 2010–2014, end before 2015',
               'test':'onset 2015–2019, end before 2020'},
      'coverage_exclusion':'All 2020 events excluded before fitting: EM-Earth version 1 ends in 2019; 54/55 source 2020 events have T10d=0.',
      'coverage_source':'https://journals.ametsoc.org/view/journals/bams/103/4/BAMS-D-21-0106.1.xml',
      'selection':'Highest validation average precision (PR-AUC/AP), then F1. Decision threshold maximises validation F1, with F2 as first tie-breaker.',
      'preprocessing':'Numeric median and category vocabulary fitted on train only. Natural validation/test distributions.',
      'imbalance':'Candidate scale_pos_weight values 1 and training negative/positive ratio; no validation/test resampling.',
      'static_vintage_limitation':'Modern static catchment/climate summaries are retrospective proxies, not a reconstruction of covariates issued in each historical year. Anthropogenic vintage fields excluded.',
      'occurrence_limitation':'Rows are sampled conditional on floods and aligned to their known onset. This is retrospective severity estimation, not a deployable flood-occurrence warning.',
      'features':FEATURES,'seed':SEED,'created_utc':datetime.now(timezone.utc).isoformat(),
      'source_manifest':manifest,'extra_inventory':extra_summary,
      'inventory_use':'Historical context only. DFSI, fatalities, affected area and whole-period impacts are excluded as predictor leakage risks; no verified non-event panel supplied.',
      'split_file_sha256':{n:digest(RUN/f'data/{n}.csv') for n in ['train','validation','test']},
      'split_counts':master.split.value_counts().to_dict(),
      'raw_class_counts':master['Flood Type'].value_counts().to_dict(),
      'raw_events':len(master),'gauges':int(master.GaugeID.nunique()),
      'maharashtra_events':int(master.State.eq('Maharashtra').sum()),
      'maharashtra_gauges':int(master.loc[master.State.eq('Maharashtra'),'GaugeID'].nunique()),
      'source_peak_discharge_available':int(master['Peak Discharge Q (cumec)'].notna().sum()),
      'threshold_boundary_equal_severe':int((master['Peak Flood Level (m)'].eq(master['Danger Level'])&master.target.eq(1)).sum()),
      'unchanged_application_model_sha256':digest(ROOT/'flood_model.pkl'),
      'versions':{'python':platform.python_version(),'xgboost':xgboost.__version__,'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__},
    }
    dump(RUN/'protocol.json',protocol)
    print(json.dumps({'prepared':len(master),'gauges':protocol['gauges'],'splits':protocol['split_counts']},indent=2))

def preprocessor(cols,scale=False):
    cat=[c for c in cols if c in STATIC_CAT];num=[c for c in cols if c not in cat]
    numeric=[('imputer',SimpleImputer(strategy='median'))]
    if scale:numeric.append(('scale',StandardScaler()))
    transformers=[('numeric',Pipeline(numeric),num)]
    if cat:transformers.append(('category',Pipeline([
      ('imputer',SimpleImputer(strategy='most_frequent')),
      ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat))
    return ColumnTransformer(transformers,sparse_threshold=0)

def metrics(y,proba,threshold):
    pred=np.asarray(proba)>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    two=len(np.unique(y))==2
    return {'n':len(y),'severe_events':int(np.sum(y)),'threshold':float(threshold),
      'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
      'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),
      'f1':float(f1_score(y,pred,zero_division=0)),'f2':float(fbeta_score(y,pred,beta=2,zero_division=0)),
      'pr_auc_ap':float(average_precision_score(y,proba)) if two else None,
      'roc_auc':float(roc_auc_score(y,proba)) if two else None,
      'brier':float(brier_score_loss(y,proba)),'mcc':float(matthews_corrcoef(y,pred)),
      'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
      'false_severe_rate':float(fp/(fp+tn)) if fp+tn else None,
      'predicted_severe_events':int(pred.sum())}

def select_threshold(y,proba):
    candidates=[metrics(y,proba,float(t)) for t in np.arange(.05,.951,.025)]
    return max(candidates,key=lambda x:(x['f1'],x['f2'],x['precision'],-abs(x['threshold']-.5)))

def checked_data(name,protocol):
    p=RUN/f'data/{name}.csv'
    assert digest(p)==protocol['split_file_sha256'][name],'Frozen split changed'
    return pd.read_csv(p,low_memory=False)

def select():
    if (RUN/'selection.json').exists():raise RuntimeError('Selection already frozen. Use a new experiment version for another protocol.')
    protocol=load(RUN/'protocol.json')
    train=checked_data('train',protocol);val=checked_data('validation',protocol)
    # No test file is opened and no test prediction is made in this stage.
    y=train.target.to_numpy();yv=val.target.to_numpy()
    weights=[1.,float((y==0).sum()/(y==1).sum())]
    grids=[
      dict(n_estimators=250,max_depth=2,learning_rate=.04,min_child_weight=3,reg_lambda=3),
      dict(n_estimators=450,max_depth=2,learning_rate=.04,min_child_weight=8,reg_lambda=10),
      dict(n_estimators=250,max_depth=3,learning_rate=.04,min_child_weight=3,reg_lambda=3),
      dict(n_estimators=450,max_depth=3,learning_rate=.04,min_child_weight=8,reg_lambda=10),
      dict(n_estimators=250,max_depth=4,learning_rate=.04,min_child_weight=3,reg_lambda=3),
      dict(n_estimators=450,max_depth=4,learning_rate=.04,min_child_weight=8,reg_lambda=10),
      dict(n_estimators=250,max_depth=3,learning_rate=.08,min_child_weight=8,reg_lambda=10),
      dict(n_estimators=450,max_depth=2,learning_rate=.08,min_child_weight=3,reg_lambda=10),
    ]
    models=RUN/'models';models.mkdir(exist_ok=True)
    results=[];frozen={}
    for family in ['xgb_static','xgb_rainfall']:
        cols=FEATURES[family];best=None
        prep=preprocessor(cols);xt=prep.fit_transform(train[cols]);xv=prep.transform(val[cols])
        for idx,params in enumerate(grids):
            for weight in weights:
                config={**params,'scale_pos_weight':weight}
                model=XGBClassifier(**config,objective='binary:logistic',eval_metric='logloss',
                    tree_method='hist',subsample=.85,colsample_bytree=.85,random_state=SEED,n_jobs=2)
                model.fit(xt,y)
                prob=model.predict_proba(xv)[:,1];score=select_threshold(yv,prob)
                row={'family':family,'config':config,'validation':score}
                results.append(row)
                if best is None or (score['pr_auc_ap'],score['f1'])>(best['validation']['pr_auc_ap'],best['validation']['f1']):
                    best=row;best_model=model;best_probs=prob
        pipe=Pipeline([('preprocessor',prep),('model',best_model)])
        joblib.dump(pipe,models/f'{family}.joblib')
        best_model.save_model(models/f'{family}.ubj')
        transformed=list(prep.get_feature_names_out())
        pd.DataFrame({'feature':transformed,'gain_based_importance':best_model.feature_importances_}).sort_values('gain_based_importance',ascending=False).to_csv(models/f'{family}_importance.csv',index=False)
        pd.DataFrame({'EventID':val.EventID,'target':yv,'score':best_probs}).to_csv(RUN/f'{family}_validation_predictions.csv',index=False)
        best['artifact_sha256']=digest(models/f'{family}.joblib')
        best['raw_feature_count']=len(cols);best['encoded_feature_count']=len(transformed)
        frozen[family]=best
        print(json.dumps({'selected':family,'validation':best['validation']},indent=2),flush=True)
    cols=FEATURES['logistic_rainfall_baseline']
    baseline=Pipeline([('preprocessor',preprocessor(cols,scale=True)),
                       ('model',LogisticRegression(C=1.,max_iter=3000,random_state=SEED))])
    baseline.fit(train[cols],y);bp=baseline.predict_proba(val[cols])[:,1]
    bm=select_threshold(yv,bp);joblib.dump(baseline,models/'logistic_rainfall_baseline.joblib')
    frozen['logistic_rainfall_baseline']={'family':'logistic_rainfall_baseline','validation':bm,
       'artifact_sha256':digest(models/'logistic_rainfall_baseline.joblib')}
    winner=max(['xgb_static','xgb_rainfall'],key=lambda f:(frozen[f]['validation']['pr_auc_ap'],frozen[f]['validation']['f1']))
    selection={'winner_by_validation':winner,'families':frozen,'training_prevalence':float(y.mean()),
       'selected_at_utc':datetime.now(timezone.utc).isoformat(),'test_used_for_selection':False,
       'protocol_sha256':digest(RUN/'protocol.json'),'candidate_fits':len(results)}
    dump(RUN/'validation_candidates.json',results);dump(RUN/'selection.json',selection)
    assert digest(ROOT/'flood_model.pkl')==ORIGINAL_MODEL_SHA
    print('FROZEN WINNER',winner,flush=True)

def evaluate():
    if (RUN/'final_results.json').exists():raise RuntimeError('Final test has already been evaluated. Do not tune this run further.')
    selection=load(RUN/'selection.json');protocol=load(RUN/'protocol.json')
    assert digest(RUN/'protocol.json')==selection['protocol_sha256']
    test=checked_data('test',protocol);train=checked_data('train',protocol)
    y=test.target.to_numpy();rows={};predictions=test[['EventID','GaugeID','Start Date','State','target']].copy()
    for family,details in selection['families'].items():
        path=RUN/f'models/{family}.joblib'
        assert digest(path)==details['artifact_sha256']
        pipe=joblib.load(path);prob=pipe.predict_proba(test[FEATURES[family]])[:,1]
        threshold=details['validation']['threshold'];rows[family]=metrics(y,prob,threshold)
        rows[family]['at_default_0_5']=metrics(y,prob,.5)
        predictions[f'{family}_score']=prob
        predictions[f'{family}_prediction']=(prob>=threshold).astype(int)
    prior=np.full(len(y),selection['training_prevalence'])
    rows['training_prevalence_baseline']=metrics(y,prior,.5)
    rows['always_severe_baseline']=metrics(y,np.ones(len(y)),.5)
    winner=selection['winner_by_validation'];subgroups={}
    for name,mask in [('Maharashtra',test.State.eq('Maharashtra')),
                      ('gauges_seen_in_training',test.GaugeID.isin(train.GaugeID)),
                      ('gauges_not_seen_in_training',~test.GaugeID.isin(train.GaugeID))]:
        mask=mask.to_numpy()
        if mask.any():
            prob=predictions.loc[mask,f'{winner}_score'].to_numpy()
            subgroups[name]=metrics(y[mask],prob,selection['families'][winner]['validation']['threshold'])
    # Paired gauge-cluster bootstrap describes uncertainty, never model selection.
    rng=np.random.default_rng(SEED);gauges=test.GaugeID.unique()
    group_indices={g:np.flatnonzero(test.GaugeID.to_numpy()==g) for g in gauges}
    diffs=[]
    for _ in range(500):
        idx=np.concatenate([group_indices[g] for g in rng.choice(gauges,size=len(gauges),replace=True)])
        yy=y[idx]
        if len(np.unique(yy))<2:continue
        pa=predictions.xgb_static_score.to_numpy()[idx];pb=predictions.xgb_rainfall_score.to_numpy()[idx]
        ta=selection['families']['xgb_static']['validation']['threshold'];tb=selection['families']['xgb_rainfall']['validation']['threshold']
        diffs.append([average_precision_score(yy,pb)-average_precision_score(yy,pa),
                      f1_score(yy,pb>=tb,zero_division=0)-f1_score(yy,pa>=ta,zero_division=0)])
    ci=np.quantile(np.asarray(diffs),[.025,.975],axis=0)
    split_stats={}
    for split in ['train','validation','test']:
        part=checked_data(split,protocol)
        split_stats[split]={'n':len(part),'gauges':int(part.GaugeID.nunique()),'severe':int(part.target.sum()),
                           'flood':int(part.target.eq(0).sum()),'first_onset':part['Start Date'].min(),'last_onset':part['Start Date'].max()}
    result={'target':protocol['target'],'selected_model':winner,'test_metrics':rows,'splits':split_stats,
        'test_subgroups_for_frozen_winner':subgroups,'paired_gauge_bootstrap_95ci_rainfall_minus_static':
        {'pr_auc_ap':ci[:,0].tolist(),'f1':ci[:,1].tolist(),'replicates':len(diffs)},
        'selection_sha256':digest(RUN/'selection.json'),'final_evaluation_utc':datetime.now(timezone.utc).isoformat(),
        'not_deployed':True,'existing_app_model_unchanged':digest(ROOT/'flood_model.pkl')==ORIGINAL_MODEL_SHA}
    predictions.to_csv(RUN/'test_predictions.csv',index=False)
    dump(RUN/'final_results.json',result)
    table=[]
    for name,score in rows.items():table.append({'model':name,**{k:v for k,v in score.items() if k!='at_default_0_5'}})
    pd.DataFrame(table).to_csv(RUN/'test_comparison.csv',index=False)
    report(result,selection,protocol)
    print(json.dumps(result,indent=2))

def report(result,selection,protocol):
    lines=['# INDOFLOODS XGBoost comparison','',
      'This experiment estimates severe versus other recorded flood events. It does not predict whether a flood will occur, and it does not validate the old application classifier.',
      '', '## Requested alternatives', '',
      '1. Exact reconstruction of the existing flood/no-flood model is unavailable: these files contain no monitored non-flood samples, and the old runoff curve-number input is absent. No labels were invented or missing inputs guessed.',
      '2. With the permitted scope adjustment, two new XGBoost models were trained on the publisher’s Flood/Severe Flood labels. Static catchment/climate features form the first version. The second adds rainfall during the 1–10 days before event onset.',
      '', '## Data and chronology','',
      'The supplied public files contain 4,548 events from 155 gauges: 2,919 Flood and 1,629 Severe Flood. Metadata also describe 59 restricted gauges without public event records; those gauges are not non-flood examples.',
      'All events join one-to-one to rainfall and many-to-one to gauge/catchment attributes. The 155 catchment shapefiles are geographical context, not additional independent samples.',
      '', '| Split | Events | Flood | Severe | Gauges | Onset range |',
      '|---|---:|---:|---:|---:|---|']
    for name,s in result['splits'].items():lines.append(f"| {name} | {s['n']} | {s['flood']} | {s['severe']} | {s['gauges']} | {s['first_onset']} to {s['last_onset']} |")
    lines+=['',protocol['coverage_exclusion'],
      'The original rainfall source covers 1950–2019: '+protocol['coverage_source'],
      'Events whose labels extend beyond a split cutoff are purged from that split. All partitions and exclusions were frozen before fitting. No district or month was selected using test outcomes.',
      '', '## Selection and leakage controls','',protocol['selection'],
      f"Each XGBoost family used 16 prespecified configurations; {selection['candidate_fits']} XGBoost fits in total. Preprocessing and class weights use training data only. The models remain fitted on training rows only; neither is refitted after final evaluation.",
      'Event peaks, event duration, recession, flood volume and post-onset dates are outcomes and excluded from predictors. Gauge/Event IDs, the warning/danger thresholds, GDP/HDI/population vintages and district-impact/DFSI aggregates are also excluded. The seasonal features represent the day before known onset.',
      'Twenty tabulated severe events equal the displayed danger level. All supplied labels match the inclusive >= relation. Publisher labels are preserved, and peak water level is never used to predict the label it defines.',
      '',f"**Model selected before test evaluation: {result['selected_model']}.**",'',
      '| Model | Precision | Recall | F1 | F2 | PR-AUC/AP | ROC-AUC | Accuracy | False severe / other floods |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in result['test_metrics'].items():
        lines.append('| '+name+' | '+' | '.join(f"{s[k]:.4f}" for k in ['precision','recall','f1','f2','pr_auc_ap','roc_auc','accuracy','false_severe_rate'])+' |')
    win=result['test_metrics'][result['selected_model']]
    lines+=['', f"Frozen winner confusion matrix (Severe Flood is positive): TN={win['tn']}, FP={win['fp']}, FN={win['fn']}, TP={win['tp']}. These false positives mean a recorded Flood classified as Severe Flood, not a false flood-occurrence alert.",
      '', 'Validation results:', '', '| Model | Threshold | F1 | PR-AUC/AP |','|---|---:|---:|---:|']
    for name,details in selection['families'].items():
        s=details['validation'];lines.append(f"| {name} | {s['threshold']:.3f} | {s['f1']:.4f} | {s['pr_auc_ap']:.4f} |")
    static=result['test_metrics']['xgb_static'];rain=result['test_metrics']['xgb_rainfall']
    always=result['test_metrics']['always_severe_baseline']
    seen=result['test_subgroups_for_frozen_winner']['gauges_seen_in_training']
    unseen=result['test_subgroups_for_frozen_winner']['gauges_not_seen_in_training']
    lines+=['','## Critical result','',
      f"Rainfall improves observed test F1 from {static['f1']:.4f} to {rain['f1']:.4f} and average precision from {static['pr_auc_ap']:.4f} to {rain['pr_auc_ap']:.4f}. It raises recall from {static['recall']:.4f} to {rain['recall']:.4f}, but false severe classifications increase from {static['fp']} to {rain['fp']}. These are validation-selected operating thresholds, not thresholds changed after seeing test labels.",
      f"Both XGBoost F1 values are below the always-severe baseline ({always['f1']:.4f}). They rank cases better than constant scores by average precision, but calibration and operating-point usefulness are weak. The constant training-prevalence baseline also has a slightly lower Brier error than either XGBoost model. This is not evidence of a dependable warning system.",
      f"The selected static model has ROC-AUC {seen['roc_auc']:.4f} for {seen['n']} later events at gauges present in training, versus {unseen['roc_auc']:.4f} for {unseen['n']} events at new gauges. This diagnostic exposes poor transfer; it is not used to remove gauges or change the final test population.",
      'No model is promoted to the application. The validation-selected static version remains the frozen experimental selection. The rainfall version is retained for transparent comparison; choosing it as a new winner solely from test results would invalidate selection.']
    lines+=['', 'Paired 95% gauge-cluster bootstrap intervals for rainfall minus static (500 replicates):',
      json.dumps(result['paired_gauge_bootstrap_95ci_rainfall_minus_static']),
      'An interval crossing zero does not establish a reliable improvement. Model choice is not changed using these test intervals.',
      '', '## Additional inventory files','',
      'India Flood Inventory has 6,876 reported-event rows. District flooded area and impact files each contain 732 rows; DFSI has 744 rows. These are different observational units and labels from the gauge-event database. They are retained separately. District whole-period impacts cannot safely be joined as pre-event predictors, and missing inventory reports are not verified no-flood days.',
      '', '## Interpretation and limits','',protocol['occurrence_limitation'],protocol['static_vintage_limitation'],
      'The chronological test contains both known and new gauges. Gauge transfer is poor in the reported subgroup analysis. Event rows within a gauge are dependent. Class probabilities have not been independently calibrated as warning probabilities.',
      'Static climate/terrain proxies may be predictive of location-specific historical severity. The rainfall model is assessed against the same static inputs and seasonal variables, so the comparison isolates its added antecedent rainfall information.',
      'Peak discharge is available for 3,993 events and could support a separate regression task. It is not a model input here. No rainfall-amount forecast is trained in this experiment.',
      '', '## Artifacts and reproduction','',
      'Source files, merged data, partition manifest, validation candidates, frozen selection, fitted pipelines, XGBoost UBJ models, feature importance and final predictions are saved in this experiment directory. Gain importance describes fitted splits and is not causal attribution.',
      '```powershell',
      '.\\.venv\\Scripts\\python.exe scripts/compare_indofloods_xgboost.py prepare --source C:/Users/Dell/Downloads',
      '.\\.venv\\Scripts\\python.exe scripts/compare_indofloods_xgboost.py select',
      '.\\.venv\\Scripts\\python.exe scripts/compare_indofloods_xgboost.py evaluate','```',
      'The run refuses to overwrite a frozen selection or repeat final evaluation. Use a new versioned experiment for another prespecified study. The existing application model and API are preserved.',
      '', '## Sources','',
      '- INDOFLOODS: https://zenodo.org/records/14584655',
      '- INDOFLOODS paper: https://doi.org/10.1175/BAMS-D-24-0008.1',
      '- Variable definitions: supplied variables_description_indofloods.pdf, especially pages 5 and 12.',
      '- EM-Earth: https://doi.org/10.1175/BAMS-D-21-0106.1',
      '- IFI-Impacts: https://zenodo.org/records/16994648','']
    (RUN/'COMPARISON_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','select','evaluate'])
    parser.add_argument('--source',type=Path,default=Path.home()/'Downloads')
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.source)
    elif args.stage=='select':select()
    else:evaluate()
