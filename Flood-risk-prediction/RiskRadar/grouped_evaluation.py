"""Leakage-controlled evaluation of the manual-input annual discharge proxy.

The original deployed model and its encoders are deliberately never loaded.
Prepare, select and evaluate are separate stages with persisted evidence gates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
import xgboost
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parent
DEFAULT_RUN=ROOT/'experiments/gloflo_grouped_evaluation_v1'
SEED=20261025
SOURCE_SHA='20a844f84f78729cf4361f3f13daa3a79626f67c2e96aeb0f1a3b872bfd1a45a'
NUMERIC_MAP={
    'annual_mean_temperature':'Annual Mean Temperature',
    'temperature_seasonality':'Temperature Seasonality',
    'annual_precipitation':'Annual Precipitation',
    'curve_number_amcii':'Curve Number AMCII',
    'drainage_density':'Drainage Density','basin_relief':'Basin Relief',
    'ruggedness_number':'Ruggedness Number','infiltration_number':'Infiltration Number',
}
CATEGORY_MAP={'climate_type':'Climate Type','landcover_type':'Landcover Type','soil_type':'Soil Type'}
NUMERIC=['precipitation_risk_index','drainage_risk_index',*NUMERIC_MAP]
CATEGORIES=list(CATEGORY_MAP)
FEATURES=NUMERIC+CATEGORIES
RAIN_ONLY=['annual_precipitation','precipitation_risk_index']
RAW_INPUTS=list(NUMERIC_MAP.values())+['Precipitation of Wettest Month','Precipitation Seasonality','Drainage Texture']+list(CATEGORY_MAP.values())

def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def features(raw):
    """Fixed inference equations; no data-fitted transformations here."""
    x=pd.DataFrame({key:pd.to_numeric(raw[col],errors='coerce') for key,col in NUMERIC_MAP.items()})
    x['precipitation_risk_index']=(raw['Annual Precipitation'].clip(upper=3000)*.4+
        raw['Precipitation of Wettest Month'].clip(upper=1000)*.3+
        raw['Precipitation Seasonality'].clip(upper=100)*.3)
    x['drainage_risk_index']=(raw['Drainage Density'].clip(upper=10)*.5+
        raw['Drainage Texture'].clip(upper=50)*.3+raw['Basin Relief'].clip(upper=5000)*.2)
    for key,col in CATEGORY_MAP.items():x[key]=raw[col].fillna('__MISSING__').astype(str)
    x[NUMERIC]=x[NUMERIC].replace([np.inf,-np.inf],np.nan)
    return x[FEATURES]

def group_records(raw):
    """Keep station IDs, shared coordinates and identical input profiles together.

    Components are built without discharge, target or year. This also blocks
    identical static area inputs from reappearing under another station ID.
    """
    n=len(raw);parent=np.arange(n)
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parent[max(a,b)]=min(a,b)
    coordinate_keys=list(zip(raw.latitude.astype(float),raw.longitude.astype(float)))
    profile_keys=pd.util.hash_pandas_object(raw[RAW_INPUTS].astype(str),index=False).to_numpy()
    for keys in [raw.gsim_code.astype(str).tolist(),coordinate_keys,profile_keys]:
        seen={}
        for i,key in enumerate(keys):
            if key in seen:union(i,seen[key])
            else:seen[key]=i
    # Stable IDs based on sorted station identifiers in each component.
    roots=[find(i) for i in range(n)];station_sets={}
    for root,station in zip(roots,raw.gsim_code.astype(str)):
        station_sets.setdefault(root,set()).add(station)
    names={root:sha256('|'.join(sorted(ids)).encode()).hexdigest()[:20] for root,ids in station_sets.items()}
    return pd.Series([names[root] for root in roots],index=raw.index,name='group_id')

def split_groups(groups,seed=SEED):
    """60/20/20 by groups, with no target stratification or seed search."""
    index=np.arange(len(groups))
    remaining,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(index,groups=groups))
    tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+1).split(remaining,groups=np.asarray(groups)[remaining]))
    return {'train':remaining[tr],'validation':remaining[va],'test':test}

def configurations():
    configs=[]
    for weighted in [False,True]:
        suffix='weighted' if weighted else 'plain'
        configs.extend([
            {'name':f'logistic_{suffix}','family':'logistic','weighted':weighted,
             'params':{'C':1.0,'solver':'lbfgs','max_iter':2000,'random_state':SEED}},
            {'name':f'extra_trees_{suffix}','family':'extra_trees','weighted':weighted,
             'params':{'n_estimators':400,'max_depth':None,'min_samples_leaf':5,
                       'max_features':.8,'n_jobs':2,'random_state':SEED}},
            {'name':f'hist_gradient_{suffix}','family':'hist_gradient','weighted':weighted,
             'params':{'max_iter':250,'max_leaf_nodes':15,'max_depth':None,'learning_rate':.05,
                       'l2_regularization':1.0,'early_stopping':False,'random_state':SEED}},
            {'name':f'xgb_regularized_{suffix}','family':'xgboost','weighted':weighted,
             'params':{'n_estimators':300,'max_depth':3,'learning_rate':.05,'subsample':.8,
                       'colsample_bytree':.8,'reg_alpha':1.0,'reg_lambda':10.0}},
            {'name':f'xgb_original_shape_{suffix}','family':'xgboost','weighted':weighted,
             'params':{'n_estimators':300,'max_depth':8,'learning_rate':.2,'subsample':1.0,
                       'colsample_bytree':.8,'reg_alpha':1.0,'reg_lambda':.1}},
        ])
    configs.append({'name':'rainfall_only_logistic','family':'logistic','weighted':True,
        'feature_view':'rainfall_only','params':{'C':1.0,'solver':'lbfgs','max_iter':2000,'random_state':SEED}})
    return configs

def prepare(run):
    if run.exists():raise FileExistsError(f'{run} already exists; frozen runs cannot be overwritten.')
    source=ROOT/'data/raw/kuntla_flooddatabase.csv'
    assert digest(source)==SOURCE_SHA,'Source differs from the registered dataset.'
    raw=pd.read_csv(source,low_memory=False)
    assert raw.gsim_code.notna().all() and raw[['latitude','longitude']].notna().all().all()
    q=pd.to_numeric(raw['Annual peak discharge'],errors='coerce')
    assert np.isfinite(q).all() and q.ge(0).all(),'Invalid discharge must be resolved before splitting.'
    groups=group_records(raw);parts=split_groups(groups)
    assert not any(set(groups.iloc[parts[a]])&set(groups.iloc[parts[b]])
                   for a,b in [('train','validation'),('train','test'),('validation','test')])
    cutoff=float(q.iloc[parts['train']].quantile(.95))
    run.mkdir(parents=True);(run/'data').mkdir();(run/'models').mkdir()
    frame=features(raw)
    frame['source_row']=np.arange(len(raw));frame['station_id']=raw.gsim_code.astype(str)
    frame['group_id']=groups;frame['source_year']=raw.year
    frame['source_discharge']=q
    mapping=[];files={};counts={}
    for name,index in parts.items():
        subset=frame.iloc[index].copy().sort_values('source_row')
        path=run/'data'/f'{name}.csv';subset.to_csv(path,index=False)
        files[name]=digest(path)
        counts[name]={'rows':len(subset),'stations':int(subset.station_id.nunique()),'groups':int(subset.group_id.nunique())}
        mapping.append(subset[['source_row','station_id','group_id']].assign(split=name))
    pd.concat(mapping).sort_values('source_row').to_csv(run/'split_manifest.csv',index=False)
    # Protocol is saved before fitting or inspecting validation/test outcomes.
    protocol={
        'created_at':utc(),'task':'Annual high-discharge susceptibility proxy using original manual-input features',
        'target':{'column':'source_discharge','positive_rule':'value > training-row 95th percentile',
                  'percentile':.95,'cutoff':cutoff,'fit_partition':'train'},
        'source_sha256':SOURCE_SHA,'original_model_sha256':digest(ROOT/'flood_model.pkl'),
        'seed':SEED,'split':'60/20/20 groups; not chronological; no outcome stratification or seed search',
        'group_definition':'Connected station IDs, exactly shared coordinates or identical raw manual-input profiles',
        'features':FEATURES,'raw_inputs':RAW_INPUTS,'excluded_inputs':['discharge','station','coordinates','year','country'],
        'sample_unit':'catchment-year record','estimand':'generalisation to held-out station/input groups in the supplied archive',
        'source_previously_analysed':True,'independent_external_validation':False,
        'preprocessing':'Train-only median imputation, scaling and one-hot categories; unknown categories ignored',
        'imbalance':'Unweighted and train-class-weighted candidates; no validation/test sampling',
        'class_weight_rule':'n_train_negative/n_train_positive applied only to positive training rows',
        'selection_rule':'Maximum validation positive-class F1; tie by average precision, then recall, then name',
        'threshold_rule':'Maximum validation F1 over unique predicted scores; ties prefer higher threshold',
        'no_test_tuning':True,'refit_after_selection':False,'test_policy':'Evaluate once after selection.json is frozen',
        'configs':configurations(),'baselines':['always_other','training_prevalence_constant','annual_rainfall_p95_rule'],
        'counts_without_test_labels':counts,'split_sha256':files,
        'confidence_intervals':'1000 bootstrap resamples of independent station/input groups, fixed selected threshold',
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                    'sklearn':sklearn.__version__,'xgboost':xgboost.__version__},
    }
    save_json(run/'protocol.json',protocol)
    print(json.dumps({'stage':'prepared','counts':counts,'training_only_cutoff':cutoff}),flush=True)

def target(frame,cutoff):return frame.source_discharge.gt(cutoff).to_numpy(dtype=np.int8)

def threshold_from_validation(y,probability):
    precision,recall,thresholds=precision_recall_curve(y,probability)
    values=np.divide(2*precision[:-1]*recall[:-1],precision[:-1]+recall[:-1],
                     out=np.zeros_like(thresholds),where=(precision[:-1]+recall[:-1])>0)
    best=np.flatnonzero(np.isclose(values,values.max(),rtol=0,atol=1e-12))[-1]
    return float(thresholds[best])

def summarize(y,probability,threshold=.5,weights=None):
    prediction=(np.asarray(probability)>=threshold).astype(int)
    report=classification_report(y,prediction,labels=[0,1],output_dict=True,zero_division=0,sample_weight=weights)
    tn,fp,fn,tp=confusion_matrix(y,prediction,labels=[0,1],sample_weight=weights).ravel()
    p=report['1']['precision'];r=report['1']['recall']
    return {'accuracy':float(np.average(y==prediction,weights=weights)),
        'precision':p,'recall':r,'f1':report['1']['f1-score'],
        'f2':float(5*p*r/(4*p+r)) if 4*p+r else 0.,
        'balanced_accuracy':float((report['0']['recall']+r)/2),
        'macro_precision':report['macro avg']['precision'],'macro_recall':report['macro avg']['recall'],
        'macro_f1':report['macro avg']['f1-score'],
        'average_precision':float(average_precision_score(y,probability,sample_weight=weights)),
        'roc_auc':float(roc_auc_score(y,probability,sample_weight=weights)),
        'brier_score':float(brier_score_loss(y,probability,sample_weight=weights)),
        'threshold':float(threshold),'confusion_matrix':[[float(tn),float(fp)],[float(fn),float(tp)]],
        'positive_count':int(np.sum(y)),'rows':len(y),'predicted_positive_count':int(prediction.sum()),
        'class_report':report}

def pipeline(config):
    rain_only=config.get('feature_view')=='rainfall_only'
    num=RAIN_ONLY if rain_only else NUMERIC
    numeric=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())])
    transformers=[('numeric',numeric,num)]
    if not rain_only:transformers.append(('categories',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATEGORIES))
    preprocess=ColumnTransformer(transformers,remainder='drop',verbose_feature_names_out=True)
    family=config['family'];params=config['params']
    if family=='logistic':model=LogisticRegression(**params)
    elif family=='extra_trees':model=ExtraTreesClassifier(**params)
    elif family=='hist_gradient':model=HistGradientBoostingClassifier(**params)
    else:model=XGBClassifier(**params,objective='binary:logistic',eval_metric='logloss',tree_method='hist',
                            random_state=SEED,n_jobs=2,device='cpu')
    return Pipeline([('preprocess',preprocess),('model',model)])

def verify_partition(run,protocol,name):
    path=run/'data'/f'{name}.csv'
    assert digest(path)==protocol['split_sha256'][name],f'{name} data changed.'
    return pd.read_csv(path,keep_default_na=False,na_values=[''])

def select(run):
    if (run/'selection.json').exists() or (run/'final_results.json').exists():
        raise FileExistsError('Selection is already frozen; no repeat tuning is allowed in this run.')
    protocol=read_json(run/'protocol.json');protocol_sha=digest(run/'protocol.json')
    train=verify_partition(run,protocol,'train');validation=verify_partition(run,protocol,'validation')
    cutoff=protocol['target']['cutoff'];y=target(train,cutoff);yv=target(validation,cutoff)
    assert len(np.unique(y))==len(np.unique(yv))==2
    ratio=float(np.sum(y==0)/np.sum(y==1));records=[]
    val_predictions=validation[['source_row','station_id','group_id']].copy();val_predictions['target']=yv
    for config in protocol['configs']:
        name=config['name'];model=pipeline(config)
        weight=np.where(y==1,ratio,1.) if config['weighted'] else np.ones(len(y))
        with threadpool_limits(limits=2):
            model.fit(train[FEATURES],y,model__sample_weight=weight)
            probability=model.predict_proba(validation[FEATURES])[:,1]
        threshold=threshold_from_validation(yv,probability)
        metrics=summarize(yv,probability,threshold)
        model_path=run/'models'/f'{name}.joblib';joblib.dump(model,model_path)
        records.append({'name':name,'family':config['family'],'feature_view':config.get('feature_view','manual13'),
            'training_positive_weight':ratio if config['weighted'] else 1.,'validation':metrics,
            'model_sha256':digest(model_path)})
        val_predictions[name]=probability
        save_json(run/'validation_progress.json',records)
        print(json.dumps({'candidate':name,'validation_f1':metrics['f1'],'precision':metrics['precision'],
                          'recall':metrics['recall'],'threshold':threshold}),flush=True)
    winner=sorted(records,key=lambda c:(-c['validation']['f1'],-c['validation']['average_precision'],
                                      -c['validation']['recall'],c['name']))[0]
    val_predictions.to_csv(run/'validation_predictions.csv',index=False)
    selection={'frozen_at':utc(),'protocol_sha256':protocol_sha,'selected_name':winner['name'],
        'threshold':winner['validation']['threshold'],'model_sha256':winner['model_sha256'],
        'training_rows':len(train),'training_positive':int(y.sum()),'validation_rows':len(validation),
        'validation_positive':int(yv.sum()),'training_prevalence':float(y.mean()),
        'annual_rainfall_baseline_cutoff':float(train.annual_precipitation.quantile(.95)),
        'candidates':records,'test_loaded_by_selection':False,'refitted_on_validation':False}
    save_json(run/'selection.json',selection)
    print(json.dumps({'stage':'selection_frozen','winner':winner['name'],'threshold':selection['threshold']}),flush=True)

def bootstrap_confusion(y,prediction,groups,repetitions=1000):
    unique,codes=np.unique(groups,return_inverse=True)
    bucket=(y.astype(int)*2+prediction.astype(int))
    counts=np.zeros((len(unique),4),dtype=float)
    np.add.at(counts,(codes,bucket),1)
    rng=np.random.default_rng(SEED+2);scores=[]
    for _ in range(repetitions):
        sample=counts[rng.integers(0,len(unique),len(unique))].sum(axis=0)
        tn,fp,fn,tp=sample
        precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        scores.append([float((tn+tp)/sample.sum()),precision,recall,f1])
    values=np.quantile(scores,[.025,.975],axis=0)
    return {key:{'lower':float(values[0,i]),'upper':float(values[1,i])}
            for i,key in enumerate(['accuracy','precision','recall','f1'])}

def evaluate(run):
    if (run/'final_results.json').exists() or (run/'test_opened.json').exists():
        raise FileExistsError('The final test has already been opened. Do not retune or rerun this evaluation.')
    protocol=read_json(run/'protocol.json');selection=read_json(run/'selection.json')
    assert digest(run/'protocol.json')==selection['protocol_sha256'],'Protocol changed after selection.'
    selection_sha=digest(run/'selection.json')
    # Write an opening receipt before access; interruptions cannot silently reset the test gate.
    save_json(run/'test_opened.json',{'opened_at':utc(),'selection_sha256':selection_sha,'protocol_sha256':selection['protocol_sha256']})
    test=verify_partition(run,protocol,'test');y=target(test,protocol['target']['cutoff'])
    assert len(np.unique(y))==2
    manifest=pd.read_csv(run/'split_manifest.csv')
    overlap={}
    for column in ['station_id','group_id']:
        overlap[column]={}
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            shared=set(manifest.loc[manifest.split.eq(a),column])&set(manifest.loc[manifest.split.eq(b),column])
            overlap[column][f'{a}_{b}']=len(shared);assert not shared
    results={};predictions=test[['source_row','station_id','group_id','source_year','source_discharge']].copy()
    predictions['target']=y
    for candidate in selection['candidates']:
        name=candidate['name'];path=run/'models'/f'{name}.joblib'
        assert digest(path)==candidate['model_sha256']
        model=joblib.load(path)
        with threadpool_limits(limits=2):probability=model.predict_proba(test[FEATURES])[:,1]
        threshold=candidate['validation']['threshold']
        results[name]={'validation':candidate['validation'],'test':summarize(y,probability,threshold),
                       'test_at_default_0_5':summarize(y,probability,.5)}
        predictions[name+'_score']=probability;predictions[name+'_class']=(probability>=threshold).astype(int)
    baselines={
        'always_other':np.zeros(len(test)),
        'training_prevalence_constant':np.repeat(selection['training_prevalence'],len(test)),
        'annual_rainfall_p95_rule':test.annual_precipitation.gt(selection['annual_rainfall_baseline_cutoff']).to_numpy(dtype=float)}
    for name,probability in baselines.items():
        results[name]={'test':summarize(y,probability,.5)}
        predictions[name+'_score']=probability
    chosen=selection['selected_name'];probability=predictions[chosen+'_score'].to_numpy()
    prediction=predictions[chosen+'_class'].to_numpy()
    ci=bootstrap_confusion(y,prediction,test.group_id.to_numpy())
    station_weight=(1/test.groupby('station_id').station_id.transform('size')).to_numpy()
    equal_station=summarize(y,probability,selection['threshold'],station_weight)
    predictions.to_csv(run/'test_predictions.csv',index=False)
    output={'evaluated_at':utc(),'selected_before_test':chosen,'selection_sha256':selection_sha,
        'protocol_sha256':selection['protocol_sha256'],'training_only_target_cutoff':protocol['target']['cutoff'],
        'test_rows':len(test),'test_positive':int(y.sum()),'test_negative':int(np.sum(y==0)),
        'test_stations':int(test.station_id.nunique()),'test_groups':int(test.group_id.nunique()),
        'overlap_counts':overlap,'metrics':results,'selected_95pct_group_bootstrap_ci':ci,
        'selected_equal_station_weighted_metrics':equal_station,
        'test_label_prevalence':float(y.mean()),'number_of_test_openings':1,
        'original_deployed_model_unchanged':digest(ROOT/'flood_model.pkl')==protocol['original_model_sha256'],
        'test_predictions_sha256':digest(run/'test_predictions.csv'),
        'interpretation':'Newly fitted models on disjoint station/input groups; previously analysed source, not prospective external validation. Target remains annual high discharge, not verified flood occurrence.'}
    save_json(run/'final_results.json',output)
    joblib.dump({'pipeline':joblib.load(run/'models'/f'{chosen}.joblib'),
        'threshold':selection['threshold'],'target_cutoff':protocol['target']['cutoff'],
        'feature_columns':FEATURES,'target':'annual high-discharge proxy','run':str(run)},run/'evaluated_model_bundle.joblib')
    write_report(run,protocol,selection,output)
    print(json.dumps({'stage':'final_test_complete','selected_model':chosen,
        'test':results[chosen]['test'],'confidence_intervals':ci},indent=2),flush=True)

def write_report(run,protocol,selection,output):
    selected=output['selected_before_test'];m=output['metrics'][selected]['test']
    lines=['# Station-separated evaluation of annual high-discharge susceptibility','',
        'This study retrains classifiers using the existing manual-input feature representation. It evaluates transfer to station/input groups excluded from fitting and model selection. The deployed website model is unchanged.','',
        '## Registered method','',
        f"- Source: GloFlo, 45,932 catchment-year records. Source SHA256: `{protocol['source_sha256']}`.",
        '- Target: annual peak discharge above the 95th percentile of training records only.',
        f"- Training-derived target threshold: {protocol['target']['cutoff']:.6f} source units.",
        '- The 60/20/20 allocation uses independent station/input groups and one fixed seed. Stations sharing exact coordinates or identical manual input profiles stay together. No labels are used to choose the split.',
        '- The split is geographical/grouped, not chronological. It evaluates previously excluded station groups within this archive; it is not a future-date forecast.',
        '- Median imputation, scaling and category vocabularies are fitted on training rows only. Test-only categories are ignored by one-hot encoding.',
        '- Positive training weights use the training negative/positive ratio. Validation and test rows retain their natural distributions. No test resampling is used.',
        '- Candidate settings are registered in protocol.json. Model and score threshold selection use validation positive-class F1 only, with the registered tie breakers. No validation refit is performed.',
        f"- Selection was frozen at {selection['frozen_at']} before the test opening. The test opening receipt records selection SHA256 `{output['selection_sha256']}`.",
        '- The full source has been analysed in earlier studies. These groups are held out from the new fits; this is not a pristine external dataset or a proof of zero bias.','',
        '## Partition sizes','', '| Partition | Rows | Stations | Groups | Positive cases |','|---|---:|---:|---:|---:|']
    for split in ['train','validation','test']:
        c=protocol['counts_without_test_labels'][split]
        positive=output['test_positive'] if split=='test' else selection['training_positive' if split=='train' else 'validation_positive']
        lines.append(f"| {split} | {c['rows']:,} | {c['stations']:,} | {c['groups']:,} | {positive:,} |")
    lines+=['','Station and group overlap is zero between every pair of partitions.','',
        '## Candidate comparison','',
        '| Method | Validation F1 | Test accuracy | Positive precision | Positive recall | Positive F1 | Test AP |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,row in output['metrics'].items():
        t=row['test'];v=f"{row['validation']['f1']:.4f}" if 'validation' in row else '—'
        lines.append(f"| {name}{' **selected**' if name==selected else ''} | {v} | {t['accuracy']:.2%} | {t['precision']:.2%} | {t['recall']:.2%} | {t['f1']:.2%} | {t['average_precision']:.4f} |")
    lines+=['','The selected model is fixed by validation. Test results for other registered candidates are disclosed for comparison and are not used to replace it.','',
        '## Selected model evaluation','',f"Selected model: **{selected}**, with validation-chosen threshold **{selection['threshold']:.6f}**.",'',
        '| Metric | Value | 95% group bootstrap interval |','|---|---:|---|']
    for key in ['accuracy','precision','recall','f1']:
        ci=output['selected_95pct_group_bootstrap_ci'][key]
        lines.append(f"| {key} | {m[key]:.2%} | {ci['lower']:.2%}–{ci['upper']:.2%} |")
    lines+=['',f"Balanced accuracy: {m['balanced_accuracy']:.2%}; macro precision: {m['macro_precision']:.2%}; macro recall: {m['macro_recall']:.2%}; macro F1: {m['macro_f1']:.2%}.",
        f"ROC-AUC: {m['roc_auc']:.4f}; average precision: {m['average_precision']:.4f}; F2: {m['f2']:.4f}; Brier score: {m['brier_score']:.4f}.",'',
        '| Actual / Predicted | Other discharge | High discharge |','|---|---:|---:|']
    for label,values in zip(['Other discharge','High discharge'],m['confusion_matrix']):
        lines.append(f'| {label} | {int(values[0]):,} | {int(values[1]):,} |')
    lines+=['','| Class | Precision | Recall | F1 | Support |','|---|---:|---:|---:|---:|']
    for key,label in [('0','Other discharge'),('1','High discharge')]:
        c=m['class_report'][key];lines.append(f"| {label} | {c['precision']:.2%} | {c['recall']:.2%} | {c['f1-score']:.2%} | {int(c['support']):,} |")
    equal=output['selected_equal_station_weighted_metrics']
    lines+=['',f"Giving each test station equal total weight yields accuracy {equal['accuracy']:.2%}, precision {equal['precision']:.2%}, recall {equal['recall']:.2%} and F1 {equal['f1']:.2%}. This sensitivity analysis reduces the influence of stations with longer records.",'',
        '## Interpretation','',
        'The earlier 96.89% accuracy / 65.64% precision / 79.08% recall / 71.74% F1 values were artifact reconstruction with overlapping stations and a source-wide target threshold. They are not directly comparable to this grouped experiment, which uses freshly fitted models, different held-out records and a training-only target threshold.',
        'Accuracy and class-specific scores are reported together without forcing equality or selecting an averaging method to conceal minority-class errors. Intervals resample whole station/input groups rather than correlated yearly rows. The discharge-percentile label remains a susceptibility proxy; this study does not add observed flood/no-flood labels.','',
        '## Reproduction and evidence','',
        'Run from RiskRadar: `.venv\\Scripts\\python.exe grouped_evaluation.py --stage all` for a new registered run. Existing frozen runs refuse overwrite or repeat test access. Use `--output-dir` to specify a different run only for a separately justified, predeclared experiment; do not search seeds after viewing results.',
        'Files: protocol.json, split_manifest.csv, data/train.csv, data/validation.csv, data/test.csv, selection.json, test_opened.json, validation_predictions.csv, test_predictions.csv, final_results.json and evaluated_model_bundle.joblib.',
        'Method references: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html and https://scikit-learn.org/stable/common_pitfalls.html#data-leakage.']
    (run/'EVALUATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','select','evaluate','all'],default='all')
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_RUN)
    args=parser.parse_args();run=args.output_dir.resolve()
    if args.stage in ['prepare','all']:prepare(run)
    if args.stage in ['select','all']:select(run)
    if args.stage in ['evaluate','all']:evaluate(run)

if __name__=='__main__':main()
