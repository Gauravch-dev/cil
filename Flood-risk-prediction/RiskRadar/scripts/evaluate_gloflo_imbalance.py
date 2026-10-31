"""Compare training-only imbalance treatments for reconstructed high discharge.

Run prepare, select, evaluate in order. Never overwrites the application model.
The raw source was previously inspected during forensics; this is a new frozen
modelling protocol, not a previously unseen external confirmation dataset.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score,average_precision_score,balanced_accuracy_score,
    brier_score_loss,confusion_matrix,f1_score,fbeta_score,matthews_corrcoef,
    precision_recall_curve,precision_score,recall_score,roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/gloflo_imbalance_v1'
SEED=20261024
FEATURES=['precipitation_risk_index','drainage_risk_index','annual_mean_temperature',
 'temperature_seasonality','annual_precipitation','curve_number_amcii','drainage_density',
 'basin_relief','ruggedness_number','infiltration_number','climate_type','landcover_type','soil_type']
CATEGORIES=['climate_type','landcover_type','soil_type']
MODEL_SHA='aad88a12f4645f3703c996641a15ce40603f6d5a608918fad4f37496d2f89395'

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str),encoding='utf-8')
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return sha256(path.read_bytes()).hexdigest()

def prepare():
    if (RUN/'selection.json').exists():raise RuntimeError('Selection already frozen. Use a new protocol version.')
    raw_path=ROOT/'data/raw/kuntla_flooddatabase.csv'
    raw=pd.read_csv(raw_path,low_memory=False)
    mapping={'annual_mean_temperature':'Annual Mean Temperature','temperature_seasonality':'Temperature Seasonality',
      'annual_precipitation':'Annual Precipitation','curve_number_amcii':'Curve Number AMCII',
      'drainage_density':'Drainage Density','basin_relief':'Basin Relief',
      'ruggedness_number':'Ruggedness Number','infiltration_number':'Infiltration Number'}
    X=pd.DataFrame({k:raw[v] for k,v in mapping.items()})
    X['precipitation_risk_index']=(np.minimum(raw['Annual Precipitation'],3000)*.4+
      np.minimum(raw['Precipitation of Wettest Month'],1000)*.3+
      np.minimum(raw['Precipitation Seasonality'],100)*.3)
    X['drainage_risk_index']=(np.minimum(raw['Drainage Density'],10)*.5+
      np.minimum(raw['Drainage Texture'],50)*.3+np.minimum(raw['Basin Relief'],5000)*.2)
    for k,v in [('climate_type','Climate Type'),('landcover_type','Landcover Type'),('soil_type','Soil Type')]:
        X[k]=raw[v].fillna('Unspecified').astype(str)
    master=raw[['gsim_code','year','Annual peak discharge']].copy()
    master=master.rename(columns={'Annual peak discharge':'source_peak_discharge'})
    master['source_row_index']=np.arange(len(raw))
    master['numeric_year']=pd.to_numeric(master.year,errors='coerce')
    master['split']='excluded_invalid_year'
    master.loc[master.numeric_year.le(1999),'split']='train'
    master.loc[master.numeric_year.between(2000,2005),'split']='validation'
    master.loc[master.numeric_year.between(2006,2015),'split']='test'
    trainmask=master.split.eq('train')
    cutoff=float(master.loc[trainmask,'source_peak_discharge'].quantile(.95))
    master['target']=master.source_peak_discharge.gt(cutoff).astype(int)
    master=pd.concat([master,X[FEATURES]],axis=1).sort_values(['numeric_year','source_row_index']).reset_index(drop=True)
    (RUN/'data').mkdir(parents=True,exist_ok=True)
    master.to_csv(RUN/'data/event_master.csv',index=False)
    for split in ['train','validation','test']:
        master.loc[master.split.eq(split)].to_csv(RUN/f'data/{split}.csv',index=False)
    master.loc[master.split.str.startswith('excluded')].to_csv(RUN/'data/excluded.csv',index=False)
    master[['source_row_index','gsim_code','numeric_year','split']].to_csv(RUN/'data/split_manifest.csv',index=False)
    protocol={
      'task':'High annual peak discharge relative to training-source 95th percentile; not verified flood occurrence',
      'cutoff':cutoff,'label_definition':'source_peak_discharge > cutoff',
      'cutoff_fit_on':'train only; differs from original reconstructed full-source percentile',
      'chronology':{'train':'1900–1999','validation':'2000–2005','test':'2006–2015'},
      'source_sha256':digest(raw_path),'original_model_sha256':digest(ROOT/'flood_model.pkl'),
      'features':FEATURES,'categorical_features':CATEGORIES,'seed':SEED,
      'preprocessing':'Numeric median and one-hot category vocabulary fitted exclusively on training rows.',
      'imbalance':'Only training arrays may be reweighted, oversampled or undersampled. Validation and test retain source distributions.',
      'selection':'Maximise validation class-1 F1 with validation-optimised threshold; ties use average precision, then precision. Includes no-treatment baseline.',
      'strategies':['untreated','class_weight','random_oversampling','random_undersampling','balanced_oversampling_weight2'],
      'smote_note':'Random oversampling repeats real training rows. No synthetic fractional soil/climate categories or new packages are introduced.',
      'split_hashes':{s:digest(RUN/f'data/{s}.csv') for s in ['train','validation','test']},
      'row_counts':master.split.value_counts().to_dict(),
      'training_class_counts':master.loc[master.split.eq('train'),'target'].value_counts().to_dict(),
      'independence_limit':'GloFlo was previously inspected for source/model reconstruction. Test is held out for fitting/selection in this new protocol, not pristine external data.',
      'static_feature_limit':'Static modern catchment/climate proxies are not historically issued forecasts; station overlap is reported and unseen-station transfer is not assumed.',
      'created_utc':datetime.now(timezone.utc).isoformat(),
      'versions':{'xgboost':xgboost.__version__,'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__},
    }
    assert protocol['original_model_sha256']==MODEL_SHA
    dump(RUN/'protocol.json',protocol)
    print(json.dumps({k:protocol[k] for k in ['row_counts','cutoff','training_class_counts']},indent=2))

def data(split,protocol):
    p=RUN/f'data/{split}.csv'
    assert digest(p)==protocol['split_hashes'][split]
    return pd.read_csv(p,low_memory=False)

def preprocessing():
    return ColumnTransformer([
      ('numeric',SimpleImputer(strategy='median'),[c for c in FEATURES if c not in CATEGORIES]),
      ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATEGORIES)
    ],sparse_threshold=0)

def metrics(y,p,t):
    pred=np.asarray(p)>=t;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    two=len(np.unique(y))==2
    return {'n':len(y),'class_1':int(np.sum(y)),'threshold':float(t),
      'precision':float(precision_score(y,pred,zero_division=0)),
      'recall':float(recall_score(y,pred,zero_division=0)),
      'f1':float(f1_score(y,pred,zero_division=0)),
      'f2':float(fbeta_score(y,pred,beta=2,zero_division=0)),
      'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
      'accuracy':float(accuracy_score(y,pred)),
      'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
      'pr_auc_ap':float(average_precision_score(y,p)) if two else None,
      'roc_auc':float(roc_auc_score(y,p)) if two else None,
      'brier':float(brier_score_loss(y,p)), 'mcc':float(matthews_corrcoef(y,pred)),
      'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
      'false_positive_rate':float(fp/(fp+tn)) if fp+tn else None}

def threshold(y,p):
    precision,recall,thresholds=precision_recall_curve(y,p)
    f1=np.divide(2*precision[:-1]*recall[:-1],precision[:-1]+recall[:-1],
      out=np.zeros_like(thresholds),where=(precision[:-1]+recall[:-1])>0)
    # Every candidate threshold is taken only from validation predictions.
    idx=max(range(len(thresholds)),key=lambda i:(f1[i],precision[i],-abs(float(thresholds[i])-.5)))
    return metrics(y,p,float(thresholds[idx]))

def sampling_indices(y,strategy,level,seed):
    rng=np.random.default_rng(seed);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    if strategy in ['untreated','class_weight']:return np.arange(len(y))
    if strategy in ['random_oversampling','balanced_oversampling_weight2']:
        n=max(len(pos),int(round(level*len(neg))))
        idx=np.concatenate([neg,pos,rng.choice(pos,n-len(pos),replace=True)])
    elif strategy=='random_undersampling':
        n=min(len(neg),int(round(len(pos)/level)))
        idx=np.concatenate([rng.choice(neg,n,replace=False),pos])
    else:raise ValueError(strategy)
    return rng.permutation(idx)

def select():
    if (RUN/'selection.json').exists():raise RuntimeError('Selection frozen. Do not retune this version.')
    protocol=load(RUN/'protocol.json');train=data('train',protocol);val=data('validation',protocol)
    # The test CSV is not opened during training or model selection.
    prep=preprocessing();xt=prep.fit_transform(train[FEATURES]);xv=prep.transform(val[FEATURES])
    y=train.target.to_numpy();yv=val.target.to_numpy();ratio=float((y==0).sum()/(y==1).sum())
    methods={
      'untreated':[1.],
      'class_weight':[float(np.sqrt(ratio)),ratio/2,ratio],
      'random_oversampling':[.25,.5,1.],
      'random_undersampling':[.25,.5,1.],
      'balanced_oversampling_weight2':[1.],
    }
    configs=[
      dict(n_estimators=300,max_depth=3,learning_rate=.05,min_child_weight=10,reg_lambda=10,reg_alpha=1),
      dict(n_estimators=500,max_depth=4,learning_rate=.05,min_child_weight=10,reg_lambda=10,reg_alpha=1),
      dict(n_estimators=300,max_depth=5,learning_rate=.08,min_child_weight=5,reg_lambda=5,reg_alpha=1),
    ]
    (RUN/'models').mkdir(exist_ok=True);candidates=[];selected={}
    for method,levels in methods.items():
        best=None;started=time.perf_counter()
        for level in levels:
            indices=sampling_indices(y,method,level,SEED)
            weight=level if method=='class_weight' else 2. if method=='balanced_oversampling_weight2' else 1.
            for config in configs:
                model=XGBClassifier(**config,objective='binary:logistic',tree_method='hist',
                    eval_metric='logloss',subsample=.85,colsample_bytree=.85,
                    scale_pos_weight=weight,n_jobs=2,random_state=SEED)
                model.fit(xt[indices],y[indices]);p=model.predict_proba(xv)[:,1]
                score=threshold(yv,p)
                item={'strategy':method,'level':level,'class_weight':weight,'config':config,
                  'fitted_rows':len(indices),'fitted_positive_rows':int(y[indices].sum()),'validation':score}
                candidates.append(item)
                key=lambda x:(x['validation']['f1'],x['validation']['pr_auc_ap'],x['validation']['precision'])
                if best is None or key(item)>key(best):best=item;chosen=model;chosen_p=p
        pipeline=Pipeline([('preprocessing',prep),('model',chosen)])
        joblib.dump(pipeline,RUN/f'models/{method}.joblib');chosen.save_model(RUN/f'models/{method}.ubj')
        best['artifact_sha256']=digest(RUN/f'models/{method}.joblib');selected[method]=best
        pd.DataFrame({'source_row_index':val.source_row_index,'target':yv,'score':chosen_p}).to_csv(RUN/f'{method}_validation_predictions.csv',index=False)
        pd.DataFrame({'feature':prep.get_feature_names_out(),'gain_importance':chosen.feature_importances_}).sort_values('gain_importance',ascending=False).to_csv(RUN/f'models/{method}_importance.csv',index=False)
        print(json.dumps({'strategy':method,'seconds':round(time.perf_counter()-started,1),'level':best['level'],'validation':best['validation']}),flush=True)
    winner=max(selected,key=lambda k:(selected[k]['validation']['f1'],selected[k]['validation']['pr_auc_ap'],selected[k]['validation']['precision']))
    dump(RUN/'validation_candidates.json',candidates)
    selection={'selected_strategy':winner,'families':selected,'training_prevalence':float(y.mean()),
      'candidate_fits':len(candidates),'test_used_for_selection':False,
      'protocol_sha256':digest(RUN/'protocol.json'),'created_utc':datetime.now(timezone.utc).isoformat()}
    dump(RUN/'selection.json',selection)
    assert digest(ROOT/'flood_model.pkl')==MODEL_SHA
    print('FROZEN SELECTION',winner,flush=True)

def evaluate():
    if (RUN/'final_results.json').exists():raise RuntimeError('Final evaluation already exists. Do not retune against it.')
    protocol=load(RUN/'protocol.json');selection=load(RUN/'selection.json')
    assert digest(RUN/'protocol.json')==selection['protocol_sha256']
    test=data('test',protocol);train=data('train',protocol);y=test.target.to_numpy()
    preds=test[['source_row_index','gsim_code','numeric_year','source_peak_discharge','target']].copy()
    results={};curves={}
    for family,s in selection['families'].items():
        path=RUN/f'models/{family}.joblib';assert digest(path)==s['artifact_sha256']
        pipeline=joblib.load(path);p=pipeline.predict_proba(test[FEATURES])[:,1]
        results[family]=metrics(y,p,s['validation']['threshold'])
        results[family]['default_0_5']=metrics(y,p,.5)
        preds[f'{family}_score']=p
        preds[f'{family}_prediction']=(p>=s['validation']['threshold']).astype(int)
        precision,recall,_=precision_recall_curve(y,p)
        curves[family]={'precision':precision.tolist(),'recall':recall.tolist()}
    results['training_prior']=metrics(y,np.full(len(y),selection['training_prevalence']),.5)
    winner=selection['selected_strategy'];groups={}
    for name,mask in [('seen_stations',test.gsim_code.isin(train.gsim_code)),
                      ('unseen_stations',~test.gsim_code.isin(train.gsim_code))]:
        if mask.any():groups[name]=metrics(y[mask],preds.loc[mask,f'{winner}_score'].to_numpy(),selection['families'][winner]['validation']['threshold'])
    rng=np.random.default_rng(SEED);stations=test.gsim_code.unique()
    indices={s:np.flatnonzero(test.gsim_code.to_numpy()==s) for s in stations};diff=[]
    baseline=preds.untreated_prediction.to_numpy();wp=preds[f'{winner}_prediction'].to_numpy()
    for _ in range(500):
        idx=np.concatenate([indices[s] for s in rng.choice(stations,len(stations),replace=True)])
        if len(np.unique(y[idx]))<2:continue
        diff.append(f1_score(y[idx],wp[idx],zero_division=0)-f1_score(y[idx],baseline[idx],zero_division=0))
    split_summary={}
    for split in ['train','validation','test']:
        part=data(split,protocol);split_summary[split]={'n':len(part),'positive':int(part.target.sum()),
          'negative':int(part.target.eq(0).sum()),'positive_percent':float(part.target.mean()*100),
          'stations':int(part.gsim_code.nunique())}
    result={'target':protocol['task'],'cutoff':protocol['cutoff'],'selected_strategy':winner,
      'test_metrics':results,'splits':split_summary,'selected_subgroups':groups,
      'gauge_bootstrap_95ci_f1_difference_vs_untreated':np.quantile(diff,[.025,.975]).tolist(),
      'selection_sha256':digest(RUN/'selection.json'),'test_fitting_or_resampling':False,
      'final_evaluation_utc':datetime.now(timezone.utc).isoformat(),'original_application_model_unchanged':digest(ROOT/'flood_model.pkl')==MODEL_SHA,
      'previously_inspected_source_not_external_validation':True}
    preds.to_csv(RUN/'test_predictions.csv',index=False);dump(RUN/'plot_data.json',curves)
    dump(RUN/'final_results.json',result)
    pd.DataFrame([{'strategy':k,**{x:v for x,v in s.items() if x!='default_0_5'}} for k,s in results.items()]).to_csv(RUN/'test_comparison.csv',index=False)
    write_report(protocol,selection,result)
    print(json.dumps(result,indent=2))

def write_report(protocol,selection,result):
    lines=['# High discharge imbalance comparison','',
      'This is a new chronological comparison of imbalance treatments using GloFlo. The positive class means high annual peak discharge, not verified flood occurrence. The raw source was already inspected during forensics; this is not an external blind validation.',
      '', '## Target and chronology','',
      f"The label is annual peak discharge above the **training-only** 95th percentile: {protocol['cutoff']:.6f} in source units. This differs intentionally from the original reconstructed full-source percentile of 1,145.198. It prevents fitting the target cutoff to future values.",
      '| Split | Years | Rows | High discharge | Other | Positive share |',
      '|---|---|---:|---:|---:|---:|']
    for split,s in result['splits'].items():lines.append(f"| {split} | {protocol['chronology'][split]} | {s['n']} | {s['positive']} | {s['negative']} | {s['positive_percent']:.2f}% |")
    lines+=['','Twenty-two records with unusable source years are excluded before fitting. Periods, labels and feature lists were frozen before this new model selection. No new final-test scores or labels were used for tuning.',
      '', '## Training treatments','',
      'All models share the same 13 engineered inputs and three model configurations. Numeric medians and category encodings are fitted on training data only. Hyperparameters and decision thresholds are selected by validation class-1 F1, then AP and precision. Validation/test retain their original distributions.',
      '- Untreated XGBoost: no class weights or resampling; its threshold is also validation-tuned for a fair comparator.',
      '- Class weighting: sqrt(training imbalance ratio), half the ratio, or the full ratio.',
      '- Random oversampling: repeat real minority training rows to minority/majority ratios 0.25, 0.5 or 1.',
      '- Random undersampling: reduce majority training rows to the same ratios.',
      '- Balanced oversampling plus weight 2: a combined-treatment control inspired by the original artifact; it is not an exact SMOTE reproduction.',
      'No synthetic interpolation of encoded soil/climate categories is used. No resampling or class weighting is applied to validation or test data.',
      '',f"**Frozen validation winner: {selection['selected_strategy']}**, selected from {selection['candidate_fits']} fitted candidates before final evaluation.",
      '', '| Strategy | Validation F1 | Validation AP | Threshold |', '|---|---:|---:|---:|']
    for method,s in selection['families'].items():
        v=s['validation'];lines.append(f"| {method} | {v['f1']:.4f} | {v['pr_auc_ap']:.4f} | {v['threshold']:.4f} |")
    lines+=['','## Chronological test results','',
      '| Strategy | Precision | Recall | F1 | F2 | AP | ROC-AUC | Accuracy | FP | FN |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method,s in result['test_metrics'].items():
        lines.append('| '+method+' | '+' | '.join(f"{s[k]:.4f}" for k in ['precision','recall','f1','f2','pr_auc_ap','roc_auc','accuracy'])+f" | {s['fp']} | {s['fn']} |")
    selected=result['test_metrics'][result['selected_strategy']];base=result['test_metrics']['untreated']
    lines+=['', '## Interpretation','',
      f"At validation-selected thresholds, the selected model changes class-1 F1 from {base['f1']:.4f} to {selected['f1']:.4f}, precision from {base['precision']:.4f} to {selected['precision']:.4f}, and recall from {base['recall']:.4f} to {selected['recall']:.4f}. False positives change from {base['fp']} to {selected['fp']}; false negatives from {base['fn']} to {selected['fn']}.",
      f"Ranking by average precision decreases from {base['pr_auc_ap']:.4f} to {selected['pr_auc_ap']:.4f}. Brier score increases from {base['brier']:.4f} to {selected['brier']:.4f}, indicating worse probability calibration. Oversampling changes the score scale; these scores are not calibrated flood probabilities.",
      f"At the default 0.5 threshold, the combined treatment would produce {selected['default_0_5']['fp']} false positives, compared with {selected['fp']} at its validation-selected threshold. Threshold selection is part of the intervention, separate from balancing; the matched untreated comparator also has a validation-tuned threshold.",
      f"The 95% station-cluster bootstrap interval for the paired F1 difference is {result['gauge_bootstrap_95ci_f1_difference_vs_untreated']}. An interval spanning zero does not establish reliable improvement. The winner is not changed based on final results.",
      'Accuracy alone is misleading for a rare class. The table explicitly reports positive-class F1 and recall; it does not substitute the majority-class F1, which explained the old 0.984 headline.',
      'Original reconstructed scores (96.89% accuracy, class-1 F1 0.717) use another split and full-source cutoff. They are not a valid before/after comparator for these chronological results. The untreated model above is the matched baseline.',
      '', '## Limits and artifacts','',protocol['independence_limit'],protocol['static_feature_limit'],
      'Annual peak-flow records and modern static attributes do not establish a causal weather forecast or an operational flood-warning model. New-station subgroup metrics are saved separately. Treating repeated records as independent locations would overstate generalisation.',
      'All selected model pipelines, category vocabularies, imputers, UBJ models, feature importance, partition hashes, prediction records and parameters are preserved in this experiment. The old API/model are unchanged because their display semantics should not silently be changed to a high-discharge label.',
      '', '## Reproduce','', '```powershell',
      '.\\.venv\\Scripts\\python.exe scripts/evaluate_gloflo_imbalance.py prepare',
      '.\\.venv\\Scripts\\python.exe scripts/evaluate_gloflo_imbalance.py select',
      '.\\.venv\\Scripts\\python.exe scripts/evaluate_gloflo_imbalance.py evaluate', '```',
      'Frozen selection and final evaluation cannot be overwritten by these commands. Use a new prespecified experiment version for another study.',
      '', 'Source: GloFlo https://zenodo.org/records/7158028. Original model reconstruction: ../../evaluation/upstream_forensics/PROVENANCE_FINDINGS.md.','']
    (RUN/'IMBALANCE_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','select','evaluate'])
    args=parser.parse_args()
    {'prepare':prepare,'select':select,'evaluate':evaluate}[args.stage]()
