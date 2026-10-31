"""Fit and freeze a rainfall-based indicator on EXISTING rainfall data only."""
from pathlib import Path
import sys
import json
from hashlib import sha256
from datetime import datetime,timezone

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (f1_score,fbeta_score,precision_score,recall_score,confusion_matrix,
                             average_precision_score,roc_auc_score,accuracy_score)
from threadpoolctl import threadpool_limits
from rainfall_risk import SOURCE,RUN,NUMERIC,weather,features,category,classify,baseline_probabilities


def dump(name,value):(RUN/name).write_text(json.dumps(value,indent=2,default=str,allow_nan=False),encoding='utf-8')
def digest(path):return sha256(path.read_bytes()).hexdigest()


def score(y,probability,cutoffs):
    predicted=classify(probability,cutoffs)
    return {'n':len(y),'counts':np.bincount(y,minlength=3).tolist(),
            'macro_f1':float(f1_score(y,predicted,labels=[0,1,2],average='macro',zero_division=0)),
            'accuracy':float(accuracy_score(y,predicted)),
            'high_precision':float(precision_score(y==2,predicted==2,zero_division=0)),
            'high_recall':float(recall_score(y==2,predicted==2,zero_division=0)),
            'high_f1':float(f1_score(y==2,predicted==2,zero_division=0)),
            'high_f2':float(fbeta_score(y==2,predicted==2,beta=2,zero_division=0)),
            'high_average_precision':float(average_precision_score(y==2,probability[:,2])),
            'high_roc_auc':float(roc_auc_score(y==2,probability[:,2])) if (y==2).any() and (y!=2).any() else None,
            'confusion_matrix_low_medium_high':confusion_matrix(y,predicted,labels=[0,1,2]).tolist(),
            'cutoffs':cutoffs}


def tune(y,probability):
    best=None
    for high in [.05,.1,.2,.3,.4,.5,.6,.7,.8]:
        for medium in [.05,.1,.2,.3,.4,.5,.6,.7,.8]:
            m=score(y,probability,{'high':high,'medium':medium})
            if best is None or (m['macro_f1'],m['high_f2'])>(best['macro_f1'],best['high_f2']):best=m
    return best


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'final_results.json').exists() or (RUN/'selection.json').exists():
        raise RuntimeError('Selection/test are frozen. Do not retune this run.')
    data=weather();frame=features(data)
    future=pd.concat([data.rain_mm.shift(-i) for i in [1,2,3]],axis=1)
    frame['target_total']=future.sum(axis=1,min_count=3)
    frame['label_end']=frame.index+pd.Timedelta(days=3)
    frame=frame.loc[frame[NUMERIC].notna().all(axis=1)&frame.target_total.notna()].copy()
    parts={}
    for name,start,end in [('train','2000-01-01','2015-12-31'),('validation','2016-01-01','2019-12-31'),('test','2020-01-01','2025-12-31')]:
        parts[name]=frame.loc[(frame.index>=start)&(frame.label_end<=end)].copy()
    wet=parts['train'].target_total.loc[parts['train'].target_total.gt(0)]
    thresholds={'medium':float(wet.quantile(.90)),'high':float(wet.quantile(.95))}
    for name,part in parts.items():
        part['target']=category(part.target_total,thresholds)
        part.to_csv(RUN/f'{name}.csv')
    train=parts['train'];val=parts['validation']
    seasonal={str(month):((np.bincount(train.loc[train.index.month==month,'target'],minlength=3)+1)/
                          (len(train.loc[train.index.month==month])+3)).tolist() for month in range(1,13)}
    protocol={'target':'Category of sum of the NEXT three daily rainfall amounts; not observed flood occurrence',
              'location':'Existing near-Mumbai ERA5 point, requested 19.1622 N / 73.2544 E',
              'source_file':str(SOURCE),'source_sha256':digest(SOURCE),'timezone':'GMT daily calendar',
              'thresholds':thresholds,'threshold_definition':'Training wet (>0) three-day totals: 90th/95th percentiles',
              'features':NUMERIC,'split':'Train 2000–2015, validation 2016–2019, test 2020–2025; crossing three-day windows purged',
              'selection':'Highest validation macro F1 then High F2; class cutoffs tuned on validation only',
              'sampling':'Natural validation/test; optional balanced training sample weights only',
              'limits':'Retrospective reanalysis experiment; overlapping three-day targets are dependent. No flood-occurrence validation, humidity or dynamic temperature predictors.',
              'split_hashes':{name:digest(RUN/f'{name}.csv') for name in parts},
              'created_utc':datetime.now(timezone.utc).isoformat()}
    dump('protocol.json',protocol)
    candidates=[];saved={};predictions={}
    options=[('logistic_1',LogisticRegression(C=1,max_iter=1200,class_weight='balanced',random_state=20261025),True),
             ('logistic_01',LogisticRegression(C=.1,max_iter=1200,class_weight='balanced',random_state=20261025),True),
             ('extra_trees_10',ExtraTreesClassifier(n_estimators=200,max_depth=12,min_samples_leaf=10,class_weight='balanced',n_jobs=2,random_state=20261025),False),
             ('extra_trees_30',ExtraTreesClassifier(n_estimators=200,max_depth=12,min_samples_leaf=30,class_weight='balanced',n_jobs=2,random_state=20261025),False),
             ('hist_gradient_15',HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=10,early_stopping=False,random_state=20261025),False),
             ('hist_gradient_31',HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=31,min_samples_leaf=40,l2_regularization=20,early_stopping=False,random_state=20261025),False)]
    (RUN/'models').mkdir(exist_ok=True)
    for name,estimator,scale in options:
        steps=[('imputer',SimpleImputer(strategy='median'))]
        if scale:steps.append(('scale',StandardScaler()))
        steps.append(('model',estimator));model=Pipeline(steps)
        kwargs={'model__sample_weight':compute_sample_weight('balanced',train.target)} if name.startswith('hist') else {}
        model.fit(train[NUMERIC],train.target,**kwargs)
        probabilities=model.predict_proba(val[NUMERIC]);m=tune(val.target.to_numpy(),probabilities)
        candidates.append({'name':name,'kind':'ml','validation':m});saved[name]=model
        joblib.dump(model,RUN/f'models/{name}.joblib',compress=3)
        print(name,'validation macro F1',round(m['macro_f1'],4),flush=True)
    context={'thresholds':thresholds,'seasonal':seasonal}
    for name in ['persistence','seasonal','always_low']:
        probability=baseline_probabilities(val,name,context)
        m=tune(val.target.to_numpy(),probability)
        candidates.append({'name':name,'kind':'baseline','validation':m})
    chosen=max(candidates,key=lambda item:(item['validation']['macro_f1'],item['validation']['high_f2']))
    selection={'selected':chosen,'candidates':candidates,'protocol_sha256':digest(RUN/'protocol.json'),
               'test_used_for_selection':False}
    dump('selection.json',selection)
    bundle={**context,'name':chosen['name'],'kind':chosen['kind'],
            'pipeline':saved.get(chosen['name']),'cutoffs':chosen['validation']['cutoffs'],
            'feature_names':NUMERIC,'source_sha256':digest(SOURCE)}
    joblib.dump(bundle,RUN/'forecast_bundle.joblib',compress=3)
    # Test evaluation starts only after selection.json and inference bundle exist.
    test=parts['test'];report={};prediction_frame=test[['target_total','target','label_end']].copy()
    for candidate in candidates:
        name=candidate['name']
        probability=(saved[name].predict_proba(test[NUMERIC]) if candidate['kind']=='ml' else
                     baseline_probabilities(test,name,context))
        report[name]=score(test.target.to_numpy(),probability,candidate['validation']['cutoffs'])
        prediction_frame[name]=classify(probability,candidate['validation']['cutoffs'])
    prediction_frame.to_csv(RUN/'test_predictions.csv')
    frozen={'selected_name':chosen['name'],'selected_kind':chosen['kind'],'test_metrics':report,
            'selection_sha256':digest(RUN/'selection.json'),'bundle_sha256':digest(RUN/'forecast_bundle.joblib'),
            'thresholds':thresholds,'rows':{name:len(part) for name,part in parts.items()},
            'selected_test':report[chosen['name']],'new_flood_occurrence_metrics_claimed':False,
            'evaluated_utc':datetime.now(timezone.utc).isoformat()}
    dump('final_results.json',frozen)
    print(json.dumps(frozen,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
