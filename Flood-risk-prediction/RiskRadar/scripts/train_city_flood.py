"""Freeze a bounded simple classifier study; no test-driven retuning."""
from pathlib import Path
import sys
import json
from hashlib import sha256
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,average_precision_score,roc_auc_score,confusion_matrix
from threadpoolctl import threadpool_limits
from city_flood import RUN,ROOT,FEATURES,HOURLY,RAIN_FILES,WARNING,DANGER,prepare_dataset,predict_manual


def dump(name,value):(RUN/name).write_text(json.dumps(value,indent=2,default=str,allow_nan=False),encoding='utf-8')
def digest(path):return sha256(path.read_bytes()).hexdigest()


def metrics(y,score,threshold):
    y=np.asarray(y,dtype=int);prediction=np.asarray(score)>=threshold
    return dict(n=len(y),positive=int(y.sum()),accuracy=float(accuracy_score(y,prediction)),
        precision=float(precision_score(y,prediction,zero_division=0)),recall=float(recall_score(y,prediction,zero_division=0)),
        f1=float(f1_score(y,prediction,zero_division=0)),
        average_precision=float(average_precision_score(y,score)) if y.any() else None,
        roc_auc=float(roc_auc_score(y,score)) if len(np.unique(y))==2 else None,
        confusion_matrix=confusion_matrix(y,prediction,labels=[0,1]).tolist(),threshold=float(threshold),
        meets_75_percent_all=all(value>=.75 for value in [accuracy_score(y,prediction),
            precision_score(y,prediction,zero_division=0),recall_score(y,prediction,zero_division=0),f1_score(y,prediction,zero_division=0)]))


def main():
    RUN.mkdir(exist_ok=True,parents=True)
    if (RUN/'protocol.json').exists():raise RuntimeError('This study is frozen. Do not overwrite or retune using test results.')
    data=prepare_dataset()
    parts={name:data.loc[data.split.eq(name)].copy() for name in ['train','validation','test']}
    for name,part in parts.items():part.to_csv(RUN/f'{name}.csv',index=False)
    protocol={'place':'Kurundwad, Kolhapur district, Maharashtra; CWC Krishna gauge',
        'target':'Measured river level at or above official warning in next 24h; binary flood-stage risk, not district inundation',
        'warning_m':WARNING,'danger_m':DANGER,'issue_time':'08:00 Asia/Kolkata once per day',
        'features':FEATURES,'splits':{name:{'n':len(part),'positive':int(part.target.sum()),'start':str(part.time.min()),'end':str(part.time.max()),
            'rain_available':int(part.rain_available.sum())} for name,part in parts.items()},
        'selection':'Maximum validation F1, then recall, then precision. Threshold grid 0.05 to 0.95 step 0.01; ML winner selected before any test inference.',
        'models':['balanced Logistic Regression C1','balanced Decision Tree depth3 leaf3','balanced Random Forest 200 trees depth5 leaf3'],
        'baselines':['current-stage persistence','6h positive trend extrapolated for 12h','always below warning'],
        'limits':['Retrospective follow-up on already inspected source years, not a fresh independent test.',
            'Rainfall reports are sparse and nonzero-only; absent reports are unknown. Missingness may reflect observation practices.',
            'Measured warning-stage exceedance is a physical flood-risk proxy, not observed inundation.',
            'High-water days within a flood are dependent; separately report origins currently below warning.',
            'No new datasets; this is a new direct classifier, distinct from previous river regression.'],
        'source_hashes':{str(path.relative_to(ROOT)):digest(path) for path in [HOURLY,*RAIN_FILES]},
        'split_hashes':{name:digest(RUN/f'{name}.csv') for name in parts}}
    dump('protocol.json',protocol)
    train,val=parts['train'],parts['validation'];candidates=[];models={}
    options=[('logistic',LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=20261025)),
        ('decision_tree',DecisionTreeClassifier(max_depth=3,min_samples_leaf=3,class_weight='balanced',random_state=20261025)),
        ('random_forest',RandomForestClassifier(n_estimators=200,max_depth=5,min_samples_leaf=3,class_weight='balanced',n_jobs=2,random_state=20261025))]
    (RUN/'models').mkdir(exist_ok=True)
    with threadpool_limits(limits=2):
        for name,learner in options:
            pipeline=Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
                ('scale',StandardScaler()),('classifier',learner)])
            pipeline.fit(train[FEATURES],train.target)
            score=pipeline.predict_proba(val[FEATURES])[:,1]
            trials=[metrics(val.target,score,float(cut)) for cut in np.round(np.arange(.05,.951,.01),2)]
            best=max(trials,key=lambda m:(m['f1'],m['recall'],m['precision'],-abs(m['threshold']-.5)))
            candidates.append({'name':name,'validation':best})
            joblib.dump(pipeline,RUN/f'models/{name}.joblib');models[name]=pipeline
            print(name,json.dumps(best),flush=True)
    winner=max(candidates,key=lambda row:(row['validation']['f1'],row['validation']['recall'],row['validation']['precision']))
    dump('selection.json',{'selected_model':winner['name'],'threshold':winner['validation']['threshold'],
        'candidates':candidates,'test_used_for_selection':False})
    bundle={'pipeline':models[winner['name']],'model_name':winner['name'],'threshold':winner['validation']['threshold'],
        'features':FEATURES,'warning_m':WARNING,'danger_m':DANGER,'training_end':'2020-12-31','target':protocol['target']}
    joblib.dump(bundle,RUN/'model_bundle.joblib')
    output={'selected_model':winner['name'],'validation':winner['validation'],'test':{},'test_currently_below_warning':{},
        'source_scope':protocol['place'],'target':protocol['target'],'limits':protocol['limits']}
    test=parts['test'];predictions=test[['time','current_level','target_peak','target','currently_below_warning']].copy()
    candidate_scores={name:(model.predict_proba(test[FEATURES])[:,1],next(row['validation']['threshold'] for row in candidates if row['name']==name))
                      for name,model in models.items()}
    candidate_scores.update(persistence=(test.current_level.to_numpy().astype(float),WARNING),
        positive_trend=(test.current_level.to_numpy()+np.maximum(test.change_6h_m.to_numpy(),0)*2,WARNING),
        always_low=(np.zeros(len(test)),.5))
    for name,(score,threshold) in candidate_scores.items():
        output['test'][name]=metrics(test.target,score,threshold)
        subset=test.currently_below_warning.to_numpy(bool)
        output['test_currently_below_warning'][name]=metrics(test.target.to_numpy()[subset],score[subset],threshold)
        predictions[name+'_score']=score;predictions[name+'_prediction']=(score>=threshold).astype(int)
    predictions.to_csv(RUN/'test_predictions.csv',index=False)
    dump('final_results.json',output)
    replays=[]
    for row in test.itertuples():
        answer=predict_manual(row.time,row.current_level,row.level_6h_ago,row.level_24h_ago,
            None if pd.isna(row.recent_rain_mm) else row.recent_rain_mm,bundle=bundle)
        replays.append(answer['elevated_risk'])
    assert np.array_equal(replays,predictions[winner['name']+'_prediction'].to_numpy(bool))
    dump('verification.json',{'manual_inference_replayed_all_test_rows':len(test),'matches':True,
        'bundle_sha256':digest(RUN/'model_bundle.joblib'),'selection_sha256':digest(RUN/'selection.json'),
        'test_predictions_sha256':digest(RUN/'test_predictions.csv')})
    print('FROZEN RESULT',json.dumps(output),flush=True)


if __name__=='__main__':main()
