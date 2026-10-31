"""Score the exact deployed XGBoost artifact without fitting or tuning anything."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix,average_precision_score,roc_auc_score
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/existing_model'
EXPECTED_MODEL='aad88a12f4645f3703c996641a15ce40603f6d5a608918fad4f37496d2f89395'
EXPECTED_SOURCE='20a844f84f78729cf4361f3f13daa3a79626f67c2e96aeb0f1a3b872bfd1a45a'
def digest(path):return sha256(path.read_bytes()).hexdigest()

def main():
    artifacts={name:digest(ROOT/name) for name in ['flood_model.pkl','encoders.pkl','feature_columns.pkl']}
    assert artifacts['flood_model.pkl']==EXPECTED_MODEL
    assert digest(ROOT/'data/raw/kuntla_flooddatabase.csv')==EXPECTED_SOURCE
    raw=pd.read_csv(ROOT/'data/raw/kuntla_flooddatabase.csv',low_memory=False)
    mapping={
        'annual_mean_temperature':'Annual Mean Temperature','temperature_seasonality':'Temperature Seasonality',
        'annual_precipitation':'Annual Precipitation','curve_number_amcii':'Curve Number AMCII',
        'drainage_density':'Drainage Density','basin_relief':'Basin Relief',
        'ruggedness_number':'Ruggedness Number','infiltration_number':'Infiltration Number'}
    x=pd.DataFrame({key:raw[name] for key,name in mapping.items()})
    x['precipitation_risk_index']=(np.minimum(raw['Annual Precipitation'],3000)*.4+
        np.minimum(raw['Precipitation of Wettest Month'],1000)*.3+
        np.minimum(raw['Precipitation Seasonality'],100)*.3)
    x['drainage_risk_index']=(np.minimum(raw['Drainage Density'],10)*.5+
        np.minimum(raw['Drainage Texture'],50)*.3+np.minimum(raw['Basin Relief'],5000)*.2)
    encoders=joblib.load(ROOT/'encoders.pkl');columns=list(joblib.load(ROOT/'feature_columns.pkl'))
    for key,name in {'climate_type':'Climate Type','landcover_type':'Landcover Type','soil_type':'Soil Type'}.items():
        x[key]=encoders[key].transform(raw[name].fillna('nan').astype(str))
    # Reproduce the documented original reconstruction, including its source-wide
    # medians and target cutoff. Do not relabel this as a new leakage-free test.
    x=x[columns].fillna(x[columns].median())
    cutoff=float(raw['Annual peak discharge'].quantile(.95))
    target=raw['Annual peak discharge'].gt(cutoff).astype(int)
    train,test=train_test_split(np.arange(len(raw)),test_size=.2,random_state=42,stratify=target)
    model=joblib.load(ROOT/'flood_model.pkl')
    model.set_params(device='cpu',n_jobs=2)  # In-memory execution settings only.
    scores=model.predict_proba(x.iloc[test])[:,1]
    predicted=scores>=.5;y=target.iloc[test]
    report=classification_report(y,predicted,labels=[0,1],output_dict=True,zero_division=0)
    matrix=confusion_matrix(y,predicted,labels=[0,1]).tolist()
    previous=json.loads((ROOT/'evaluation/upstream_forensics/reconstruction_summary.json').read_text())
    assert matrix==previous['confusion_matrix']
    positive=report['1']
    for key in ['precision','recall','f1-score']:
        assert np.isclose(positive[key],previous['reconstructed_report']['1'][key],atol=1e-12,rtol=0)
    OUT.mkdir(exist_ok=True)
    predictions=raw.iloc[test][['gsim_code','year','Annual peak discharge']].copy()
    predictions['source_row']=test;predictions['target']=y.to_numpy()
    predictions['score']=scores.astype(float);predictions['predicted_class']=predicted.astype(int)
    predictions.to_csv(OUT/'predictions.csv',index=False)
    result={'evaluated_at':datetime.now(timezone.utc).isoformat(),
        'model':'Exact deployed flood_model.pkl XGBoost classifier','artifact_sha256':artifacts,
        'source_sha256':EXPECTED_SOURCE,'training_performed':False,'tuning_performed':False,
        'model_replaced':False,'target':'Reconstructed annual high-discharge proxy',
        'target_cutoff':cutoff,'decision_threshold':.5,'evaluation_rows':len(test),'positive_cases':int(y.sum()),
        'accuracy':report['accuracy'],'precision':positive['precision'],'recall':positive['recall'],
        'f1':positive['f1-score'],'macro_metrics':report['macro avg'],'class_report':report,
        'roc_auc':float(roc_auc_score(y,scores)),'average_precision':float(average_precision_score(y,scores)),
        'confusion_matrix':matrix,'evaluation_type':'Reproduction on inferred original stratified row holdout',
        'test_rows_at_training_stations':int(raw.iloc[test].gsim_code.isin(raw.iloc[train].gsim_code).sum()),
        'independent_unseen_labelled_dataset_available':False,
        'independent_generalisation_accuracy_established':False,
        'basis':'Target/split inferred from existing evidence; source-wide cutoff and medians reproduce original processing. Previously analysed records do not become unseen to the frozen model by creating another split.'}
    assert artifacts=={name:digest(ROOT/name) for name in artifacts}
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    text=f'''# Evaluation of the existing deployed model

The evaluated model is the exact XGBoost artifact used by the Flood interface: `flood_model.pkl`. Its SHA256 is `{EXPECTED_MODEL}`. No training, calibration, threshold tuning, model selection or artifact replacement was performed.

| Metric | Existing model result |
|---|---:|
| Accuracy | {result['accuracy']:.2%} |
| Positive-class precision | {result['precision']:.2%} |
| Positive-class recall | {result['recall']:.2%} |
| Positive-class F1 | {result['f1']:.2%} |

The reconstructed original holdout contains {len(test):,} records and {int(y.sum()):,} positive high-discharge cases. The binary score threshold remains 0.5. Results reproduce the earlier original-model evidence exactly.

| Actual / Predicted | Other discharge | High discharge |
|---|---:|---:|
| Other discharge | {matrix[0][0]:,} | {matrix[0][1]:,} |
| High discharge | {matrix[1][0]:,} | {matrix[1][1]:,} |

The positive target is reconstructed as annual peak discharge above the full-source 95th percentile ({cutoff:.3f} source units). It is not a supplied verified flood/no-flood label. The target and original split are inferred, and {result['test_rows_at_training_stations']:,} holdout records share stations with the inferred training partition. The reconstruction reproduces source-wide numerical median handling. These are reproduction metrics, not an independently established unbiased estimate of future performance.

Evaluating this frozen artifact on genuinely unseen observations requires an independently labelled dataset compatible with its input schema and target. Creating new train/test partitions of its previously used source does not establish that those rows were unseen by this model. No such independent dataset has been identified among the supplied files.

The separate HistGradientBoosting and other retraining experiment evaluates different newly fitted models. Its results must not be attributed to this deployed XGBoost artifact.

Reproduce inference-only scoring: `.venv\\Scripts\\python.exe scripts/evaluate_existing_model.py`. Machine-readable results and per-record predictions are saved beside this report.
'''
    (OUT/'EVALUATION.md').write_text(text,encoding='utf-8')
    print(json.dumps({key:result[key] for key in ['model','training_performed','tuning_performed',
        'accuracy','precision','recall','f1','confusion_matrix','independent_generalisation_accuracy_established']},indent=2))

if __name__=='__main__':main()
