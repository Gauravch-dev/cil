"""Reconstruct a strongly supported original-data hypothesis without training.

This reproduces published artifacts/headline numbers. It is not a new independent
evaluation and does not turn a discharge-percentile label into observed flooding.
"""
from hashlib import sha256
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import DMatrix

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/upstream_forensics'
MODEL_SHA='aad88a12f4645f3703c996641a15ce40603f6d5a608918fad4f37496d2f89395'

def reconstruct():
    OUT.mkdir(exist_ok=True)
    model_path=ROOT/'flood_model.pkl'
    assert sha256(model_path.read_bytes()).hexdigest()==MODEL_SHA
    source=ROOT/'data/raw/kuntla_flooddatabase.csv'
    raw=pd.read_csv(source,low_memory=False)
    saved_map=pd.read_csv(ROOT/'high_risk_locations.csv')
    encoders=joblib.load(ROOT/'encoders.pkl')
    columns=list(joblib.load(ROOT/'feature_columns.pkl'))
    model=joblib.load(model_path)
    model.set_params(device='cpu',n_jobs=2)  # In memory only; never resave it.
    mapping={
      'annual_mean_temperature':'Annual Mean Temperature',
      'temperature_seasonality':'Temperature Seasonality',
      'annual_precipitation':'Annual Precipitation',
      'curve_number_amcii':'Curve Number AMCII',
      'drainage_density':'Drainage Density','basin_relief':'Basin Relief',
      'ruggedness_number':'Ruggedness Number','infiltration_number':'Infiltration Number',
    }
    X=pd.DataFrame({key:raw[name] for key,name in mapping.items()})
    X['precipitation_risk_index']=(np.minimum(raw['Annual Precipitation'],3000)*.4+
        np.minimum(raw['Precipitation of Wettest Month'],1000)*.3+
        np.minimum(raw['Precipitation Seasonality'],100)*.3)
    X['drainage_risk_index']=(np.minimum(raw['Drainage Density'],10)*.5+
        np.minimum(raw['Drainage Texture'],50)*.3+np.minimum(raw['Basin Relief'],5000)*.2)
    cat={'climate_type':'Climate Type','landcover_type':'Landcover Type',
         'soil_type':'Soil Type','lithology_type':'Lithology Type'}
    vocab={}
    for key,name in cat.items():
        values=raw[name].fillna('nan').astype(str)
        vocab[key]=bool(np.array_equal(np.sort(values.unique()),encoders[key].classes_))
        if key in columns:X[key]=encoders[key].transform(values)
    assert all(vocab.values())
    X=X[columns].fillna(X[columns].median())
    probability=model.predict_proba(X)[:,1]
    reproduced=raw[['latitude','longitude']].copy()
    reproduced['predicted_risk']=probability
    high=reproduced.loc[probability>.7].reset_index(drop=True)
    assert len(high)==len(saved_map)==2538
    assert np.allclose(high.to_numpy(),saved_map.to_numpy(),atol=1e-6,rtol=0)
    # A hypothesis, selected during forensics, not a supplied scientific label.
    cutoff=float(raw['Annual peak discharge'].quantile(.95))
    target=raw['Annual peak discharge'].gt(cutoff).astype(int)
    train,test=train_test_split(np.arange(len(raw)),test_size=.2,random_state=42,stratify=target)
    report=classification_report(target.iloc[test],probability[test]>=.5,output_dict=True,zero_division=0)
    booster=model.get_booster()
    margin=booster.predict(DMatrix(X.iloc[test]),output_margin=True,
        iteration_range=(0,int(booster.attr('best_iteration'))+1))
    loss=float(np.mean(np.logaddexp(0,margin.astype(float))-target.iloc[test].to_numpy()*margin))
    stored_loss=float(booster.attr('best_score'))
    assert abs(loss-stored_loss)<1e-6
    assert round(report['accuracy']*100,1)==96.9
    assert round(report['0']['f1-score'],3)==.984
    assert round(report['1']['recall'],3)==.791
    joined=raw[['gsim_code','latitude','longitude','year','Annual peak discharge']].copy()
    joined=joined.rename(columns={'Annual peak discharge':'source_annual_peak_discharge'})
    joined['source_row_index']=np.arange(len(raw))
    joined['reconstructed_high_discharge_class']=target
    joined['inferred_original_split']='train'
    joined.loc[test,'inferred_original_split']='inferred_holdout'
    joined=pd.concat([joined,X],axis=1)
    joined.to_csv(OUT/'reconstructed_original_dataset.csv',index=False)
    evidence={
      'dataset_name':'GloFlo kuntla_flooddatabase.csv',
      'source_sha256':sha256(source.read_bytes()).hexdigest(),
      'source_rows':len(raw),'source_columns':len(raw.columns),
      'exact_category_vocabulary_matches':vocab,
      'reproduced_high_risk_rows':len(high),
      'map_order_and_coordinates_match':True,
      'maximum_map_score_error':float(np.max(np.abs(high.predicted_risk.to_numpy()-saved_map.predicted_risk.to_numpy()))),
      'reconstructed_target_rule':'Annual peak discharge > full-source 95th percentile',
      'threshold_source_units':cutoff,
      'target_is_author_hypothesis_not_verified_flood_label':True,
      'positive_rows':int(target.sum()),'negative_rows':int(target.eq(0).sum()),
      'inferred_split':{'test_size':.2,'random_state':42,'stratify':'derived target'},
      'training_rows':len(train),'inferred_holdout_rows':len(test),
      'reconstructed_report':report,
      'confusion_matrix':confusion_matrix(target.iloc[test],probability[test]>=.5).tolist(),
      'reconstructed_validation_logloss':loss,'stored_best_logloss':stored_loss,
      'loss_absolute_difference':abs(loss-stored_loss),
      'test_rows_with_station_already_in_train':int(raw.iloc[test].gsim_code.isin(raw.iloc[train].gsim_code).sum()),
      'training_was_not_performed':True,'original_model_unchanged':True,
    }
    (OUT/'reconstruction_summary.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    text=f'''# Original model dataset and target reconstruction

## Finding

The original data source is strongly corroborated as GloFlo, `kuntla_flooddatabase.csv`, the same 45,932-row, 93-column CSV already supplied by the user. The original README explicitly names it. More importantly, all four saved categorical encoder vocabularies match exactly. Replaying its 13-feature inference pipeline with median imputation recreates all 2,538 original high-risk map rows in their original order. Maximum score error is {evidence['maximum_map_score_error']:.9g}, consistent with floating-point/export rounding.

## Missing target step reconstructed

The strongest supported hypothesis is:

```python
cutoff = df['Annual peak discharge'].quantile(0.95)  # 1145.198 in source units
y = (df['Annual peak discharge'] > cutoff).astype(int)
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

This yields 2,297 high-discharge and 43,635 other peak-flow records. The inferred split has 36,745 training rows and 9,187 held-out rows. Replaying the existing model, without training, reproduces every advertised headline value after rounding:

| Quantity | Reconstructed | README claim |
|---|---:|---:|
| Overall accuracy | {report['accuracy']*100:.5f}% | 96.9% |
| Class 1 recall | {report['1']['recall']:.9f} | 0.791 |
| Class 0 F1 | {report['0']['f1-score']:.9f} | 0.984 |
| Class 1 precision | {report['1']['precision']:.9f} | Not reported |
| Class 1 F1 | {report['1']['f1-score']:.9f} | Not reported |

The advertised F1 therefore matches class 0, whereas the advertised recall matches class 1. This strongly suggests metrics were taken from different rows of a classification report. It does not establish misconduct or intent.

Reconstructed log-loss is {loss:.12f}; the model stores best validation log-loss {stored_loss:.12f}, a difference of {abs(loss-stored_loss):.3g}. The saved first-tree Hessian cover is also compatible with balancing the inferred 34,907 training negatives and 1,838 positives to 34,907 per class with scale_pos_weight=2. This is additional corroboration, not an exact reconstruction of the synthetic SMOTE rows.

## Public source history

The public GitHub history has six commits, one main branch and no tags. The root commit already contains the model, encoders and map outputs. Its initial file tree contains no notebook, raw training CSV or target-generation code. Later commits add documentation/development configuration. The public Hugging Face Space also contains deployment files and artifacts, without the original notebook. API snapshots are retained alongside this report.

The visible `flood_fast.py` is a demo helper. It can generate random examples, but it does not explain the saved artifact: it exports empty encoders and uses 100 estimators with different parameters. The saved artifact requests up to 300 estimators, uses early stopping, retains 214 trees with best_iteration=193, and has real categorical encoders. The deployed artifacts should not be called synthetic solely because this separate helper has a synthetic fallback.

## What this establishes and what it does not

The raw dataset connection is exceptionally strong because its saved map is reproduced. The threshold and split are a highly corroborated reconstruction, not source code recovered from the author’s private Colab. The fitted imputation values and exact SMOTE interpolation rows are not independently recovered.

The target is **high annual peak discharge relative to a global percentile**, not an observed operational flood/no-flood label. The full-source percentile uses the entire data distribution. The inferred split is random by record; {evidence['test_rows_with_station_already_in_train']:,} of 9,187 held-out rows are at stations also present in training. Repeated static catchment features therefore span both partitions. These reconstructed scores are not a new chronological or gauge-independent evaluation and cannot be compared directly with the INDOFLOODS Flood/Severe Flood task.

The reconstruction was selected by forensic agreement with model artifacts and published numbers. It is not a blind confirmation experiment and must not be promoted as newly verified real-world forecasting accuracy. No training or application model replacement was performed.

## Reproduction

```powershell
.\\.venv\\Scripts\\python.exe scripts/reconstruct_original_dataset.py
```

`reconstructed_original_dataset.csv` retains source identifiers, measured peak discharge, all 13 engineered inputs, an explicitly named derived high-discharge class and the inferred partition. The raw source and original model remain unchanged.

## Sources

- Original repository and README: https://github.com/nitinc264/Flood-prediction-Model/tree/2d823f3692e891ac92de4435bbe2c006131c30b3
- GloFlo source: https://zenodo.org/records/7158028
- Public deployed files: https://huggingface.co/spaces/nitin454545/riskradar/tree/main
- Local raw source SHA256: {evidence['source_sha256']}
- Original model SHA256: {MODEL_SHA}
'''
    (OUT/'PROVENANCE_FINDINGS.md').write_text(text,encoding='utf-8')
    assert sha256(model_path.read_bytes()).hexdigest()==MODEL_SHA
    print(json.dumps({'dataset_rows':len(joined),'target_cutoff':cutoff,
        'accuracy':report['accuracy'],'positive_f1':report['1']['f1-score'],
        'negative_f1':report['0']['f1-score'],'positive_recall':report['1']['recall'],
        'logloss_difference':abs(loss-stored_loss),'map_reproduced':len(high),
        'report':str(OUT/'PROVENANCE_FINDINGS.md')},indent=2))

if __name__=='__main__':reconstruct()
