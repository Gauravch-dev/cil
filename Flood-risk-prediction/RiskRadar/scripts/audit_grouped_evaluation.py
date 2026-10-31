"""Independently recompute frozen evaluation evidence without tuning models."""
from pathlib import Path
from hashlib import sha256
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score,average_precision_score,confusion_matrix,
                             f1_score,precision_score,recall_score,roc_auc_score,
                             precision_recall_curve)

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/gloflo_grouped_evaluation_v1'
def read(name):return json.loads((RUN/name).read_text(encoding='utf-8'))
def digest(path):return sha256(path.read_bytes()).hexdigest()

def main():
    protocol=read('protocol.json');selection=read('selection.json');result=read('final_results.json')
    assert digest(RUN/'selection.json')==result['selection_sha256']
    assert digest(RUN/'protocol.json')==result['protocol_sha256']
    assert result['selected_before_test']==selection['selected_name']
    assert read('test_opened.json')['selection_sha256']==result['selection_sha256']
    parts={name:pd.read_csv(RUN/'data'/f'{name}.csv',keep_default_na=False,na_values=[''])
           for name in ['train','validation','test']}
    for name,part in parts.items():assert digest(RUN/'data'/f'{name}.csv')==protocol['split_sha256'][name]
    assert np.isclose(parts['train'].source_discharge.quantile(.95),protocol['target']['cutoff'])
    for column in ['station_id','group_id']:
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            assert not set(parts[a][column])&set(parts[b][column])
    predictions=pd.read_csv(RUN/'test_predictions.csv',float_precision='round_trip')
    assert digest(RUN/'test_predictions.csv')==result['test_predictions_sha256']
    assert predictions.source_row.tolist()==parts['test'].source_row.tolist()
    y=parts['test'].source_discharge.gt(protocol['target']['cutoff']).astype(int)
    np.testing.assert_array_equal(y,predictions.target)
    chosen=selection['selected_name'];p=predictions[chosen+'_score'].to_numpy()
    prediction=(p>=selection['threshold']).astype(int)
    np.testing.assert_array_equal(prediction,predictions[chosen+'_class'])
    metrics=result['metrics'][chosen]['test']
    computed={'accuracy':accuracy_score(y,prediction),'precision':precision_score(y,prediction,zero_division=0),
        'recall':recall_score(y,prediction,zero_division=0),'f1':f1_score(y,prediction,zero_division=0),
        'roc_auc':roc_auc_score(y,p),'average_precision':average_precision_score(y,p)}
    for key,value in computed.items():assert np.isclose(value,metrics[key],atol=1e-12,rtol=0),(key,value,metrics[key])
    np.testing.assert_array_equal(confusion_matrix(y,prediction),metrics['confusion_matrix'])
    for row in selection['candidates']:
        assert digest(RUN/'models'/f"{row['name']}.joblib")==row['model_sha256']
    validation_predictions=pd.read_csv(RUN/'validation_predictions.csv',float_precision='round_trip')
    assert validation_predictions.source_row.tolist()==parts['validation'].source_row.tolist()
    validation_y=parts['validation'].source_discharge.gt(protocol['target']['cutoff']).astype(int)
    np.testing.assert_array_equal(validation_y,validation_predictions.target)
    for row in selection['candidates']:
        values=validation_predictions[row['name']].to_numpy()
        # XGBoost emits float32 probabilities; its CSV stores their shortest
        # float32 representation. Restore that original dtype at threshold ties.
        if row['family']=='xgboost':values=values.astype(np.float32)
        p,r,thresholds=precision_recall_curve(validation_y,values)
        f1=np.divide(2*p[:-1]*r[:-1],p[:-1]+r[:-1],out=np.zeros(len(thresholds)),where=(p[:-1]+r[:-1])>0)
        recorded=row['validation']
        assert np.isclose(recorded['f1'],f1.max(),atol=1e-12,rtol=0)
        # Recompute metrics from stored validation scores independently of the
        # runner's summaries; do not use any test result in this verification.
        selected_predictions=values>=recorded['threshold']
        assert np.isclose(f1_score(validation_y,selected_predictions),recorded['f1'],atol=1e-12,rtol=0),row['name']
    # Recheck the train-fitted transforms, not merely the final model scores.
    model=joblib.load(RUN/'models'/f'{chosen}.joblib');pre=model['preprocess']
    numeric_columns=pre.transformers_[0][2]
    expected=parts['train'][numeric_columns].median().to_numpy()
    np.testing.assert_allclose(expected,pre.named_transformers_['numeric']['impute'].statistics_)
    if 'categories' in pre.named_transformers_:
        columns=pre.transformers_[1][2]
        for col,values in zip(columns,pre.named_transformers_['categories'].categories_):
            np.testing.assert_array_equal(np.sort(parts['train'][col].astype(str).unique()),values)
    assert digest(ROOT/'flood_model.pkl')==protocol['original_model_sha256']
    assert result['original_deployed_model_unchanged']
    order=sorted(selection['candidates'],key=lambda c:(-c['validation']['f1'],
                 -c['validation']['average_precision'],-c['validation']['recall'],c['name']))
    assert order[0]['name']==chosen
    audit={'selected_model':chosen,'independently_recomputed_test_metrics':computed,
        'station_and_group_overlap_zero':True,'training_only_percentile_verified':True,
        'training_only_imputation_and_categories_verified':True,'frozen_selection_and_threshold_verified':True,
        'all_validation_f1_and_thresholds_independently_verified':True,
        'unmodified_models_and_partitions_verified':True,'original_website_model_unchanged':True,
        'confusion_matrix':confusion_matrix(y,prediction).tolist()}
    (RUN/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
