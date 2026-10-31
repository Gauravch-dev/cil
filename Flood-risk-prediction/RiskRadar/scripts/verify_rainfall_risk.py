"""Verify the frozen rainfall benchmark, thresholds and causal runtime replay."""
from pathlib import Path
import sys
import json
from hashlib import sha256
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
from rainfall_risk import SOURCE,RUN,NUMERIC,weather,features,category,classify,ml_bundle,predict,baseline_probabilities


def digest(path):return sha256(path.read_bytes()).hexdigest()


def main():
    protocol=json.loads((RUN/'protocol.json').read_text());selection=json.loads((RUN/'selection.json').read_text())
    result=json.loads((RUN/'final_results.json').read_text())
    assert digest(SOURCE)==protocol['source_sha256']
    assert digest(RUN/'protocol.json')==selection['protocol_sha256']
    assert digest(RUN/'selection.json')==result['selection_sha256']
    assert digest(RUN/'forecast_bundle.joblib')==result['bundle_sha256']
    frames={name:pd.read_csv(RUN/f'{name}.csv',parse_dates=['date','label_end']).set_index('date')
            for name in ['train','validation','test']}
    for name,value in protocol['split_hashes'].items():assert digest(RUN/f'{name}.csv')==value
    raw=weather();bank=features(raw)
    for name,frame in frames.items():
        actual=pd.concat([raw.rain_mm.shift(-i) for i in [1,2,3]],axis=1).sum(axis=1,min_count=3).loc[frame.index]
        np.testing.assert_allclose(actual,frame.target_total,atol=1e-9)
        np.testing.assert_array_equal(category(actual,protocol['thresholds']),frame.target)
    train=frames['train'];wet=train.loc[train.target_total.gt(0),'target_total']
    np.testing.assert_allclose(wet.quantile([.90,.95]).to_numpy(),list(protocol['thresholds'].values()))
    assert train.label_end.max()<frames['validation'].index.min()
    assert frames['validation'].label_end.max()<frames['test'].index.min()
    predictions=pd.read_csv(RUN/'test_predictions.csv',parse_dates=['date']).set_index('date')
    test=frames['test'];primary=ml_bundle();baseline=joblib.load(RUN/'forecast_bundle.joblib')
    for bundle in [primary,baseline]:
        probability=(bundle['pipeline'].predict_proba(bank.loc[test.index]) if bundle['kind']=='ml' else
                     baseline_probabilities(bank.loc[test.index],bundle['name'],bundle))
        np.testing.assert_array_equal(classify(probability,bundle['cutoffs']),predictions[bundle['name']])
    for pos in [0,len(test)//2,len(test)-1]:
        date=test.index[pos];first=predict(date,bundle=primary)
        altered=raw.copy();altered.loc[altered.index>date,'rain_mm']=9999.
        with patch('rainfall_risk.weather',return_value=altered):second=predict(date,bundle=primary)
        assert first==second
    output={'frozen_artifacts':'passed','source':'Existing near-Mumbai archive only; no additional location data downloaded',
            'test_rows':len(test),'ml_replayed':primary['name'],'validation_winner':baseline['name'],
            'model_output_is_flood_occurrence':False,'classes':'Next-three-day rainfall categories',
            'future_mutation_runtime_checks':3,'thresholds_derived_from_train_only':True,
            'all_saved_test_classes_reproduced':True}
    (RUN/'verification.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
