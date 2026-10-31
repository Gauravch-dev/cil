"""Reconstruct actual labels from raw CWC observations and verify frozen fits."""
from pathlib import Path
import sys
import json
from hashlib import sha256
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from city_flood import ROOT,RUN,WARNING,FEATURES,prepare_dataset
from river_forecast import clean_station


def main():
    protocol=json.loads((RUN/'protocol.json').read_text());verification=json.loads((RUN/'verification.json').read_text())
    for name,expected in protocol['source_hashes'].items():assert sha256((ROOT/name).read_bytes()).hexdigest()==expected
    for name,expected in protocol['split_hashes'].items():assert sha256((RUN/f'{name}.csv').read_bytes()).hexdigest()==expected
    for name,hashkey in [('model_bundle.joblib','bundle_sha256'),('selection.json','selection_sha256'),('test_predictions.csv','test_predictions_sha256')]:
        assert sha256((RUN/name).read_bytes()).hexdigest()==verification[hashkey]
    raw=pd.concat([pd.read_csv(ROOT/f'data/river_stage/raw/kurundwad_{period}.csv') for period in ['historical','recent']],ignore_index=True)
    hourly,audit=clean_station(raw,'kurundwad')
    rebuilt=prepare_dataset(hourly)
    saved=pd.concat([pd.read_csv(RUN/f'{name}.csv',parse_dates=['time','label_end']) for name in ['train','validation','test']],ignore_index=True)
    pd.testing.assert_frame_equal(saved[FEATURES+['time','label_end','target_peak','target','split']].reset_index(drop=True),
        rebuilt[FEATURES+['time','label_end','target_peak','target','split']].reset_index(drop=True),check_dtype=False,atol=1e-10)
    train=saved.loc[saved.split.eq('train')];val=saved.loc[saved.split.eq('validation')];test=saved.loc[saved.split.eq('test')]
    assert train.label_end.max()<val.time.min() and val.label_end.max()<test.time.min()
    assert saved.future_observed_hours.eq(24).all()
    assert np.array_equal(saved.target,saved.target_peak.ge(WARNING).astype(int))
    predictions=pd.read_csv(RUN/'test_predictions.csv');result=json.loads((RUN/'final_results.json').read_text())
    chosen=result['selected_model'];observed=test.target.to_numpy();pred=predictions[chosen+'_prediction']
    for name,fn in [('accuracy',accuracy_score),('precision',precision_score),('recall',recall_score),('f1',f1_score)]:
        assert abs(fn(observed,pred)-result['test'][chosen][name])<1e-12
    verification.update(raw_cwc_reconstruction_matches=True,all_source_labels_verified=len(saved),
        current_and_lagged_levels_only=True,future_windows_complete=True,chronological_boundaries_purged=True,
        target='Official warning-stage exceedance in next 24h',raw_quality_audit=audit)
    (RUN/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
