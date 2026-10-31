"""Matched rainfall and recorded-flood analysis with frozen severity models.

The observational unit is a recorded flood event. No missing inventory report is
treated as a verified non-flood day. Model scores are conditional severity scores.
"""
from pathlib import Path
from hashlib import sha256
import json

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'experiments/indofloods_severity_v1'
DATA=ROOT/'data/rainfall_flood'
RELEASE=ROOT/'experiments/rainfall_flood_release_v1'
RAIN=[f'T{day}d' for day in range(1,11)]
MODELS={'xgb_static':'Catchment and season', 'xgb_rainfall':'Catchment and recent rainfall'}


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()


def daily_rainfall(row):
    """Convert cumulative PRE-onset depths to ten dated daily depths in mm."""
    cumulative=pd.to_numeric(pd.Series([row[key] for key in RAIN]),errors='coerce').to_numpy(float)
    differences=np.diff(np.r_[0.,cumulative])
    if not np.isfinite(differences).all() or (differences < -1e-6).any():
        raise ValueError('Incomplete or inconsistent cumulative rainfall.')
    start=pd.Timestamp(row['Start Date'])
    return pd.DataFrame({'date':pd.date_range(start-pd.Timedelta(days=10),periods=10),
                         'rain_mm':np.maximum(differences,0)[::-1]})


def filter_events(frame,state='All covered states',years=None):
    mask=pd.Series(True,index=frame.index)
    if state!='All covered states':mask &= frame.State.eq(state)
    if years is not None:mask &= frame['Start Date'].dt.year.between(*years)
    return frame.loc[mask].copy()


def rain_groups(frame,window='T3d',edges=None):
    """Descriptive severe share among floods; never population flood probability."""
    if edges is None:
        meta=json.loads((DATA/'manifest.json').read_text())
        edges=meta['training_rainfall_quartiles'][window]
    if len(edges)!=3 or np.any(np.diff(edges)<=0):
        raise ValueError('Three increasing training quartiles are required.')
    labels=[f'Below {edges[0]:.0f}',f'{edges[0]:.0f}–{edges[1]:.0f}',
            f'{edges[1]:.0f}–{edges[2]:.0f}',f'Above {edges[2]:.0f}']
    part=frame.assign(rain_group=pd.cut(frame[window],[-np.inf,*edges,np.inf],
                                       labels=labels,include_lowest=True))
    group=part.groupby('rain_group',observed=False).agg(events=('EventID','size'),severe=('target','sum'))
    group['severe_share']=group.severe.div(group.events.where(group.events.gt(0)))
    return group.reset_index()


def state_tokens(value):
    aliases={'orissa':'Odisha','uttaranchal':'Uttarakhand','madras':'Tamil Nadu','pondicherry':'Puducherry'}
    if pd.isna(value):return []
    return list(dict.fromkeys(aliases.get(part.strip().lower(),part.strip())
                             for part in str(value).split(',') if part.strip()))


def filter_inventory(frame,state='All covered states',years=None):
    mask=pd.Series(True,index=frame.index)
    if state!='All covered states':mask &= frame.State.map(lambda value:state in state_tokens(value))
    if years is not None:mask &= frame['Start Date'].dt.year.between(*years)
    # Multi-state events remain one source record in a national total.
    return frame.loc[mask].copy()


def load_events():
    return pd.read_csv(DATA/'matched_events.csv',parse_dates=['Start Date','End Date'])


def load_inventory():
    return pd.read_csv(DATA/'flood_inventory.csv',parse_dates=['Start Date','End Date'])


def load_bundle():return joblib.load(RELEASE/'severity_comparison.joblib')


def gauge_risk_summary(frame,training_reference):
    """Relative observed severe share, NOT a model of daily flood probability."""
    reference=training_reference.groupby('GaugeID').agg(n=('EventID','size'),share=('target','mean'))
    qualified=reference.loc[reference.n.ge(10),'share']
    if len(qualified)<3:raise ValueError('Not enough reference gauges to define historical bands.')
    lower,upper=qualified.quantile([1/3,2/3]).tolist()
    grouped=frame.groupby('GaugeID').agg(Station=('Station','first'),State=('State','first'),
        Latitude=('Latitude','first'),Longitude=('Longitude','first'),events=('EventID','size'),
        severe=('target','sum'),rain_3d_median=('T3d','median')).reset_index()
    grouped['severe_share']=grouped.severe/grouped.events
    grouped['band']=np.where(grouped.events.lt(10),'Limited records',
        np.where(grouped.severe_share.le(lower),'Low',np.where(grouped.severe_share.le(upper),'Medium','High')))
    return grouped,{'low_max':lower,'medium_max':upper,'minimum_events':10}


def severity_estimates(row,bundle=None):
    """Replay a held-out known event without exposing outcomes to predictors."""
    if row['split']!='test':
        raise ValueError('The comparison is available only for held-out 2015–2019 events.')
    bundle=load_bundle() if bundle is None else bundle
    predictions=[]
    for name,label in MODELS.items():
        columns=bundle['features'][name]
        inputs=pd.DataFrame([{key:row[key] for key in columns}])
        score=float(bundle['models'][name].predict_proba(inputs)[0,1])
        threshold=float(bundle['thresholds'][name])
        predictions.append({'model':name,'name':label,'score':score,'threshold':threshold,
                            'estimated_severity':'Severe Flood' if score>=threshold else 'Flood'})
    return predictions
