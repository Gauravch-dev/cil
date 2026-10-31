"""Small, manually usable Kurundwad flood-warning classifier.

Target: a measured river stage reaches the published warning level in the
following 24 hours. It does not label district inundation or all urban floods.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'experiments/kurundwad_flood_risk_v1'
HOURLY=ROOT/'data/river_stage/processed/kurundwad_hourly.csv'
RAIN_FILES=[ROOT/'research/flood_observations_20261024'/f'kurundwad_rain_{period}.csv'
            for period in ['historical','recent']]
WARNING=535.64
DANGER=537.16
FEATURES=['distance_to_warning_m','change_6h_m','change_24h_m','acceleration_m_per_h',
          'recent_rain_mm','rain_available','month_sin','month_cos']


def rainfall_reports():
    records=pd.concat([pd.read_csv(path) for path in RAIN_FILES],ignore_index=True)
    if not records.Station.eq('KURUNDWAD').all():raise ValueError('Rain station mismatch')
    records['time']=pd.to_datetime(records['Data Acquisition Time'],format='%d-%m-%Y %H:%M',errors='coerce')
    records['rain_mm']=pd.to_numeric(records['Manual Daily Rainfall (mm)'],errors='coerce')
    records=records.loc[records.time.notna() & records.rain_mm.ge(0)]
    group=records.groupby('time').rain_mm
    values=group.first().sort_index()
    values.loc[(group.max()-group.min()).gt(.001)]=np.nan
    return values.dropna().rename('recent_rain_mm').reset_index()


def feature_values(issue,current,previous6,previous24,rain=None):
    """Identical feature calculation for manual and source-derived inputs."""
    issue=pd.Timestamp(issue)
    if pd.isna(issue):raise ValueError('Enter a valid observation date.')
    if issue.tzinfo is not None:issue=issue.tz_convert('Asia/Kolkata').tz_localize(None)
    levels=np.asarray([current,previous6,previous24],dtype=float)
    if not np.isfinite(levels).all() or not ((levels>=509.455)&(levels<=567.16)).all():
        raise ValueError('Enter river elevations between 509.455 and 567.16 metres using the Kurundwad gauge datum.')
    if abs(current-previous6)>60 or abs(current-previous24)>240:
        raise ValueError('These river changes exceed the source quality limits. Check the units.')
    if rain is not None and (not np.isfinite(rain) or rain<0):
        raise ValueError('Rainfall must be nonnegative, or marked unavailable.')
    change6=float(current-previous6);change24=float(current-previous24)
    return pd.DataFrame([dict(distance_to_warning_m=float(current-WARNING),change_6h_m=change6,
        change_24h_m=change24,acceleration_m_per_h=change6/6-change24/24,
        recent_rain_mm=np.nan if rain is None else float(rain),rain_available=int(rain is not None),
        month_sin=float(np.sin(2*np.pi*issue.month/12)),month_cos=float(np.cos(2*np.pi*issue.month/12)))],
        columns=FEATURES)


def prepare_dataset(hourly=None,rain=None):
    """Use real past readings; all 24 future bins must be measured for labels."""
    if hourly is None:hourly=pd.read_csv(HOURLY,parse_dates=['time']).set_index('time')
    if rain is None:rain=rainfall_reports()
    hourly=hourly.sort_index().reindex(pd.date_range(hourly.index.min(),hourly.index.max(),freq='h'))
    level=hourly.level
    table=pd.DataFrame({'current_level':level,'level_6h_ago':level.shift(6),'level_24h_ago':level.shift(24)})
    future=pd.concat([hourly.observed_max.shift(-h) for h in range(1,25)],axis=1)
    table['future_observed_hours']=future.notna().sum(axis=1)
    table['target_peak']=future.max(axis=1).where(table.future_observed_hours.eq(24))
    table=table.loc[(table.index.hour==8)&table[['current_level','level_6h_ago','level_24h_ago','target_peak']].notna().all(axis=1)].copy()
    table.index.name='time'
    table=table.reset_index()
    table['time']=table.time.astype('datetime64[ns]')
    rain=rain.copy();rain['time']=pd.to_datetime(rain.time).astype('datetime64[ns]')
    # At 08:00 today's 08:30 report is future. Use only a measured report
    # within the preceding 30h; missing/omitted rain never means zero rainfall.
    table=pd.merge_asof(table.sort_values('time'),rain.sort_values('time'),on='time',
                        direction='backward',tolerance=pd.Timedelta(hours=30),allow_exact_matches=True)
    rows=[]
    for row in table.itertuples():
        rain_value=None if pd.isna(row.recent_rain_mm) else row.recent_rain_mm
        rows.append(feature_values(row.time,row.current_level,row.level_6h_ago,row.level_24h_ago,rain_value).iloc[0])
    table[FEATURES]=pd.DataFrame(rows,index=table.index)
    table['target']=table.target_peak.ge(WARNING).astype(int)
    table['label_end']=table.time+pd.Timedelta(hours=24)
    table['currently_below_warning']=table.current_level.lt(WARNING)
    table['split']='excluded'
    for name,start,end in [('train','2014-01-01','2020-12-31 23:59:59'),
                           ('validation','2021-01-01','2022-12-31 23:59:59'),
                           ('test','2023-01-01','2025-12-31 23:59:59')]:
        table.loc[table.time.ge(start)&table.label_end.le(end),'split']=name
    return table.loc[table.split.ne('excluded')].reset_index(drop=True)


def load_bundle():return joblib.load(RUN/'model_bundle.joblib')


def predict_manual(issue,current,previous6,previous24,rain=None,bundle=None):
    features=feature_values(issue,current,previous6,previous24,rain)
    bundle=load_bundle() if bundle is None else bundle
    score=float(bundle['pipeline'].predict_proba(features)[0,1])
    elevated=score>=bundle['threshold']
    return {'elevated_risk':bool(elevated),'risk':'Elevated' if elevated else 'Lower',
        'current_status':'Danger' if current>=DANGER else 'Warning' if current>=WARNING else 'Below warning',
        'period_start':str(pd.Timestamp(issue)),'period_end':str(pd.Timestamp(issue)+pd.Timedelta(hours=24)),
        'model_score':score,'distance_to_warning_m':float(WARNING-current),
        'change_6h_m':float(current-previous6),'change_24h_m':float(current-previous24),
        'rain_mm':None if rain is None else float(rain),'model':bundle['model_name']}
