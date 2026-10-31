"""CWC river-level regression: causal features and official stage rules.

Local timestamps are interpreted as India Standard Time. This is a historical
forecast demonstrator, not a live warning service or an inundation model.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/river_stage'
EXPERIMENT=ROOT/'experiments/river_stage_v1'
STATIONS={
 'kurundwad':dict(name='Kurundwad',district='Kolhapur',river='Krishna',warning=535.64,danger=537.16,
                 zero=519.455,highest=541.35),
 'narsingpur':dict(name='Narsingpur',district='Solapur',river='Bhima',warning=458.403,danger=459.033,
                  zero=446.,highest=462.393),
 'takli':dict(name='Takli',district='Solapur',river='Bhima',warning=419.,danger=420.5,
             zero=408.,highest=424.5),
}
NUMERIC_FEATURES=['current_relative_warning','warning_danger_gap','month_sin','month_cos','day_sin','day_cos']
for lag in [1,3,6,12,24,48,72]:
    NUMERIC_FEATURES += [f'change_{lag}h']
for window in [6,24,72]:
    NUMERIC_FEATURES += [f'mean_change_{window}h',f'max_change_{window}h',f'min_change_{window}h',f'std_{window}h']
NUMERIC_FEATURES += ['rise_6h','rise_24h','acceleration_6h_vs_24h']
FEATURES=['station']+NUMERIC_FEATURES


def stage_category(level,station):
    """Rules use station-specific published CWC thresholds, not fitted classes."""
    if not np.isfinite(level):
        raise ValueError('A finite river level is required.')
    info=STATIONS[station]
    return 'Danger' if level>=info['danger'] else 'Warning' if level>=info['warning'] else 'Normal'


def clean_station(raw,station):
    """Conservative causal QC; never repair datum changes or decimal errors.

    Extremely wide bounds only flag obvious unit/datum errors. Limits are fixed
    from station metadata, not fitted to evaluation outcomes. Conflicting same-
    timestamp duplicates are unknown. Accepted-value jump and stale checks use
    past observations only; future records cannot revise earlier clean values.
    """
    meta=STATIONS[station]
    t=pd.to_datetime(raw['Data Acquisition Time'],format='%d-%m-%Y %H:%M',errors='coerce')
    v=pd.to_numeric(raw['River Water Level Manual Hourly (meter)'],errors='coerce')
    frame=pd.DataFrame({'time':t,'level':v})
    audit={'raw_rows':len(frame),'invalid_time':int(t.isna().sum()),'non_numeric':int(v.isna().sum())}
    frame=frame.loc[frame.time.notna()].sort_values('time')
    grouped=frame.groupby('time')['level']
    conflicting=(grouped.max()-grouped.min()).gt(.001)
    obs=grouped.first().sort_index()
    audit['duplicate_rows']=int(frame.time.duplicated().sum())
    audit['conflicting_timestamps']=int(conflicting.sum())
    obs.loc[conflicting]=np.nan
    bad=~obs.between(meta['zero']-10,meta['danger']+30)
    audit['wide_bound_or_missing']=int(bad.sum())
    obs.loc[bad]=np.nan
    accepted=[];last_t=None;last_v=None;flat_start=None;jumps=0;stale=0
    for timestamp,value in obs.items():
        if not np.isfinite(value):
            accepted.append(np.nan);continue
        if last_t is not None:
            hours=(timestamp-last_t).total_seconds()/3600
            if hours>0 and abs(value-last_v)/hours>10:
                accepted.append(np.nan);jumps+=1;continue
        if last_v is None or abs(value-last_v)>.0005:
            flat_start=timestamp
        if flat_start is not None and timestamp-flat_start>pd.Timedelta(hours=72):
            accepted.append(np.nan);stale+=1
            # Preserve the last measured timestamp so an unchanged value cannot
            # become fresh merely by being re-published.
            continue
        accepted.append(value);last_t=timestamp;last_v=value
    obs=pd.Series(accepted,index=obs.index,dtype=float,name='level')
    audit.update(gross_rate_jumps=jumps,stale_after_72h=stale,accepted=int(obs.notna().sum()))
    # Bins are right-closed: an 08:30 reading never affects the 08:00 features.
    hourly=obs.resample('1h',label='right',closed='right').last().to_frame()
    hourly['observed_max']=obs.resample('1h',label='right',closed='right').max()
    hourly['station']=station
    hourly.index.name='time'
    return hourly,audit


def causal_features(hourly,station):
    """Every operation is trailing or shifted backwards. Never use targets."""
    meta=STATIONS[station]
    observed=hourly.level
    past=observed.ffill(limit=2)
    frame=pd.DataFrame(index=hourly.index)
    frame['station']=station
    frame['current_level']=past
    frame['current_relative_warning']=past-meta['warning']
    frame['warning_danger_gap']=meta['danger']-meta['warning']
    for lag in [1,3,6,12,24,48,72]:frame[f'change_{lag}h']=past-past.shift(lag)
    frame['rise_6h']=frame.change_6h.clip(lower=0)
    frame['rise_24h']=frame.change_24h.clip(lower=0)
    frame['acceleration_6h_vs_24h']=frame.change_6h/6-frame.change_24h/24
    for window in [6,24,72]:
        rolling=past.rolling(window,min_periods=int(np.ceil(window*.9)))
        frame[f'mean_change_{window}h']=rolling.mean()-past
        frame[f'max_change_{window}h']=rolling.max()-past
        frame[f'min_change_{window}h']=rolling.min()-past
        frame[f'std_{window}h']=rolling.std(ddof=0)
    frame['month_sin']=np.sin(2*np.pi*frame.index.month/12)
    frame['month_cos']=np.cos(2*np.pi*frame.index.month/12)
    frame['day_sin']=np.sin(2*np.pi*frame.index.dayofyear/365.25)
    frame['day_cos']=np.cos(2*np.pi*frame.index.dayofyear/365.25)
    frame['past_observed_hours']=observed.notna().rolling(72,min_periods=72).sum()
    # Need a real current observation and adequate recent history. Missing or
    # stale input produces unavailable, never an invented normal forecast.
    frame['input_eligible']=(observed.notna() & frame.past_observed_hours.ge(65) &
                              frame[NUMERIC_FEATURES].notna().all(axis=1))
    return frame


def supervised_rows(hourly,station):
    frame=causal_features(hourly,station)
    # This forward-looking calculation is label construction ONLY. It is never
    # used by causal_features or forecast_at.
    future=pd.concat([hourly.observed_max.shift(-h) for h in range(1,25)],axis=1)
    frame['future_observed_hours']=future.notna().sum(axis=1)
    frame['target_peak']=future.max(axis=1).where(frame.future_observed_hours.eq(24))
    frame['target_change']=frame.target_peak-frame.current_level
    frame['label_end']=frame.index+pd.Timedelta(hours=24)
    frame['warning_level']=STATIONS[station]['warning']
    frame['danger_level']=STATIONS[station]['danger']
    daily=frame.loc[(frame.index.hour==8)&(frame.index.minute==0)].copy()
    daily.index.name='time'
    return daily


def load_hourly(station):
    frame=pd.read_csv(DATA/f'processed/{station}_hourly.csv',parse_dates=['time']).set_index('time')
    return frame


def load_bundle():
    return joblib.load(EXPERIMENT/'forecast_bundle.joblib')


def forecast_at(station,issue_time,bundle=None,hourly=None):
    """Inference requires past observations only, with no future target columns."""
    if station not in STATIONS:raise ValueError('Unsupported station.')
    issue=pd.Timestamp(issue_time)
    bundle=load_bundle() if bundle is None else bundle
    if issue<=pd.Timestamp(bundle['training_end']):
        raise ValueError('A forecast time must follow model training.')
    if issue.hour!=8 or issue.minute!=0:raise ValueError('Forecast issue time is 08:00 IST.')
    hourly=load_hourly(station) if hourly is None else hourly
    history=hourly.loc[hourly.index<=issue].copy()
    if issue not in history.index:raise ValueError('No observation at the selected issue time.')
    # Preserve the same trailing-statistics calculation path as dataset creation.
    # Truncating the prefix can introduce rounding differences near tree splits.
    row=causal_features(history,station).loc[[issue]]
    if not bool(row.input_eligible.iloc[0]):raise ValueError('Not enough recent observations for a reliable input window.')
    if bundle['predictor_type']=='persistence':
        peak=float(row.current_level.iloc[0])
    elif bundle['predictor_type']=='trend':
        peak=float(row.current_level.iloc[0]+np.maximum(row.change_6h.iloc[0],0)*2)
    elif bundle['predictor_type']=='previous_max':
        peak=float(row.current_level.iloc[0]+row.max_change_24h.iloc[0])
    else:
        delta=float(bundle['pipeline'].predict(row[FEATURES])[0])
        peak=float(row.current_level.iloc[0]+delta)
    error=float(bundle['error_ranges'][station])
    current=float(row.current_level.iloc[0]);meta=STATIONS[station]
    return {'station':station,'issue_time':issue.isoformat(),'horizon_end':(issue+pd.Timedelta(hours=24)).isoformat(),
            'current_level':current,'predicted_peak':peak,'lower':peak-error,'upper':peak+error,
            'status':stage_category(peak,station),'current_status':stage_category(current,station),
            'warning':meta['warning'],'danger':meta['danger'],'change_6h':float(row.change_6h.iloc[0]),
            'past_observed_hours':int(row.past_observed_hours.iloc[0]),'source':'CWC manual observations',
            'mode':'Historical forecast replay; not a live alert'}
