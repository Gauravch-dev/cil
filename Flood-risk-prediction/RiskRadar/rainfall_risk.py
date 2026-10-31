"""Rainfall-based hazard indicator using the existing near-Mumbai ERA5 archive.

Low/Medium/High are categories of the following three-day rainfall total, using
training-derived local wet-window percentiles. They are not flood labels.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'data/weather/IN_0000317_era5_2000_2025.json'
RUN=ROOT/'experiments/rainfall_risk_v1'
LABELS=['Low','Medium','High']
NUMERIC=[f'rain_lag_{i}' for i in range(14)]+[f'rain_sum_{i}d' for i in [3,5,7,14]]+[
    'rain_max_7d','rain_std_7d','rain_max_14d','rain_std_14d','rainy_days_7d','rainy_days_14d',
    'consecutive_rain_days','rain_change','month_sin','month_cos','day_sin','day_cos']


def weather():
    document=json.loads(SOURCE.read_text())
    if document['response']['daily_units']['rain_sum']!='mm':raise ValueError('Rainfall units must be mm.')
    daily=document['response']['daily']
    frame=pd.DataFrame({'date':pd.to_datetime(daily['time']),'rain_mm':daily['rain_sum']}).set_index('date')
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:raise ValueError('Invalid archive dates.')
    frame['rain_mm']=pd.to_numeric(frame.rain_mm,errors='coerce')
    frame['rain_mm']=frame.rain_mm.where(np.isfinite(frame.rain_mm)&frame.rain_mm.ge(0))
    return frame.reindex(pd.date_range('2000-01-01','2025-12-31',name='date'))


def features(frame):
    rain=frame.rain_mm;out=pd.DataFrame(index=frame.index)
    for lag in range(14):out[f'rain_lag_{lag}']=rain.shift(lag)
    for window in [3,5,7,14]:out[f'rain_sum_{window}d']=rain.rolling(window,min_periods=window).sum()
    for window in [7,14]:
        rolling=rain.rolling(window,min_periods=window)
        out[f'rain_max_{window}d']=rolling.max();out[f'rain_std_{window}d']=rolling.std(ddof=0)
        out[f'rainy_days_{window}d']=rain.gt(0).rolling(window,min_periods=window).sum().where(rolling.count().eq(window))
    wet=rain.gt(0)&rain.notna()
    out['consecutive_rain_days']=wet.groupby((~wet).cumsum()).cumsum().clip(upper=14)
    out['rain_change']=rain-rain.shift(1)
    out['month_sin']=np.sin(2*np.pi*out.index.month/12);out['month_cos']=np.cos(2*np.pi*out.index.month/12)
    out['day_sin']=np.sin(2*np.pi*out.index.dayofyear/365.25);out['day_cos']=np.cos(2*np.pi*out.index.dayofyear/365.25)
    return out[NUMERIC]


def category(amount,thresholds):
    values=np.asarray(amount,dtype=float)
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Finite nonnegative rainfall is required.')
    return np.where((values>0)&(values>=thresholds['high']),2,
                    np.where((values>0)&(values>=thresholds['medium']),1,0))


def classify(probabilities,cutoffs):
    probability=np.asarray(probabilities)
    return np.where(probability[:,2]>=cutoffs['high'],2,
                    np.where(probability[:,1]>=cutoffs['medium'],1,0))


def baseline_probabilities(rows,name,bundle):
    if name=='persistence':return np.eye(3)[category(rows.rain_sum_3d,bundle['thresholds'])]
    if name=='always_low':return np.tile([1.,0.,0.],(len(rows),1))
    return np.array([bundle['seasonal'][str(date.month)] for date in rows.index])


def ml_bundle():
    """Best ML candidate by FROZEN validation results; do not replace the winner."""
    selection=json.loads((RUN/'selection.json').read_text())
    candidate=max((row for row in selection['candidates'] if row['kind']=='ml'),
                  key=lambda row:(row['validation']['macro_f1'],row['validation']['high_f2']))
    bundle=joblib.load(RUN/'forecast_bundle.joblib')
    return {**bundle,'name':candidate['name'],'kind':'ml',
            'pipeline':joblib.load(RUN/f'models/{candidate["name"]}.joblib'),
            'cutoffs':candidate['validation']['cutoffs']}


def predict(issue,rainfall=None,bundle=None):
    """Use exactly 14 past readings; real future values are never required."""
    bundle=joblib.load(RUN/'forecast_bundle.joblib') if bundle is None else bundle
    date=pd.Timestamp(issue).normalize()
    if date<=pd.Timestamp('2015-12-31'):raise ValueError('Use a date after training ended.')
    index=pd.date_range(date-pd.Timedelta(days=13),date,name='date')
    if rainfall is None:
        archive=weather()
        if date not in archive.index:raise ValueError('No saved rainfall for that date. Use a what-if simulation.')
        # Retain the same rolling-statistics arithmetic path as training.
        history=archive.loc[archive.index<=date].copy()
    else:
        values=np.asarray(rainfall,dtype=float)
        if values.shape!=(14,) or not np.isfinite(values).all() or (values<0).any():
            raise ValueError('Enter 14 finite nonnegative daily rainfall amounts.')
        history=pd.DataFrame({'rain_mm':values},index=index)
    row=features(history).tail(1)
    if row.isna().any().any():raise ValueError('Fourteen complete daily rainfall readings are required.')
    probability=(bundle['pipeline'].predict_proba(row) if bundle['kind']=='ml' else
                 baseline_probabilities(row,bundle['name'],bundle))
    code=int(classify(probability,bundle['cutoffs'])[0])
    return {'location':'Near Mumbai · existing ERA5 weather point','issue_date':str(date.date()),
            'period_start':str((date+pd.Timedelta(days=1)).date()),'period_end':str((date+pd.Timedelta(days=3)).date()),
            'category':LABELS[code],'category_code':code,'rain_today':float(history.rain_mm.iloc[-1]),
            'rain_3d':float(row.rain_sum_3d.iloc[0]),'rain_7d':float(row.rain_sum_7d.iloc[0]),
            'rain_14d':float(row.rain_sum_14d.iloc[0]),'peak_14d':float(history.rain_mm.tail(14).max()),
            'thresholds_mm':bundle['thresholds'],'model':bundle['name'],
            'meaning':'Rainfall-based indicator for next three-day rainfall; not flood occurrence',
            'mode':'What-if simulation' if rainfall is not None else 'Historical replay'}


def historical_indicators(frame,baseline=None):
    """Descriptive heavy-rain and monthly anomalies against a fixed past baseline."""
    reference=frame.loc[frame.index.year<=2015] if baseline is None else baseline
    wet=reference.rain_mm.loc[reference.rain_mm.gt(0)].dropna()
    threshold=float(wet.quantile(.95)) if len(wet) else np.nan
    flags=(frame.rain_mm.gt(threshold).where(frame.rain_mm.notna()) if np.isfinite(threshold)
           else pd.Series(np.nan,index=frame.index))
    monthly=frame.rain_mm.resample('MS').sum(min_count=1)
    counts=frame.rain_mm.resample('MS').count()
    monthly=monthly.where(counts.eq(monthly.index.days_in_month))
    historical=reference.rain_mm.resample('MS').sum(min_count=1)
    historical=historical.where(reference.rain_mm.resample('MS').count().eq(historical.index.days_in_month))
    mean=historical.groupby(historical.index.month).mean();std=historical.groupby(historical.index.month).std()
    normal=monthly.index.month.map(mean)
    spread=monthly.index.month.map(std)
    result=pd.DataFrame({'rain_mm':monthly,'baseline_mm':np.asarray(normal),
                         'anomaly_mm':monthly.to_numpy()-np.asarray(normal),
                         'z_score':(monthly.to_numpy()-np.asarray(normal))/np.where(np.asarray(spread)>0,spread,np.nan)})
    return {'heavy_threshold_mm':threshold,'heavy_days':flags,'monthly':result}
