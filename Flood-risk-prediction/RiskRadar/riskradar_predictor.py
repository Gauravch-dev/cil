"""Validated local inference using the original RiskRadar model and schema.

Preserves the original feature equations, encoder order and risk bands. The
saved model estimates relative annual high-discharge susceptibility, not an
issued short-term flood warning or calibrated flood probability.
"""
from pathlib import Path
from functools import lru_cache
from numbers import Integral
import math
import joblib
import numpy as np

ROOT=Path(__file__).resolve().parent
DEFAULTS=dict(annual_precipitation=1200.,precipitation_of_wettest_month=300.,precipitation_seasonality=50.,
    drainage_density=2.5,drainage_texture=10.,basin_relief=500.,annual_mean_temperature=20.,
    temperature_seasonality=500.,curve_number_amcii=70.,ruggedness_number=.5,infiltration_number=5.,
    climate_type=2,landcover_type=1,soil_type=3)
EXAMPLES={'Typical':DEFAULTS,
    'Drier':{**DEFAULTS,'annual_precipitation':400.,'precipitation_of_wettest_month':120.,
        'precipitation_seasonality':25.,'drainage_density':1.,'curve_number_amcii':50.},
    'Wetter':{**DEFAULTS,'annual_precipitation':2600.,'precipitation_of_wettest_month':800.,
        'precipitation_seasonality':85.,'drainage_density':6.,'drainage_texture':30.,'basin_relief':2000.,
        'annual_mean_temperature':27.,'curve_number_amcii':90.,'ruggedness_number':2.,
        'infiltration_number':10.,'climate_type':1,'landcover_type':0,'soil_type':9}}
CATEGORIES=['climate_type','landcover_type','soil_type']
FEATURES=['precipitation_risk_index','drainage_risk_index','annual_mean_temperature','temperature_seasonality',
    'annual_precipitation','curve_number_amcii','drainage_density','basin_relief','ruggedness_number',
    'infiltration_number','climate_type','landcover_type','soil_type']


@lru_cache(maxsize=1)
def artifacts():
    model=joblib.load(ROOT/'flood_model.pkl')
    model.set_params(device='cpu',n_jobs=2)
    encoders=joblib.load(ROOT/'encoders.pkl')
    columns=list(joblib.load(ROOT/'feature_columns.pkl'))
    if columns!=FEATURES:raise ValueError('The saved predictor and input schema do not match.')
    return model,encoders,columns


def category_choices():
    return {key:['Unspecified' if str(value)=='nan' else str(value) for value in artifacts()[1][key].classes_]
            for key in CATEGORIES}


def validate(values):
    missing=[key for key in DEFAULTS if key not in values or values[key] is None]
    if missing:raise ValueError('Complete all the rainfall and area fields before checking risk.')
    clean={}
    for key in DEFAULTS:
        value=values[key]
        if isinstance(value,(bool,np.bool_)):raise ValueError('Use numbers for rainfall and area measurements.')
        if key in CATEGORIES:
            if not isinstance(value,Integral) or value<0 or value>=len(artifacts()[1][key].classes_):
                raise ValueError('Choose climate, land cover and soil from the supplied options.')
            clean[key]=int(value);continue
        try:value=float(value)
        except (TypeError,ValueError):raise ValueError('Use valid numbers for rainfall and area measurements.') from None
        if not math.isfinite(value) or abs(value)>np.finfo(np.float32).max:
            raise ValueError('Enter finite rainfall and area measurements.')
        if key!='annual_mean_temperature' and value<0:
            raise ValueError('Rainfall, terrain and variation values cannot be negative.')
        clean[key]=value
    if not 0<=clean['curve_number_amcii']<=100:
        raise ValueError('Runoff potential must be between 0 and 100.')
    if not -90<=clean['annual_mean_temperature']<=60:
        raise ValueError('Check the mean temperature. Enter degrees Celsius, between -90 and 60.')
    annual=clean['annual_precipitation'];wettest=clean['precipitation_of_wettest_month']
    if wettest>annual+1e-7:
        raise ValueError('The wettest month cannot have more rainfall than the entire year.')
    if annual>12*wettest+1e-7:
        raise ValueError('Annual rainfall cannot exceed twelve times the wettest-month rainfall.')
    return clean


def build_features(values):
    data=validate(values)
    feature_map={key:data[key] for key in FEATURES if key in data}
    feature_map['precipitation_risk_index']=(min(data['annual_precipitation'],3000)*.4+
        min(data['precipitation_of_wettest_month'],1000)*.3+min(data['precipitation_seasonality'],100)*.3)
    feature_map['drainage_risk_index']=(min(data['drainage_density'],10)*.5+
        min(data['drainage_texture'],50)*.3+min(data['basin_relief'],5000)*.2)
    return np.array([[feature_map[key] for key in artifacts()[2]]],dtype=float)


def risk_band(score):
    if not math.isfinite(score) or not 0<=score<=1:raise ValueError('Risk score is invalid.')
    return 'HIGH' if score>.7 else 'MEDIUM' if score>.4 else 'LOW'


def predict(values):
    features=build_features(values)
    score=float(artifacts()[0].predict_proba(features)[0,1])
    return {'flood_risk_score':round(score,4),'risk_level':risk_band(score)}
