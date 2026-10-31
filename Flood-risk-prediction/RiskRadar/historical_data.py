"""Reproducible GloFlo audit and dated ERA5 rainfall, independent of prediction."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
SOURCE_URL = 'https://zenodo.org/records/7158028/files/kuntla_flooddatabase.csv?download=1'
SOURCE_MD5 = '4e0d207c21c54577bd5a1e176a96ed3e'
ARCHIVE_URL = 'https://archive-api.open-meteo.com/v1/archive'
START, END = '2000-01-01', '2025-12-31'


def prepare_source():
    raw = ROOT / 'data/raw/kuntla_flooddatabase.csv'
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        response = requests.get(SOURCE_URL, timeout=(10, 120))
        response.raise_for_status()
        if hashlib.md5(response.content).hexdigest() != SOURCE_MD5:
            raise ValueError('Source checksum mismatch; the downloaded file was not saved.')
        raw.write_bytes(response.content)
    if hashlib.md5(raw.read_bytes()).hexdigest() != SOURCE_MD5:
        raise ValueError('The local GloFlo source checksum does not match the published file.')
    source = pd.read_csv(raw, low_memory=False)
    required = ['gsim_code', 'latitude', 'longitude', 'year', 'countryname',
                'Annual Precipitation', 'Annual peak discharge', 'Unit peak discharge']
    if not set(required).issubset(source.columns):
        raise ValueError('The source schema is incomplete.')
    catalog = source[['gsim_code', 'latitude', 'longitude', 'countryname', 'Annual Precipitation']].drop_duplicates()
    if catalog.gsim_code.duplicated().any():
        raise ValueError('A station has conflicting metadata; review the source before merging.')
    catalog = catalog.rename(columns={'gsim_code': 'station_id', 'countryname': 'country',
                                     'Annual Precipitation': 'climate_annual_precip_mm'})
    valid = (catalog.latitude.between(-90, 90) & catalog.longitude.between(-180, 180)
             & catalog.country.notna())
    catalog = catalog.loc[valid].sort_values(['country', 'station_id']).reset_index(drop=True)
    year = pd.to_numeric(source.year, errors='coerce')
    valid_year = year.notna() & year.between(1800, 2025) & (year % 1 == 0)
    events = source.loc[valid_year, ['gsim_code', 'Annual peak discharge', 'Unit peak discharge']].copy()
    events['year'] = year[valid_year].astype(int)
    events = events.rename(columns={'gsim_code': 'station_id', 'Annual peak discharge': 'annual_peak_discharge',
                                     'Unit peak discharge': 'unit_peak_discharge'})
    events = events.merge(catalog[['station_id', 'country', 'latitude', 'longitude']],
                          on='station_id', how='inner', validate='many_to_one')
    processed = ROOT / 'data/processed'
    processed.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(processed / 'stations.csv', index=False)
    events.to_csv(processed / 'recorded_peaks.csv', index=False)
    report = {
        'source': 'Global Flood characterization (GloFlo), Kuntla, Saharia & Kirstetter (2022)',
        'source_url': SOURCE_URL, 'source_record': 'https://zenodo.org/records/7158028',
        'source_md5': SOURCE_MD5, 'source_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
        'raw_rows': len(source), 'raw_columns': len(source.columns), 'stations': len(catalog),
        'countries': int(catalog.country.nunique()), 'valid_year_records': len(events),
        'invalid_year_records_excluded': int((~valid_year).sum()),
        'year_min': int(events.year.min()), 'year_max': int(events.year.max()),
        'exact_duplicate_rows': int(source.duplicated().sum()),
        'fully_missing_columns': source.columns[source.isna().all()].tolist(),
        'missing_values_by_column': {c: int(v) for c, v in source.isna().sum().items()},
        'stations_with_changing_annual_precipitation': int(source.groupby('gsim_code')['Annual Precipitation'].nunique().gt(1).sum()),
        'contains_binary_flood_labels': False,
        'rainfall_is_time_series': False,
        'model_retrained': False,
        'validation_status': 'Original classifier cannot be independently evaluated: no binary flood/no-flood target or original label-generation pipeline supplied.',
        'weather': {'provider': 'Open-Meteo', 'dataset': 'ERA5', 'variable': 'rain_sum',
                    'unit': 'mm', 'timezone': 'GMT', 'start': START, 'end': END,
                    'docs': 'https://open-meteo.com/en/docs/historical-weather-api'},
    }
    (processed / 'data_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def station_catalog():
    return pd.read_csv(ROOT / 'data/processed/stations.csv')


def history_path(station_id):
    # IDs must exist in the fixed source catalog; no user-controlled file paths.
    if station_id not in set(station_catalog().station_id):
        raise ValueError('Unknown source station.')
    return ROOT / 'data/weather' / f'{station_id}_era5_2000_2025.json'


def validate_weather(document):
    daily = document['response']['daily']
    if document['response'].get('daily_units', {}).get('rain_sum') != 'mm':
        raise ValueError('Unexpected rainfall units.')
    dates = pd.to_datetime(daily['time'], errors='raise')
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('Weather dates must be unique and ordered.')
    values = pd.to_numeric(pd.Series(daily['rain_sum']), errors='coerce')
    if len(dates) != len(values) or len(dates) == 0:
        raise ValueError('Incomplete weather response.')
    index = pd.date_range(START, END, freq='D')
    if not dates.isin(index).all():
        raise ValueError('Weather dates are outside the requested archive range.')
    # Preserve missing/invalid values as NaN. Missing rain must never become dry weather.
    values = values.where(np.isfinite(values) & values.ge(0))
    frame = pd.DataFrame({'date': dates, 'rain_mm': values.to_numpy()}).set_index('date')
    return frame.reindex(index).rename_axis('date').reset_index()


def fetch_weather(station_id, overwrite=False):
    path = history_path(station_id)
    if path.exists() and not overwrite:
        document = json.loads(path.read_text(encoding='utf-8'))
        validate_weather(document)
        return document
    station = station_catalog().set_index('station_id').loc[station_id]
    params = dict(latitude=float(station.latitude), longitude=float(station.longitude),
                  start_date=START, end_date=END, daily='rain_sum', models='era5', timezone='GMT',
                  precipitation_unit='mm')
    response = requests.get(ARCHIVE_URL, params=params, timeout=(10, 60))
    response.raise_for_status()
    document = {'station_id': station_id, 'provider': 'Open-Meteo', 'dataset': 'ERA5',
                'requested': params, 'fetched_utc': datetime.now(timezone.utc).isoformat(),
                'response': response.json(), 'request_url': response.url}
    validate_weather(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(document), encoding='utf-8')
    temporary.replace(path)
    return document


def read_weather(station_id):
    path = history_path(station_id)
    if not path.exists():
        return None, None
    document = json.loads(path.read_text(encoding='utf-8'))
    return validate_weather(document), document


def aggregate_rainfall(frame, frequency):
    series = frame.set_index('date').rain_mm
    # A complete calendar period is required to report a total.
    resampler = series.resample(frequency)
    totals = resampler.sum(min_count=1)
    expected = totals.index.days_in_month if frequency == 'MS' else np.where(totals.index.is_leap_year, 366, 365)
    return totals.where(resampler.count().to_numpy() == expected)


def default_station(catalog):
    india = catalog.loc[catalog.country.eq('India')]
    if len(india):
        # A geographic default near Mumbai, independent of scores or outcomes.
        return india.loc[((india.latitude - 19.076) ** 2 + (india.longitude - 72.878) ** 2).idxmin(), 'station_id']
    return catalog.iloc[0].station_id
