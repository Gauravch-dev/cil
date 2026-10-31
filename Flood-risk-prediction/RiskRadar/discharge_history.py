"""Causal annual-flow descriptors, with an explicit separate reference table.

No current-year flow, future flow, validation flow, or test flow is required.
These are historical high-discharge descriptors, not a flood-warning API.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY_FEATURES = [
    'history_count', 'history_age_years', 'history_last_ratio', 'history_mean_ratio',
    'history_median_ratio', 'history_std_ratio', 'history_max_ratio',
    'history_last3_mean_ratio', 'history_exceedance_rate', 'history_log_mean_ratio',
]


def build_history_features(query: pd.DataFrame, reference: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Only use reference observations from the same station in STRICTLY earlier years.

    At evaluation callers must supply the training reference, never concatenate
    validation/test outcomes. Same-year duplicates are collapsed before lookup.
    Missing history stays missing, with count zero; it is not zero discharge.
    """
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError('A positive training-derived discharge cutoff is required.')
    if not np.isfinite(query.numeric_year.to_numpy(dtype=float)).all():
        raise ValueError('A finite target year is required for causal history lookup.')
    if query.gsim_code.isna().any():
        raise ValueError('A station identifier is required for history lookup.')
    ref = reference[['gsim_code', 'numeric_year', 'source_peak_discharge']].copy()
    ref = ref.loc[np.isfinite(ref.numeric_year) & np.isfinite(ref.source_peak_discharge) &
                  ref.source_peak_discharge.ge(0)]
    annual = ref.groupby(['gsim_code', 'numeric_year'], as_index=False).source_peak_discharge.max()
    lookup = {}
    for station, group in annual.groupby('gsim_code', sort=False):
        group = group.sort_values('numeric_year')
        lookup[station] = (group.numeric_year.to_numpy(), group.source_peak_discharge.to_numpy())
    result = query.copy()
    values = []
    last_years = []
    for station, year in zip(query.gsim_code, query.numeric_year):
        years, flows = lookup.get(station, (np.array([]), np.array([])))
        end = np.searchsorted(years, year, side='left')
        past = flows[:end]
        if not len(past):
            values.append([0.]+[np.nan]*(len(HISTORY_FEATURES)-1))
            last_years.append(np.nan)
            continue
        assert years[end-1] < year
        mean = float(past.mean())
        values.append([float(len(past)), float(year-years[end-1]), float(past[-1]/cutoff),
                       mean/cutoff, float(np.median(past)/cutoff), float(past.std(ddof=0)/cutoff),
                       float(past.max()/cutoff), float(past[-3:].mean()/cutoff),
                       float(np.mean(past > cutoff)), float(np.log1p(mean)/np.log1p(cutoff))])
        last_years.append(float(years[end-1]))
    result[HISTORY_FEATURES] = np.asarray(values, dtype=float).reshape(len(query), len(HISTORY_FEATURES))
    result['history_latest_year'] = last_years  # Provenance only, not a model input.
    return result


def predict_annual_discharge(query: pd.DataFrame, bundle_path) -> pd.DataFrame:
    """Run the frozen experts with their actual thresholds and training snapshot.

    Query requires station/year and the static columns listed in bundle metadata.
    No query outcome is needed. Returned scores are not calibrated flood odds.
    """
    import joblib
    bundle = joblib.load(bundle_path)
    if not query.numeric_year.gt(bundle['reference_end_year']).all():
        raise ValueError('Prediction years must follow the model training/reference period.')
    frame = build_history_features(query, bundle['reference'], bundle['target_cutoff'])
    out = frame[['gsim_code', 'numeric_year', 'history_count', 'history_latest_year', 'history_age_years']].copy()
    out['high_discharge_score'] = np.nan
    out['predicted_high_annual_discharge'] = False
    out['expert'] = ''
    out['decision_threshold'] = np.nan
    for route, item in bundle['experts'].items():
        mask = frame.history_count.ge(1) if route == 'history' else frame.history_count.eq(0)
        if not mask.any():
            continue
        missing = [column for column in item['features'] if column not in frame]
        if missing:
            raise ValueError('Missing static predictors: '+', '.join(missing))
        scores = np.round(item['pipeline'].predict_proba(frame.loc[mask, item['features']])[:, 1], 12)
        out.loc[mask, 'high_discharge_score'] = scores
        out.loc[mask, 'predicted_high_annual_discharge'] = scores >= item['threshold']
        out.loc[mask, 'expert'] = route
        out.loc[mask, 'decision_threshold'] = item['threshold']
    return out
