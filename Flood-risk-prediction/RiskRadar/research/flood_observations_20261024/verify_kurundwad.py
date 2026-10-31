"""Read-only station/threshold feasibility queries against official CKAN data."""
from pathlib import Path
import json
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = 'https://nwdp.nwic.gov.in/api/3/action/'
HIST = '9579b739-ccb3-4c5a-b571-e3720cef442d'
RECENT = '4ddede56-5e49-4956-bb6f-df3786bf601f'
session = requests.Session()

def save(name, action, params):
    response = session.get(BASE+action, params=params, timeout=(15, 45))
    if not response.ok:
        (ROOT/f'{name}_error.txt').write_text(response.text[:3000], encoding='utf-8')
        print(name, 'HTTP', response.status_code, flush=True)
        return {}
    payload = response.json()
    (ROOT/f'{name}.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    result = payload.get('result', {})
    print(name, json.dumps({'success': payload.get('success'), 'total': result.get('total'),
                            'records': result.get('records', [])[:8], 'error': payload.get('error')}, ensure_ascii=False), flush=True)
    return payload

for label, resource in [('historical', HIST), ('recent', RECENT)]:
    payload = save(f'kurundwad_{label}_verified', 'datastore_search', {
        'resource_id': resource, 'filters': json.dumps({'Station': 'KURUNDWAD'}), 'limit': 5})
    if not payload.get('success'):
        continue
    total = payload['result']['total']
    records = []
    columns = ['Station', 'Data Acquisition Time', 'River Water Level Telemetry Hourly (meter)',
               'RL_of_zeroGauge', 'Latitude', 'Longitude']
    for offset in range(0, min(total, 50000), 5000):
        response = session.get(BASE+'datastore_search', params={
            'resource_id': resource, 'filters': json.dumps({'Station': 'KURUNDWAD'}),
            'fields': ','.join(columns), 'limit': min(5000, total-offset), 'offset': offset,
        }, timeout=(15, 45))
        response.raise_for_status()
        block = response.json()
        if not block.get('success'):
            break
        records.extend(block['result']['records'])
    frame = pd.DataFrame(records)
    if frame.empty:
        continue
    frame.to_csv(ROOT/f'kurundwad_{label}_observations.csv', index=False)
    times = pd.to_datetime(frame['Data Acquisition Time'], format='%d-%m-%Y %H:%M', errors='coerce')
    level = pd.to_numeric(frame['River Water Level Telemetry Hourly (meter)'], errors='coerce')
    info = {'official_resource_id': resource, 'api_total_records': total, 'downloaded_records': len(frame),
            'complete_retrieval': len(frame) == total, 'first_time': str(times.min()), 'last_time': str(times.max()),
            'invalid_times': int(times.isna().sum()), 'invalid_numeric_levels': int(level.isna().sum()),
            'below_zero_levels': int(level.lt(0).sum()), 'min_level_raw': float(level.min()),
            'max_level_raw': float(level.max()), 'at_or_above_indofloods_danger_raw': int(level.ge(537.16).sum()),
            'below_indofloods_danger_raw': int(level.lt(537.16).sum()),
            'warning': 'RAW source counts only. Datum, outliers, duplicate timestamps, coverage and threshold validity need audit before labels.'}
    above = frame.loc[level.ge(537.16), columns].head(5).to_dict('records')
    info['above_threshold_examples_raw'] = above
    (ROOT/f'kurundwad_{label}_summary.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print('OBSERVATION_SUMMARY', label, json.dumps(info, ensure_ascii=False), flush=True)

payload = save('cwc_rainfall_catalog', 'package_search', {'q': 'title:rainfall AND title:CWC', 'rows': 15})
for item in payload.get('result', {}).get('results', []):
    matches = [r for r in item.get('resources', []) if 'maharashtra' in r.get('name', '').lower()]
    if matches:
        print('MAHARASHTRA_RAINFALL', json.dumps({'dataset': item.get('name'), 'resources': [
            {k: r.get(k) for k in ['id', 'name', 'url', 'datastore_active', 'size']} for r in matches]}, ensure_ascii=False), flush=True)
