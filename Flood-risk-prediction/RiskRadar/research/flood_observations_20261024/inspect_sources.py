"""Read small public previews only; do not download basin-sized archives."""
from pathlib import Path
import json
from datetime import datetime, timezone
import requests

ROOT = Path(__file__).resolve().parent
BASE = 'https://nwdp.nwic.gov.in'
PACKAGE = '68600163-c5a0-4327-aa1a-fa7157b86cce'
RESOURCES = {
    'krishna_hourly_1991_2020': ('9579b739-ccb3-4c5a-b571-e3720cef442d', 'rwl_tele_hr_cwc_007_1991_2020.csv'),
    'godavari_hourly_2021_2025': ('4d139325-8371-422f-acef-d34ccec4300e', 'rwl_tel_hr_cwc_006_2021_2025.csv'),
}
session = requests.Session()
summary = {'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'resources': {}}
for name, (resource, filename) in RESOURCES.items():
    url = f'{BASE}/dataset/{PACKAGE}/resource/{resource}/download/{filename}'
    info = {'url': url, 'preview_only': True, 'byte_limit': 65536}
    try:
        with session.get(url, stream=True, timeout=(15, 35)) as response:
            info['http_status'] = response.status_code
            info['content_type'] = response.headers.get('Content-Type')
            response.raise_for_status()
            chunk = next(response.iter_content(chunk_size=65536))
            # Keep only complete lines, not a misleading truncated CSV row.
            chunk = chunk.rsplit(b'\n', 1)[0]+b'\n'
            (ROOT/f'{name}_preview.csv').write_bytes(chunk)
            info['saved_bytes'] = len(chunk)
            info['first_lines'] = chunk.decode('utf-8-sig', errors='replace').splitlines()[:3]
    except Exception as error:
        info['error'] = str(error)
    summary['resources'][name] = info
    print(json.dumps({name: info}, ensure_ascii=False), flush=True)
for action, params in [('resource_show', {'id': RESOURCES['krishna_hourly_1991_2020'][0]}),
                       ('datastore_search', {'resource_id': RESOURCES['krishna_hourly_1991_2020'][0], 'limit': 3})]:
    try:
        response = session.get(f'{BASE}/api/3/action/{action}', params=params, timeout=(15, 35))
        info = {'http_status': response.status_code}
        if 'json' in response.headers.get('Content-Type', ''):
            payload = response.json()
            (ROOT/f'nwdp_{action}.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
            info['success'] = payload.get('success')
            if action == 'datastore_search' and payload.get('success'):
                info['total'] = payload['result'].get('total')
                info['fields'] = payload['result'].get('fields')
                info['sample'] = payload['result'].get('records', [])[:1]
        summary[action] = info
        print(action, json.dumps(info, ensure_ascii=False), flush=True)
    except Exception as error:
        summary[action] = {'error': str(error)}
(ROOT/'source_access_checks.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
