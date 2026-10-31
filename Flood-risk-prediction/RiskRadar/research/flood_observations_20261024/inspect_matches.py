"""Verify candidate station matches and official observation resources, no training."""
from pathlib import Path
import json
import requests

ROOT = Path(__file__).resolve().parent
BASE = 'https://nwdp.nwic.gov.in/api/3/action/'
session = requests.Session()
queries = [
    ('krishna_historical_maharashtra', 'datastore_search', {'resource_id': '9579b739-ccb3-4c5a-b571-e3720cef442d',
       'filters': json.dumps({'State': 'Maharashtra'}), 'limit': 5}),
    ('karad_historical', 'datastore_search', {'resource_id': '9579b739-ccb3-4c5a-b571-e3720cef442d',
       'filters': json.dumps({'Station': 'Karad'}), 'limit': 5}),
    ('kurundwad_historical', 'datastore_search', {'resource_id': '9579b739-ccb3-4c5a-b571-e3720cef442d',
       'filters': json.dumps({'Station': 'Kurundwad'}), 'limit': 5}),
    ('krishna_recent_metadata', 'resource_show', {'id': '4ddede56-5e49-4956-bb6f-df3786bf601f'}),
    ('karad_recent', 'datastore_search', {'resource_id': '4ddede56-5e49-4956-bb6f-df3786bf601f',
       'filters': json.dumps({'Station': 'Karad'}), 'limit': 5}),
    ('rainfall_catalog', 'package_search', {'q': 'rainfall', 'rows': 6}),
]
summary = {}
for name, action, params in queries:
    try:
        response = session.get(BASE+action, params=params, timeout=(15, 40))
        response.raise_for_status()
        payload = response.json()
        (ROOT/f'{name}.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        result = payload.get('result', {})
        item = {'success': payload.get('success')}
        if action == 'datastore_search' and payload.get('success'):
            item.update(total=result.get('total'), samples=result.get('records', [])[:2])
        elif action == 'resource_show' and payload.get('success'):
            item.update({k: result.get(k) for k in ['name', 'url', 'size', 'datastore_active', 'last_modified']})
        elif action == 'package_search' and payload.get('success'):
            item['datasets'] = [{'id': r['id'], 'name': r.get('name'), 'title': r.get('title')} for r in result.get('results', [])]
        summary[name] = item
        print(name, json.dumps(item, ensure_ascii=False), flush=True)
    except Exception as error:
        summary[name] = {'error': str(error)}
        print(name, str(error), flush=True)
(ROOT/'candidate_match_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
