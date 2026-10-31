"""Exercise the redesigned UI's real API integration and state transitions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
checks = []


def check(name, condition):
    assert condition, name
    checks.append(name)


def clean(page, name):
    check(name, not page.exception and not page.error)


unchanged = subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'api.py',
    'flood_fast.py', 'flood_model.pkl', 'encoders.pkl', 'feature_columns.pkl', 'high_risk_locations.csv'],
    cwd=ROOT, capture_output=True)
check('Original API, model, encoders, schema and map data unchanged', unchanged.returncode == 0)
check('Original model fingerprint preserved', hashlib.sha256((ROOT / 'flood_model.pkl').read_bytes()).hexdigest() ==
      'aad88a12f4645f3703c996641a15ce40603f6d5a608918fad4f37496d2f89395')
health = requests.get('http://127.0.0.1:8000/health', timeout=10)
check('XGBoost API online with original 13 features', health.status_code == 200 and health.json()['features'] == 13)
check('Streamlit online', requests.get('http://127.0.0.1:8501/_stcore/health', timeout=10).status_code == 200)

page = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
clean(page, 'Rainfall landing page renders')
check('Exactly two task-based navigation choices', page.radio(key='workspace_page').options == ['Rainfall history', 'Check flood risk'])
check('Rainfall history is the landing page', page.radio(key='workspace_page').value == 'Rainfall history')
check('No sidebar content is generated', not page.sidebar.children)
check('Map implementation is preserved as an archive', (ROOT / 'legacy/risk_map.py').exists())
check('No model panels are in the public landing page', not any(e.label == 'About the model' for e in page.expander) and
      not any('XGBoost' in e.proto.body or 'Model connected' in e.proto.body for e in page.get('html')))
page.radio(key='workspace_page').set_value('Check flood risk').run(timeout=30)
clean(page, 'Risk check opens from top navigation')
check('All 14 raw inputs remain available', len(page.slider) + len(page.selectbox) == 14)
check('Complete readable category choices', [len(box.options) for box in page.selectbox] == [6, 11, 25])
check('Human category labels replace numeric codes', page.selectbox[2].options[3] == 'Arenosols')
check('Initial screen does not invent a prediction', 'assessment' not in page.session_state)
page.button[0].click().run(timeout=45)
clean(page, 'Real model assessment renders')
first = page.session_state['assessment'].copy()
check('Default prediction matches pre-redesign score', first['result']['flood_risk_score'] == .0779)
check('Gauge and rainfall comparison rendered', len(page.get('plotly_chart')) == 2)
spec = json.loads(page.get('plotly_chart')[1].proto.spec)
check('Comparison highlights actual submitted model score', spec['data'][1]['x'] == [1200] and
      abs(spec['data'][1]['y'][0] - 7.79) < 1e-8)
check('Assessment export available', len(page.get('download_button')) == 1)
check('Advanced conditions are available in a collapsed section', any(
    e.label == 'Land & weather details' and not e.proto.expanded for e in page.expander))
explore_key = 'explore_' + first['time']
page.slider(key=explore_key).set_value(2400).run(timeout=45)
clean(page, 'Interactive rainfall explorer renders a real comparison')
spec = json.loads(page.get('plotly_chart')[1].proto.spec)
expected = requests.post('http://127.0.0.1:8000/predict', json={**first['inputs'],
    'annual_precipitation': 2400}, timeout=15).json()['flood_risk_score']
check('Explorer plots the exact API response for changed rainfall', spec['data'][2]['x'] == [2400] and
      abs(spec['data'][2]['y'][0] - expected * 100) < 1e-8)
check('Explorer leaves the checked assessment and history unchanged', page.session_state['assessment'] == first and
      len(page.session_state['assessments']) == 1)

page.slider(key='input_annual_precipitation').set_value(100)
page.button[0].click().run(timeout=30)
check('Inconsistent rainfall inputs rejected', len(page.error) == 1 and not page.exception)
check('Invalid inputs preserve last assessment', page.session_state['assessment'] == first)
page.get('button_group')[0].set_value('Wetter landscape').run(timeout=30)
check('Example scenario populates all inputs', page.slider(key='input_annual_precipitation').value == 2600 and
      page.selectbox(key='input_soil_type').value == 9)
page.button[0].click().run(timeout=45)
clean(page, 'Wetter scenario assessed without exceptions')
wet = page.session_state['assessment'].copy()
check('Session comparison records distinct genuine assessments', len(page.session_state['assessments']) == 2 and
      wet['result']['flood_risk_score'] != first['result']['flood_risk_score'] and len(page.get('plotly_chart')) == 3)
page.get('button_group')[0].set_value(None).run(timeout=30)
clean(page, 'Example control can be deselected')
with patch('requests.post', side_effect=requests.ConnectionError('test-only unavailable service')):
    page.button[0].click().run(timeout=30)
    check('Prediction outage shown without crash', len(page.error) == 1 and not page.exception)
    check('Outage preserves last successful result', page.session_state['assessment'] == wet)

page.radio(key='workspace_page').set_value('Rainfall history').run(timeout=30)
clean(page, 'Rainfall page works after risk checks')
check('Rainfall charts stay separate from risk results', len(page.get('plotly_chart')) == 3)
page.radio(key='workspace_page').set_value('Check flood risk').run(timeout=30)
check('Assessment and its inputs survive module navigation', page.session_state['assessment'] == wet and
      page.slider(key='input_annual_precipitation').value == 2600)

page.slider(key='input_annual_precipitation').set_value(0)
page.slider(key='input_precipitation_of_wettest_month').set_value(0)
page.button[0].click().run(timeout=45)
clean(page, 'Zero rainfall renders an assessment without an invalid equal-bound slider')
check('Zero-rainfall explorer preserves its exact submitted value', page.session_state['assessment']['inputs']['annual_precipitation'] == 0)

with patch.dict(os.environ, {'RISKRADAR_API_URL': 'http://127.0.0.1:1'}):
    offline = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
    offline.radio(key='workspace_page').set_value('Check flood risk').run(timeout=30)
    check('Offline service disables assessment without crashing', not offline.exception and offline.button[0].disabled)
    offline.radio(key='workspace_page').set_value('Rainfall history').run(timeout=30)
    clean(offline, 'Rainfall history remains available when risk checks are offline')

report = {'passed': len(checks), 'failed': 0, 'checks': checks,
          'default_score': first['result']['flood_risk_score'], 'wetter_score': wet['result']['flood_risk_score'],
          'model_retrained': False, 'api_logic_changed': False}
(ROOT / '.run/ui-verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
