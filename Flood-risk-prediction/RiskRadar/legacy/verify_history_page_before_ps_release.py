"""Check the historical page, using a genuine cached archive and simulated outages."""
import base64
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import requests
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from historical_data import ROOT, aggregate_rainfall, default_station, read_weather, station_catalog

checks = []


def check(name, condition):
    assert condition, name
    checks.append(name)


def plotted_array(value):
    return np.frombuffer(base64.b64decode(value['bdata']), dtype=value['dtype']) if isinstance(value, dict) else np.array(value)


with patch('requests.get', side_effect=requests.ConnectionError('test-only offline service')):
    page = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=45)
    page.radio(key='workspace_page').set_value('Rainfall history').run(timeout=45)
    check('History works with both model and weather services offline', not page.exception and not page.error)
    check('Default source station is a real catalog location', page.selectbox(key='history_station').value == default_station(station_catalog()))
    check('Annual, monthly and daily visuals render', len(page.get('plotly_chart')) == 3)
    check('One simple rainfall download is available', len(page.get('download_button')) == 1)
    check('Technical source panels are absent from the user page', len(page.expander) == 0)
    selected = page.selectbox(key='history_station').value
    frame, _ = read_weather(selected)
    spec = json.loads(page.get('plotly_chart')[0].proto.spec)
    annual = aggregate_rainfall(frame.loc[frame.date.dt.year.between(2016, 2025)], 'YS')
    check('Yearly chart uses the actual dated rainfall totals', np.allclose(plotted_array(spec['data'][0]['y']), annual.to_numpy()))
    page.radio(key='annual_view').set_value('Above / below average').run(timeout=45)
    spec = json.loads(page.get('plotly_chart')[0].proto.spec)
    check('Average comparison changes the plotted data correctly', np.allclose(plotted_array(spec['data'][0]['y']), (annual - annual.mean()).to_numpy()))
    page.slider(key='history_years').set_value((2020, 2021)).run(timeout=45)
    check('Year range restricts daily year choices', page.selectbox(key='daily_year').options == ['2020', '2021'])
    spec = json.loads(page.get('plotly_chart')[0].proto.spec)
    check('Year range updates annual bars', len(plotted_array(spec['data'][0]['y'])) == 2)
    page.selectbox(key='daily_year').set_value(2020).run(timeout=45)
    daily_spec = json.loads(page.get('plotly_chart')[2].proto.spec)
    check('Leap year daily plot has 366 days', len(plotted_array(daily_spec['data'][0]['y'])) == 366)
    page.selectbox(key='history_country').set_value('Argentina').run(timeout=45)
    check('An uncached source location has a useful load state', not page.exception and len(page.info) == 1 and page.button(key='load_history'))
    check('Uncached location does not reuse another location rainfall', len(page.get('plotly_chart')) == 0)
    page.button(key='load_history').click().run(timeout=45)
    check('Weather outage is shown without fabricated results', not page.exception and len(page.error) == 1 and len(page.get('plotly_chart')) == 0)
    page.selectbox(key='history_country').set_value('India').run(timeout=45)
    check('Saved location recovers after an unavailable archive request', not page.exception and not page.error and len(page.get('plotly_chart')) == 3)
    page.radio(key='workspace_page').set_value('Check flood risk').run(timeout=45)
    check('Risk form remains available after historical analysis', not page.exception and len(page.slider) == 11)

report = {'passed': len(checks), 'failed': 0, 'checks': checks}
(ROOT / '.run/history-verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
