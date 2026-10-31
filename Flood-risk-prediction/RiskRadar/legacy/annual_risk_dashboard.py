"""Rainfall analysis and flood-risk dashboard; supplied API/model stay unchanged."""
from datetime import datetime
import json
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

from dashboard_ui import (chart, gauge, history_chart, metric_cards,
                          section_title, what_if_chart, risk_scale)
from historical_page import render_history

ROOT = Path(__file__).resolve().parent
API_URL = os.environ.get('RISKRADAR_API_URL', 'http://127.0.0.1:8000').rstrip('/')
DEFAULTS = dict(annual_precipitation=1200, precipitation_of_wettest_month=300,
    precipitation_seasonality=50, drainage_density=2.5, drainage_texture=10.,
    basin_relief=500, annual_mean_temperature=20, temperature_seasonality=500,
    curve_number_amcii=70., ruggedness_number=.5, infiltration_number=5.,
    climate_type=2, landcover_type=1, soil_type=3)
PRESETS = {'Default': DEFAULTS,
    'Drier landscape': {**DEFAULTS, 'annual_precipitation': 400,
        'precipitation_of_wettest_month': 120, 'precipitation_seasonality': 25,
        'drainage_density': 1., 'curve_number_amcii': 50.},
    'Wetter landscape': {**DEFAULTS, 'annual_precipitation': 2600,
        'precipitation_of_wettest_month': 800, 'precipitation_seasonality': 85,
        'drainage_density': 6., 'drainage_texture': 30., 'basin_relief': 2000,
        'annual_mean_temperature': 27, 'curve_number_amcii': 90.,
        'ruggedness_number': 2., 'infiltration_number': 10.,
        'climate_type': 1, 'landcover_type': 0, 'soil_type': 9}}

st.set_page_config(page_title='RiskRadar · Rainfall & Flood Risk', page_icon='◉',
                   layout='wide', initial_sidebar_state='collapsed')
st.html(f'<style>{(ROOT / "assets/dashboard.css").read_text(encoding="utf-8")}</style>')


@st.cache_resource
def category_names():
    encoders = joblib.load(ROOT / 'encoders.pkl')
    return {name: ['Unspecified' if str(value) == 'nan' else str(value) for value in encoder.classes_]
            for name, encoder in encoders.items()}


@st.cache_data(ttl=15, show_spinner=False)
def service_health(url):
    try:
        response = requests.get(url + '/health', timeout=3)
        response.raise_for_status()
        body = response.json()
        return body.get('status') == 'running' and body.get('features') == 13
    except (requests.RequestException, ValueError):
        return False


def predict(payload):
    response = requests.post(API_URL + '/predict', json=payload, timeout=15)
    response.raise_for_status()
    result = response.json()
    score = result.get('flood_risk_score')
    if (not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1
            or result.get('risk_level') not in {'LOW', 'MEDIUM', 'HIGH'}):
        raise ValueError('The prediction service returned an invalid result.')
    return result


@st.cache_data(ttl=600, show_spinner=False)
def explore_prediction(payload_json, url):
    response = requests.post(url + '/predict', json=json.loads(payload_json), timeout=15)
    response.raise_for_status()
    result = response.json()
    score = result.get('flood_risk_score')
    if (not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1
            or result.get('risk_level') not in {'LOW', 'MEDIUM', 'HIGH'}):
        raise ValueError('Invalid comparison score')
    return result


@st.cache_data(ttl=600, show_spinner=False)
def rainfall_scenarios(payload_json, url):
    """Real one-variable model responses, not feature importance or a forecast."""
    payload = json.loads(payload_json)
    wettest = payload['precipitation_of_wettest_month']
    lower, upper = wettest, min(3000, 12 * wettest) if wettest else 0
    values = sorted(set([int(value) for value in np.linspace(lower, upper, 8)] +
                        [payload['annual_precipitation']]))
    rows = []
    for value in values:
        response = requests.post(url + '/predict', json={**payload, 'annual_precipitation': value}, timeout=15)
        response.raise_for_status()
        result = response.json()
        score = result.get('flood_risk_score')
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('The comparison service returned an invalid score.')
        rows.append({'annual_mm': value, 'score_pct': score * 100})
    return pd.DataFrame(rows)




def apply_preset():
    if st.session_state.example is None:
        return
    for key, value in PRESETS[st.session_state.example].items():
        st.session_state['input_' + key] = value


def initialise_inputs():
    submitted = st.session_state.get('assessment', {}).get('inputs', DEFAULTS)
    for key, value in submitted.items():
        if 'input_' + key not in st.session_state:
            st.session_state['input_' + key] = value


def slider(label, name, start, end, step=1, help=None):
    return st.slider(label, start, end, step=step, key='input_' + name, help=help)


def assessment_form(ready):
    initialise_inputs()
    with st.container(border=True, key='input_panel'):
        section_title('', 'Set the conditions')
        st.pills('Try an example', list(PRESETS), key='example', default=None,
                 on_change=apply_preset, selection_mode='single')
        with st.form('risk_inputs', border=False):
            annual = slider('Rainfall in a year · mm', 'annual_precipitation', 0, 3000, 50,
                            help='Total annual rainfall for the area. Use measured values, or try an example above.')
            wettest = slider('Rainfall in the wettest month · mm', 'precipitation_of_wettest_month', 0, 1000, 10)
            season = slider('Variation through the year · %', 'precipitation_seasonality', 0, 100,
                            help='Rainfall seasonality: a lower value means more evenly spread rainfall; a higher value means stronger seasonal variation.')
            with st.expander('Land & weather details'):
                st.markdown('**Terrain & water runoff**')
                left, right = st.columns(2, gap='medium')
                with left:
                    density = slider('Drainage density · km/km²', 'drainage_density', 0., 10., .1,
                                     help='Total stream length divided by the basin area.')
                    relief = slider('Elevation difference · m', 'basin_relief', 0, 5000, 50,
                                    help='Basin relief: the difference between the highest and lowest elevations.')
                    curve = slider('Runoff potential · curve number', 'curve_number_amcii', 0., 100., 1.,
                                   help='A higher curve number generally means more rainfall runs off rather than soaking into the ground.')
                with right:
                    texture = slider('Drainage texture', 'drainage_texture', 0., 50., .5,
                                     help='A basin measure describing the spacing and frequency of drainage channels.')
                    rugged = slider('Terrain ruggedness', 'ruggedness_number', 0., 5., .1,
                                    help='A terrain index combining basin relief and drainage density.')
                    infiltration = slider('Infiltration number', 'infiltration_number', 0., 20., .5,
                                          help='A measure of drainage and how water enters the ground; enter the value recorded for your area.')
                st.markdown('**Climate & land**')
                left, right = st.columns(2, gap='medium')
                with left:
                    temperature = slider('Mean temperature · °C', 'annual_mean_temperature', -20, 40)
                with right:
                    temp_season = slider('Temperature seasonality', 'temperature_seasonality', 0, 1500, 10,
                                        help='An index of how much temperature varies through the year, rather than a temperature in degrees.')
                names = category_names()
                categories = {}
                for name, label in [('climate_type', 'Climate'), ('landcover_type', 'Land cover'), ('soil_type', 'Soil class')]:
                    choices = names[name]
                    categories[name] = st.selectbox(label, range(len(choices)),
                        format_func=lambda code, labels=choices: labels[code], key='input_' + name)
            submitted = st.form_submit_button('Check flood risk', type='primary',
                width='stretch', icon=':material/radar:', disabled=not ready)
        if submitted:
            if wettest > annual or annual > 12 * wettest:
                st.error('Annual rainfall must be between one and twelve times the wettest-month rainfall.')
                return
            payload = dict(annual_precipitation=annual, precipitation_of_wettest_month=wettest,
                precipitation_seasonality=season, drainage_density=density, drainage_texture=texture,
                basin_relief=relief, annual_mean_temperature=temperature, temperature_seasonality=temp_season,
                curve_number_amcii=curve, ruggedness_number=rugged, infiltration_number=infiltration, **categories)
            try:
                with st.spinner('Assessing your scenario…'):
                    result = predict(payload)
                entry = {'inputs': payload, 'result': result,
                         'name': st.session_state.get('example') or 'Custom conditions',
                         'time': datetime.now(ZoneInfo('Asia/Kolkata')).isoformat()}
                st.session_state.assessment = entry
                st.session_state.assessments = (st.session_state.get('assessments', []) + [entry])[-8:]
            except (requests.RequestException, ValueError) as exc:
                st.error('We couldn’t check the risk. Please try again shortly.')
                st.session_state.last_prediction_error = str(exc)
        if not ready:
            st.info('Risk checks are temporarily unavailable. You can still explore rainfall history.')


def result_panel():
    entry = st.session_state.get('assessment')
    with st.container(border=True, key='result_panel'):
        if not entry:
            st.html('''<div class="result-empty"><div class="empty-radar"><i></i><i></i><i></i><b>◉</b></div>
                <h2>Your result,<br>at a glance.</h2>
                <p>Choose your conditions and select<br><b>Check flood risk</b>.</p>
                <div class="risk-key"><span><i class="dot low"></i>Low</span><span><i class="dot medium"></i>Medium</span><span><i class="dot high"></i>High</span></div></div>''')
            return
        result = entry['result']
        level = result['risk_level']
        section_title('', 'Your result')
        st.caption('Checked at ' + datetime.fromisoformat(entry['time']).strftime('%I:%M %p'))
        st.html(f'<div class="risk-answer {level.lower()}"><span class="status-dot"></span>{level.title()} risk</div>')
        chart(gauge(result['flood_risk_score'], level), 'risk_gauge', toolbar=False,
              alt=f"Risk score {result['flood_risk_score']:.1%}, {level.lower()} risk")
        notes = {'LOW': 'These conditions fall in the lower-risk group.',
                 'MEDIUM': 'These conditions fall in the medium-risk group.',
                 'HIGH': 'These conditions fall in the higher-risk group.'}
        st.html(f'<p class="result-note">{notes[level]}</p>')
        risk_scale(result['flood_risk_score'])
        st.caption('Scenario estimate · not a live flood alert')
        saved = pd.DataFrame([{'Risk level': level.title(), 'Risk score (%)': round(result['flood_risk_score'] * 100, 2),
            'Annual rainfall (mm)': entry['inputs']['annual_precipitation'],
            'Wettest month (mm)': entry['inputs']['precipitation_of_wettest_month'], 'Checked at': entry['time']}])
        st.download_button('Save result', saved.to_csv(index=False), file_name='riskradar-result.csv',
                           mime='text/csv', icon=':material/download:', width='stretch')


def render_assessment(ready):
    st.html('<div class="page-heading"><h1>Check flood risk<span>.</span></h1><p>Adjust the conditions. See the result. Compare a different scenario.</p></div>')
    left, right = st.columns([1.15, 1], gap='large')
    with left:
        assessment_form(ready)
    with right:
        result_panel()
    entry = st.session_state.get('assessment')
    if entry:
        payload = entry['inputs']
        metric_cards([('Annual rainfall', f"{payload['annual_precipitation']:,}", 'mm / year'),
                      ('Wettest month', f"{payload['precipitation_of_wettest_month']:,}", 'mm / month'),
                      ('Runoff potential', f"{payload['curve_number_amcii']:g}", 'curve number · 0–100'),
                      ('Mean temperature', f"{payload['annual_mean_temperature']:g}°", 'Celsius')])
        with st.container(border=True, key='comparison_panel'):
            section_title('', 'What if rainfall changed?', 'Move the slider to compare with your result.')
            try:
                frame = rainfall_scenarios(json.dumps(payload, sort_keys=True), API_URL)
                controls, visual = st.columns([1, 2.2], gap='large')
                with controls:
                    lower = int(payload['precipitation_of_wettest_month'])
                    upper = int(min(3000, 12 * lower)) if lower else 0
                    if lower < upper:
                        trial_mm = st.slider('Try a different annual rainfall · mm', lower, upper,
                            value=payload['annual_precipitation'], key='explore_' + entry['time'])
                    else:
                        trial_mm = lower
                        st.caption('With a dry wettest month, annual rainfall must also be zero.')
                    trial = explore_prediction(json.dumps({**payload, 'annual_precipitation': trial_mm}, sort_keys=True), API_URL)
                    delta = (trial['flood_risk_score'] - entry['result']['flood_risk_score']) * 100
                    delta_text = 'Same as your assessment' if abs(delta) < .005 else f'{abs(delta):.1f} points {"higher" if delta > 0 else "lower"}'
                    st.html(f'<div class="explore-answer {trial["risk_level"].lower()}"><span>AT {trial_mm:,} MM / YEAR</span><strong>{trial["risk_level"].title()} risk</strong><b>{trial["flood_risk_score"]:.1%}</b><small>{delta_text}</small></div>')
                with visual:
                    chart(what_if_chart(frame, payload['annual_precipitation'], entry['result']['flood_risk_score'],
                          trial_mm, trial['flood_risk_score']), 'rainfall_comparison', toolbar=False,
                          alt='Rainfall and risk score: submitted conditions compared with the slider scenario')
            except (requests.RequestException, ValueError):
                st.info('Rainfall comparison is unavailable. Your last successful assessment is preserved.')
        recent = st.session_state.get('assessments', [])
        if len(recent) > 1:
            with st.container(border=True, key='history_panel'):
                section_title('', 'Compare your results', 'Your last eight checks. Hover for details.')
                chart(history_chart(recent), 'assessment_history', alt='Risk scores of recent submitted assessments')




with st.container(key='app_navbar'):
    brand, navigation = st.columns([1, 1.25], gap='large', vertical_alignment='center')
    with brand:
        st.html('<div class="app-brand"><span class="app-mark"></span><strong>RiskRadar</strong><span class="brand-description">Rainfall & risk</span></div>')
    with navigation, st.container(key='main_navigation'):
        if st.session_state.get('workspace_page') not in {'Rainfall history', 'Check flood risk'}:
            st.session_state.workspace_page = 'Rainfall history'
        page = st.radio('Navigate', ['Rainfall history', 'Check flood risk'],
                        key='workspace_page', horizontal=True, label_visibility='collapsed', width='stretch')

if page == 'Check flood risk':
    render_assessment(service_health(API_URL))
else:
    render_history()
st.html('<div class="page-footer"><span>RiskRadar</span><span>Explore rainfall. Understand risk.</span></div>')
