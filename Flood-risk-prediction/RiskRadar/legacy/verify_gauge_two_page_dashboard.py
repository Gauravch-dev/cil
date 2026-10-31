"""Validate both public pages, source-backed chart values and manual checks."""
from pathlib import Path
import sys
import json
import base64
import traceback
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import requests
from streamlit.testing.v1 import AppTest
from rainfall_page import saved_rainfall
from historical_data import aggregate_rainfall
from city_flood import predict_manual

ROOT=Path(__file__).resolve().parents[1]


def array(value):
    if isinstance(value,dict) and 'bdata' in value:
        return np.frombuffer(base64.b64decode(value['bdata']),dtype=value['dtype'])
    return np.array(value)


def main():
    checks=[]
    def check(name,value):
        assert value,name
        checks.append({'check':name,'passed':True})
    def clear(page):return not page.exception and not page.error
    def plot(page):return json.loads(page.get('plotly_chart')[0].proto.spec)
    def no_technical_panels(page):
        text=' '.join(str(element.value) for kind in ['markdown','html','caption'] for element in page.get(kind))
        return not any(word in text.lower() for word in ['logistic','precision','recall','roc-auc','validation','model details']) and len(page.expander)==0
    with patch('requests.get',side_effect=requests.ConnectionError('offline check')):
        page=AppTest.from_file(str(ROOT/'app.py'),default_timeout=40).run()
        check('Exactly two pages, Rainfall and Flood',clear(page) and page.radio(key='workspace_page').options==['Rainfall','Flood'])
        check('Rainfall opens with one chart and one download',page.radio(key='workspace_page').value=='Rainfall' and len(page.get('plotly_chart'))==1 and len(page.get('download_button'))==1)
        check('No model or evaluation panels on Rainfall',no_technical_panels(page))
        source=saved_rainfall();chosen=source.loc[source.date.dt.year.between(2016,2025)]
        np.testing.assert_allclose(array(plot(page)['data'][0]['y']),aggregate_rainfall(chosen,'YS').to_numpy())
        check('Yearly chart matches actual saved rainfall',True)
        page.radio(key='rain_view').set_value('Monthly').run()
        check('Monthly view is one interactive heatmap',clear(page) and len(page.get('plotly_chart'))==1 and plot(page)['data'][0]['type']=='heatmap')
        page.radio(key='rain_view').set_value('Daily').run()
        page.selectbox(key='rain_daily_year').set_value(2024).run()
        daily=source.loc[source.date.dt.year.eq(2024)]
        np.testing.assert_allclose(array(plot(page)['data'][0]['y']),daily.rain_mm.to_numpy())
        check('Daily view preserves all 366 actual leap-year readings',clear(page) and len(array(plot(page)['data'][0]['y']))==366)
        check('Unusually wet days are highlighted in the daily chart',len(plot(page)['data'])==2)
        page.radio(key='rain_view').set_value('Monsoon').run()
        expected=chosen.loc[chosen.date.dt.month.isin([6,7,8,9])].groupby(chosen.loc[chosen.date.dt.month.isin([6,7,8,9])].date.dt.year).rain_mm.sum(min_count=122)
        np.testing.assert_allclose(array(plot(page)['data'][0]['y']),expected.to_numpy())
        check('Monsoon view uses actual June–September totals',clear(page))
        page.slider(key='rain_years').set_value((2020,2022)).run()
        check('Period control changes the chart without extra panels',clear(page) and len(array(plot(page)['data'][0]['y']))==3 and len(page.get('plotly_chart'))==1)
        page.radio(key='workspace_page').set_value('Flood').run()
        check('Flood opens a simple manual form and reading chart',clear(page) and page.button(key='city_predict') and len(page.get('plotly_chart'))==1)
        check('No model names, metrics or evaluation panels on Flood',no_technical_panels(page))
        page.button(key='city_predict').click().run()
        check('Manual river readings produce a result',clear(page) and 'city_result' in page.session_state and len(page.get('download_button'))==1)
        row=page.session_state['city_result']['result']
        expected=predict_manual(pd.Timestamp(page.date_input(key='city_date').value)+pd.Timedelta(hours=8),533.5,533.2,532.8)
        check('The redesigned result matches existing frozen inference',row==expected)
        page.number_input(key='city_rain_mm').set_value(35.).run()
        check('Changed inputs hide the previous result and export',clear(page) and len(page.get('download_button'))==0)
        page.button(key='city_predict').click().run()
        check('Optional measured rainfall is included',clear(page) and page.session_state['city_result']['result']['rain_mm']==35.)
        page.number_input(key='city_current').set_value(538.).run()
        page.button(key='city_predict').click().run()
        check('Already dangerous entered levels are clearly identified',clear(page) and len(page.warning)==1 and page.session_state['city_result']['result']['current_status']=='Danger')
        page.radio(key='workspace_page').set_value('Rainfall').run()
        check('Switching back restores the chosen rainfall view',clear(page) and page.radio(key='rain_view').value=='Monsoon')
        page.radio(key='workspace_page').set_value('Flood').run()
        check('Flood readings and result survive page navigation',clear(page) and page.number_input(key='city_current').value==538. and len(page.get('download_button'))==1)
        check('No API connection is needed for either page',True)
    return checks


if __name__=='__main__':
    try:
        checks=main();result={'passed':len(checks),'failed':0,'checks':checks}
    except Exception:result={'failed':1,'error':traceback.format_exc()}
    (ROOT/'evaluation/two_page_dashboard_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if result['failed']:raise SystemExit(1)
