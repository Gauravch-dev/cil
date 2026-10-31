"""Exercise user interactions in the PS-aligned Streamlit application."""
from pathlib import Path
import sys
import json
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from streamlit.testing.v1 import AppTest
from rainfall_flood import RELEASE,load_events


def main():
    checks=[]
    def check(name,condition):
        checks.append({'check':name,'passed':bool(condition)})
        assert condition,name
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=40).run()
    check('Main app opens manual Prediction with preserved navigation',not app.exception and
          app.radio(key='workspace_page').options==['Rainfall & floods','Prediction','Rainfall history','Flood records'] and
          app.radio(key='workspace_page').value=='Prediction')
    app.radio(key='workspace_page').set_value('Rainfall & floods').run()
    check('Core matched-event charts map and export load',len(app.get('plotly_chart'))==5 and len(app.get('download_button'))==1)
    for state in ['Maharashtra','Kerala','Rajasthan','All covered states']:
        app.selectbox(key='analysis_state').set_value(state).run()
        check(f'{state}: filters render safely',not app.exception and not app.error)
    app.radio(key='event_grouping').set_value('By month').run()
    check('Seasonal event counts render',not app.exception and len(app.get('plotly_chart'))==5)
    for window in ['7 days','10 days','3 days']:
        app.radio(key='rain_window').set_value(window).run()
        check(f'Rainfall window {window} updates',not app.exception and len(app.get('plotly_chart'))==5)
    app.slider(key='analysis_years').set_value((2015,2019)).run()
    app.toggle(key='compare_severity').set_value(True).run()
    check('Held-out severity comparison is available',not app.exception and len(app.dataframe)==1)
    app.slider(key='analysis_years').set_value((1965,1965)).run()
    check('Empty or sparse selections do not crash',not app.exception and not app.error)
    app.radio(key='workspace_page').set_value('Flood records').run()
    check('Flood inventory loads with chart and table',not app.exception and len(app.dataframe)==1 and
          len(app.get('plotly_chart'))==1)
    app.selectbox(key='inventory_state').set_value('Maharashtra').run()
    check('Inventory state filter works',not app.exception and not app.error and len(app.dataframe)==1)
    app.slider(key='inventory_years').set_value((2020,2023)).run()
    check('Inventory year filter and export work',not app.exception and not app.error and
          len(app.get('download_button'))==1)
    app.radio(key='workspace_page').set_value('Rainfall history').run()
    check('Rainfall history includes monsoon and anomaly views',not app.exception and len(app.get('plotly_chart'))==5)
    check('Rainfall history is scoped to India',app.selectbox(key='history_country').options==['India'])
    app.radio(key='annual_view').set_value('Above / below average').run()
    app.slider(key='history_years').set_value((2020,2024)).run()
    app.selectbox(key='daily_year').set_value(2023).run()
    check('Rainfall anomaly period and daily controls work',not app.exception and len(app.get('plotly_chart'))==5)
    app.radio(key='workspace_page').set_value('Prediction').run()
    check('Prediction is directly accessible with a visible run button',not app.exception and app.button(key='city_predict'))
    app.button(key='city_predict').click().run()
    check('Manual flood-stage ML prediction and export display',not app.exception and
          'city_result' in app.session_state and len(app.get('download_button'))==1)
    app.number_input(key='city_rain_mm').set_value(35.).run()
    app.button(key='city_predict').click().run()
    check('Manual rainfall is accepted with recent river levels',not app.exception and
          app.session_state['city_result']['result']['rain_mm']==35.)
    app.number_input(key='city_current').set_value(538.).run()
    check('Changed input clears visible stale prediction',not app.exception and len(app.get('download_button'))==0)
    app.button(key='city_predict').click().run()
    check('Current danger-stage condition is identified separately',not app.exception and
          app.session_state['city_result']['result']['current_status']=='Danger' and len(app.warning)==1)
    app.radio(key='workspace_page').set_value('Rainfall & floods').run()
    check('Return to PS-aligned analysis works',not app.exception and not app.error)
    return checks


if __name__=='__main__':
    try:
        checks=main();result={'passed':len(checks),'failed':0,'checks':checks}
    except Exception:result={'failed':1,'error':traceback.format_exc()}
    (RELEASE/'dashboard_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if result['failed']:raise SystemExit(1)
