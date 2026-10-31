"""Exercise actual Streamlit controls and preserve a compact integration result."""
from pathlib import Path
import sys
import json
import traceback
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from streamlit.testing.v1 import AppTest
from river_forecast import EXPERIMENT,STATIONS


def main():
    checks=[]
    dates=pd.read_csv(EXPERIMENT/'test.csv',parse_dates=['time'])
    def check(name,condition):
        checks.append({'check':name,'passed':bool(condition)})
        assert condition,name
    app=AppTest.from_file(str(ROOT/'legacy/river_forecast_entrypoint.py'),default_timeout=35).run()
    check('Initial forecast loads without exceptions',not app.exception)
    check('Only river forecast and rainfall history navigation',
          app.radio(key='workspace_page').options==['River forecast','Rainfall history'])
    for station in STATIONS:
        app.selectbox(key='forecast_station').set_value(station).run()
        for year in [2023,2024,2025]:
            app.selectbox(key='forecast_year_'+station).set_value(year).run()
            check(f'{station} {year}: forecast, chart and export available',
                  not app.exception and len(app.get('plotly_chart'))==1 and
                  len(app.get('download_button'))==1 and not app.error)
        for label in ['Rising river','Highest observed level','Latest available']:
            app.button(key=f'example_{label}_{station}_2025').click().run()
            rows=dates.loc[dates.station.eq(station)&dates.time.dt.year.eq(2025)]
            expected=(rows.time.max() if label=='Latest available' else
                      rows.loc[rows['change_6h' if label=='Rising river' else 'current_level'].idxmax(),'time'])
            check(f'{station}: {label} preset renders a forecast',
                  not app.exception and len(app.get('plotly_chart'))==1 and not app.error and
                  app.selectbox(key=f'forecast_date_{station}_2025').value==expected)
    app.toggle(key='show_actual').set_value(True).run()
    check('Recorded-outcome comparison loads',not app.exception and any(
          'Recorded peak' in item.value for item in app.get('html')))
    app.radio(key='workspace_page').set_value('Rainfall history').run()
    check('Rainfall history loads all three chart views',not app.exception and
          len(app.get('plotly_chart'))==3 and not app.error)
    check('Yearly, monthly and daily tabs retained',
          [tab.label for tab in app.tabs]==['Year by year','Monthly pattern','Daily rainfall'])
    app.radio(key='annual_view').set_value('Above / below average').run()
    check('Rainfall anomaly view works',not app.exception and len(app.get('plotly_chart'))==3)
    app.slider(key='history_years').set_value((2020,2024)).run()
    app.selectbox(key='daily_year').set_value(2023).run()
    check('Rainfall period and daily-year controls work',not app.exception and not app.error)
    check('Rainfall export retained',len(app.get('download_button'))==1)
    app.radio(key='workspace_page').set_value('River forecast').run()
    check('Return to forecast works after history navigation',not app.exception and
          len(app.get('plotly_chart'))==1 and not app.error)
    return checks


if __name__=='__main__':
    try:
        checks=main()
        result={'passed':len(checks),'failed':0,'checks':checks}
    except Exception:
        result={'failed':1,'error':traceback.format_exc()}
    (EXPERIMENT/'dashboard_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if result['failed']:raise SystemExit(1)
