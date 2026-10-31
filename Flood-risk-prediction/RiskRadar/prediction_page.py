"""Visible ML prediction, baseline comparison and rainfall what-if simulation."""
from datetime import datetime
from zoneinfo import ZoneInfo
from html import escape
import json
import hashlib

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_ui import chart,layout,metric_cards,section_title
from rainfall_risk import (RUN,LABELS,weather,ml_bundle,predict,category,historical_indicators)

COLORS={'Low':'#238b87','Medium':'#b5842b','High':'#c65c4c'}


@st.cache_data
def saved_weather():return weather()


@st.cache_resource
def bundles():return ml_bundle(),joblib.load(RUN/'forecast_bundle.joblib')


def render_prediction():
    st.html('<div class="page-heading"><span class="eyebrow">ML PREDICTION · RAINFALL BASED INDICATOR</span>'
            '<h1>Predict rainfall risk<span>.</span></h1>'
            '<p>Use the past 14 days of rainfall to estimate the next three-day rainfall category.</p></div>')
    if not (RUN/'forecast_bundle.joblib').exists():
        st.info('The prediction models are being prepared. Historical analysis remains available.')
        return
    data=saved_weather();ml,reference=bundles()
    with st.container(border=True,key='prediction_inputs'):
        col1,col2=st.columns([1,1.6],gap='large')
        with col1:
            st.selectbox('Weather location',['Near Mumbai · saved ERA5 point'],key='risk_location')
        with col2:
            mode=st.radio('Input source',['Saved observations','What-if simulation'],horizontal=True,key='risk_input_mode')
        if mode=='Saved observations':
            issue=st.date_input('Last observed day',value=pd.Timestamp('2025-08-01').date(),
                min_value=pd.Timestamp('2020-01-01').date(),max_value=pd.Timestamp('2025-12-28').date(),key='risk_saved_date')
            series=data.reindex(pd.date_range(pd.Timestamp(issue)-pd.Timedelta(days=13),pd.Timestamp(issue))).rain_mm
        else:
            today=datetime.now(ZoneInfo('Asia/Calcutta')).date()
            issue=st.date_input('Scenario date',value=today,min_value=pd.Timestamp('2016-01-01').date(),
                                max_value=pd.Timestamp('2035-12-31').date(),key='risk_scenario_date')
            sample=data.loc['2025-07-19':'2025-08-01','rain_mm'].to_numpy()
            editor=pd.DataFrame({'Day':[f'{13-i} days before' if i<13 else 'Scenario day' for i in range(14)],
                                 'Rainfall (mm)':sample})
            st.caption('Example rainfall is prefilled. Replace it with your 14 daily values; these are scenario inputs, not live readings.')
            edited=st.data_editor(editor,hide_index=True,num_rows='fixed',disabled=['Day'],key='risk_rain_editor',
                column_config={'Rainfall (mm)':st.column_config.NumberColumn(min_value=0.,max_value=2000.,step=.1,format='%.1f')},
                width='stretch')
            series=pd.to_numeric(edited['Rainfall (mm)'],errors='coerce')
        values=series.to_numpy(float)
        valid=len(values)==14 and np.isfinite(values).all() and (values>=0).all()
        if not valid:st.warning('Enter all 14 rainfall values as nonnegative numbers.')
        signature=hashlib.sha256(json.dumps([mode,str(issue),values.tolist()]).encode()).hexdigest()
        submitted=st.button('Predict risk',type='primary',icon=':material/bolt:',key='run_rainfall_prediction',
                            width='stretch',disabled=not valid)
    if valid:
        metric_cards([('Rainfall on last day',f'{values[-1]:.1f} mm','Input observation'),
                      ('Previous 3 days',f'{values[-3:].sum():.1f} mm','Cumulative rainfall'),
                      ('Previous 7 days',f'{values[-7:].sum():.1f} mm','Cumulative rainfall'),
                      ('Previous 14 days',f'{values.sum():.1f} mm','Cumulative rainfall')])
    if submitted:
        kwargs={'rainfall':values} if mode=='What-if simulation' else {}
        try:
            primary=predict(issue,bundle=ml,**kwargs);baseline=predict(issue,bundle=reference,**kwargs)
            st.session_state.rainfall_prediction={'signature':signature,'ml':primary,'baseline':baseline}
        except (ValueError,KeyError) as error:
            st.error(str(error));return
    result=st.session_state.get('rainfall_prediction')
    if result and result['signature']==signature:
        primary=result['ml'];baseline=result['baseline']
        st.html(f'<div class="forecast-context"><span class="demo-pill">{escape(mode)}</span>'
                f'<span>Prediction period · {primary["period_start"]} to {primary["period_end"]}</span></div>')
        left,right=st.columns(2,gap='large')
        for col,label,answer in [(left,'ML indicator',primary),(right,'Recent-rainfall baseline',baseline)]:
            with col,st.container(border=True):
                st.html(f'<div class="risk-result"><span class="eyebrow">{label}</span>'
                        f'<h2 style="color:{COLORS[answer["category"]]}">{answer["category"]}</h2>'
                        '<p>Expected category of the next three-day rainfall total</p></div>')
        threshold=primary['thresholds_mm']
        st.html('<div class="rain-risk-bands">'+''.join(
            f'<div style="--band:{COLORS[label]}"><strong>{label}</strong><span>{detail}</span></div>'
            for label,detail in [('Low',f'Below {threshold["medium"]:.1f} mm'),
                                 ('Medium',f'{threshold["medium"]:.2f} to < {threshold["high"]:.2f} mm'),
                                 ('High',f'{threshold["high"]:.2f} mm or more')])+'</div>')
        st.caption('Bands use the training-period 90th and 95th percentiles of nonzero three-day rainfall totals. '
                   'The baseline scored better on validation. Both outputs are experimental.')
        with st.expander('Explore contributing rainfall'):
            scenarios=[{'Scenario':'Your entered rainfall','ML category':primary['category'],
                        'Previous 3-day rainfall (mm)':round(float(values[-3:].sum()),1)}]
            for days in [3,7]:
                changed=values.copy();changed[-days:]*=.5
                alternative=predict(issue,rainfall=changed,bundle=ml)
                scenarios.append({'Scenario':f'Half as much rain during the last {days} days',
                    'ML category':alternative['category'],'Previous 3-day rainfall (mm)':round(float(changed[-3:].sum()),1)})
            st.dataframe(pd.DataFrame(scenarios),hide_index=True,width='stretch')
            st.caption('These sensitivity examples recalculate all rainfall features. They show model responses, not cause and effect.')
        if mode=='Saved observations' and st.toggle('Show the recorded next three days',key='risk_show_outcome'):
            future=data.loc[pd.Timestamp(issue)+pd.Timedelta(days=1):pd.Timestamp(issue)+pd.Timedelta(days=3),'rain_mm']
            if len(future)==3 and future.notna().all():
                total=float(future.sum());actual=LABELS[int(category([total],threshold)[0])]
                display=st.success if actual==primary['category'] else st.warning
                display(f'Recorded rainfall: {total:.1f} mm · actual category: {actual} · ML estimate: {primary["category"]}')
        export=pd.DataFrame([{**{key:value for key,value in answer.items() if key!='thresholds_mm'},'estimate':name}
                             for name,answer in [('ML',primary),('Baseline',baseline)]])
        st.download_button('Download prediction',export.to_csv(index=False),file_name='rainfall-risk-prediction.csv',mime='text/csv')
    else:
        st.info('Choose your inputs and click Predict risk. Changing an input requires a new prediction.')
    if valid:
        with st.container(border=True,key='prediction_rain_history'):
            section_title('','Rainfall used for this prediction')
            dates=pd.date_range(pd.Timestamp(issue)-pd.Timedelta(days=13),pd.Timestamp(issue))
            fig=go.Figure(go.Bar(x=dates,y=values,marker_color='#238b87',marker_cornerradius=4,
                                hovertemplate='%{x|%d %b %Y}<br>%{y:.1f} mm<extra>Input</extra>'))
            chart(layout(fig,285).update_layout(yaxis_title='Daily rainfall (mm)',xaxis=dict(tickformat='%d %b')),
                  'prediction_input_chart',alt='Fourteen past daily rainfall values used as prediction inputs')
    st.caption('Rainfall-based risk indicator, not a probability or validated prediction of flooding. '
               'Prediction coverage is the existing near-Mumbai ERA5 point only; daily record dates are UTC. '
               'No live weather is fetched.')


def render_anomalies(frame,filtered):
    # The selected period is always compared with the full fixed 2000–2015
    # reference, not a baseline silently refitted to the user's selected years.
    indexed=frame.set_index('date')[['rain_mm']]
    indicators=historical_indicators(indexed)
    selected=filtered.set_index('date')[['rain_mm']]
    flags=indicators['heavy_days'].reindex(selected.index)
    monthly=indicators['monthly'].loc[lambda data:(data.index.year>=int(filtered.date.dt.year.min())) &
                                     (data.index.year<=int(filtered.date.dt.year.max()))]
    st.html('<div class="section-title"><h3>Heavy rainfall &amp; unusual months</h3></div>')
    metric_cards([('Selected rainfall',f'{selected.rain_mm.sum(min_count=1):,.1f} mm','Sum of available daily readings'),
                  ('Heavy rainfall days',str(int(flags.fillna(False).sum())),
                   f'Above {indicators["heavy_threshold_mm"]:.1f} mm · baseline wet-day 95th percentile'),
                  ('Unusually wet months',str(int(monthly.z_score.ge(2).sum())),'At least 2 baseline standard deviations above normal')],key='anomaly_stats')
    fig=go.Figure(go.Bar(x=monthly.index,y=monthly.anomaly_mm,
        marker_color=['#d46655' if value>=0 else '#7cafaa' for value in monthly.anomaly_mm],
        hovertemplate='%{x|%b %Y}<br>Difference from normal %{y:+.1f} mm<extra></extra>'))
    fig.add_hline(y=0,line_color='#829797',line_width=1)
    chart(layout(fig,295).update_layout(yaxis_title='Monthly difference from baseline (mm)'),
          'rainfall_monthly_anomalies',alt='Monthly rainfall departures from the fixed 2000 to 2015 calendar-month average')
    st.caption('Baseline: 2000–2015 for this location. Heavy-rain and anomaly flags indicate unusual rainfall, not confirmed floods. '
               'Incomplete months remain unknown.')
