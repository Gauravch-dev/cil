"""Minimal flood check; model evaluation stays in project documents."""
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from city_flood import RUN,WARNING,DANGER,load_bundle,predict_manual
from dashboard_ui import chart,layout


@st.cache_resource
def model():return load_bundle()


def render_prediction():
    st.html('<div class="page-heading"><span class="eyebrow">FLOOD · KURUNDWAD, MAHARASHTRA</span>'
        '<h1>Check flood risk<span>.</span></h1>'
        '<p>Add your readings. Check the river’s warning risk for the next 24 hours.</p></div>')
    if not (RUN/'model_bundle.joblib').exists():st.info('The flood check is temporarily unavailable.');return
    preferences=st.session_state.get('city_preferences',{'day':datetime.now(ZoneInfo('Asia/Kolkata')).date(),
        'rain':None,'current':533.5,'previous6':533.2,'previous24':532.8})
    left,right=st.columns([1,1.12],gap='large')
    with left,st.container(border=True,key='flood_form'):
        st.html('<div class="form-kicker">Your readings</div><div class="quiet-note">Krishna River · Kurundwad gauge</div>')
        date_col,rain_col=st.columns(2,gap='medium')
        with date_col:
            day=st.date_input('Observation date',value=preferences['day'],
                min_value=pd.Timestamp('2021-01-01').date(),key='city_date',help='The daily check starts at 08:00 IST.')
        with rain_col:
            rain=st.number_input('Rainfall (mm, optional)',min_value=0.,max_value=2000.,value=preferences['rain'],step=.1,
                placeholder='If available',key='city_rain_mm',help='Use a measured daily report available before 08:00, from the preceding 30 hours. Leave blank if unknown.')
        current=st.number_input('River level now (m)',min_value=509.455,max_value=567.16,value=preferences['current'],
            step=.01,format='%.3f',key='city_current',help='Use the Kurundwad gauge elevation in metres above mean sea level, not water depth.')
        earlier6,earlier24=st.columns(2,gap='medium')
        with earlier6:previous6=st.number_input('6 hours earlier (m)',min_value=509.455,max_value=567.16,
            value=preferences['previous6'],step=.01,format='%.3f',key='city_previous6')
        with earlier24:previous24=st.number_input('24 hours earlier (m)',min_value=509.455,max_value=567.16,
            value=preferences['previous24'],step=.01,format='%.3f',key='city_previous24')
        st.html('<div class="form-note">Example river values are prefilled. Replace them with your measured readings.</div>')
        submitted=st.button('Check flood risk',type='primary',icon=':material/arrow_forward:',key='city_predict',width='stretch')
    issue=pd.Timestamp(day)+pd.Timedelta(hours=8)
    st.session_state.city_preferences={'day':day,'rain':rain,'current':current,'previous6':previous6,'previous24':previous24}
    signature=hashlib.sha256(json.dumps([str(issue),current,previous6,previous24,rain]).encode()).hexdigest()
    if submitted:
        try:st.session_state.city_result={'signature':signature,'result':predict_manual(issue,current,previous6,previous24,rain,model())}
        except ValueError as error:
            with right:st.error(str(error))
            return
    saved=st.session_state.get('city_result')
    with right,st.container(border=True,key='flood_result'):
        if saved and saved['signature']==signature:
            answer=saved['result'];color='#bd7054' if answer['elevated_risk'] else '#237f79'
            message='The river may reach its warning level in the next 24 hours.' if answer['elevated_risk'] else 'No warning-level crossing is indicated for the next 24 hours.'
            st.html(f'<div class="assessment-answer" style="--risk:{color}"><span class="result-label">NEXT 24 HOURS</span>'
                f'<h2>{answer["risk"]} risk</h2><p>{message}</p>'
                f'<div class="assessment-window">{issue:%d %b, %H:%M} → {issue+pd.Timedelta(hours=24):%d %b, %H:%M} IST</div></div>')
            change=current-previous6;direction='Rising' if change>0 else 'Falling' if change<0 else 'Steady'
            st.html(f'<div class="reading-summary"><div><small>Current stage</small><strong>{answer["current_status"]}</strong></div>'
                f'<div><small>Last 6 hours</small><strong>{direction} · {change:+.2f} m</strong></div></div>')
            if current>=DANGER:st.warning('Your entered level is already above the danger threshold.')
            elif current>=WARNING:st.warning('Your entered level is already above the warning threshold.')
        else:
            st.html('<div class="empty-assessment"><div class="water-icon">≈</div><strong>Your assessment, at a glance</strong>'
                '<p>Enter your readings and select Check flood risk to see your result here.</p></div>')
        fig=go.Figure(go.Scatter(x=[issue-pd.Timedelta(hours=24),issue-pd.Timedelta(hours=6),issue],
            y=[previous24,previous6,current],mode='lines+markers',line=dict(color='#237f79',width=2.5),marker=dict(size=7),
            hovertemplate='%{x|%d %b %H:%M}<br>%{y:.3f} m<extra>Your reading</extra>'))
        fig.add_hline(y=WARNING,line_color='#b18d57',line_dash='dot',annotation_text='Warning',annotation_position='top left')
        fig.add_hline(y=DANGER,line_color='#bd7054',line_dash='dot',annotation_text='Danger',annotation_position='top left')
        chart(layout(fig,220).update_layout(yaxis_title='River level (m)',xaxis=dict(tickformat='%H:%M'),
            margin=dict(l=5,r=5,t=10,b=5)), 'city_inputs_chart',toolbar=False,
            alt='Your recent river readings compared with warning and danger levels')
        if saved and saved['signature']==signature:
            export={key:value for key,value in saved['result'].items() if key not in ['model','model_score']}
            st.download_button('Save result',pd.DataFrame([export]).to_csv(index=False),file_name='kurundwad-risk.csv',
                               mime='text/csv',key='city_download')
    st.caption('River-stage estimate for the Kurundwad gauge · Entered values, not live readings · Lower risk does not establish safety.')
    if issue.month not in [6,7,8,9,10]:st.caption('Checks for this date fall outside the June–October assessed season.')
