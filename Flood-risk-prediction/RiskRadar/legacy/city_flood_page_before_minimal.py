"""One place, one manual form, one understandable flood-stage result."""
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from city_flood import RUN,WARNING,DANGER,load_bundle,predict_manual
from dashboard_ui import metric_cards,chart,layout,section_title


@st.cache_resource
def model():return load_bundle()


def render_prediction():
    st.html('<div class="page-heading"><span class="eyebrow">KURUNDWAD · KOLHAPUR · MAHARASHTRA</span>'
            '<h1>Check flood risk<span>.</span></h1>'
            '<p>Enter rainfall and recent river readings to assess the next 24 hours.</p></div>')
    if not (RUN/'model_bundle.joblib').exists():
        st.info('The flood-risk model is being prepared.');return
    with st.container(border=True,key='prediction_inputs'):
        st.caption('Krishna River · Kurundwad gauge · observation time 08:00 IST')
        date_col,rain_col=st.columns([1,1],gap='large')
        with date_col:
            day=st.date_input('Observation date',value=datetime.now(ZoneInfo('Asia/Kolkata')).date(),
                              min_value=pd.Timestamp('2021-01-01').date(),key='city_date')
        with rain_col:
            rain=st.number_input('Recent daily rainfall (mm, optional)',min_value=0.,max_value=2000.,
                value=None,step=.1,placeholder='Leave blank if unavailable',key='city_rain_mm')
        cols=st.columns(3,gap='large')
        with cols[0]:current=st.number_input('River level now (m)',min_value=509.455,max_value=567.16,
            value=533.5,step=.01,format='%.3f',key='city_current')
        with cols[1]:previous6=st.number_input('River level 6 hours ago (m)',min_value=509.455,max_value=567.16,
            value=533.2,step=.01,format='%.3f',key='city_previous6')
        with cols[2]:previous24=st.number_input('River level 24 hours ago (m)',min_value=509.455,max_value=567.16,
            value=532.8,step=.01,format='%.3f',key='city_previous24')
        st.caption('Use the gauge’s river elevation in metres above mean sea level. '
                   'Rainfall must be a measured daily report available before 08:00, from the preceding 30 hours. '
                   'Leave rainfall unavailable if unknown.')
        submitted=st.button('Predict flood risk',type='primary',icon=':material/bolt:',
                            key='city_predict',width='stretch')
    issue=pd.Timestamp(day)+pd.Timedelta(hours=8)
    signature=hashlib.sha256(json.dumps([str(issue),current,previous6,previous24,rain]).encode()).hexdigest()
    if submitted:
        try:st.session_state.city_result={'signature':signature,'result':predict_manual(issue,current,previous6,previous24,rain,model())}
        except ValueError as error:st.error(str(error));return
    saved=st.session_state.get('city_result')
    if saved and saved['signature']==signature:
        answer=saved['result'];color='#c65c4c' if answer['elevated_risk'] else '#238b87'
        with st.container(border=True,key='city_prediction_result'):
            st.html(f'<div class="risk-result"><span class="eyebrow">NEXT 24 HOURS</span>'
                f'<h2 style="color:{color}">{answer["risk"]} flood-stage risk</h2>'
                f'<p>{"The model flags a warning-level exceedance." if answer["elevated_risk"] else "The model does not flag a warning-level exceedance."}</p></div>')
            st.caption(f'{issue:%d %b %Y, %H:%M} → {issue+pd.Timedelta(hours=24):%d %b %Y, %H:%M} IST · '
                       f'Official warning level: {WARNING:.2f} m')
        if current>=DANGER:st.warning('The entered river level is already at or above the official danger level. This is the current observed condition.')
        elif current>=WARNING:st.warning('The entered river level is already at or above the official warning level. This is an ongoing high-water condition.')
        metric_cards([('Current river stage',answer['current_status'],'From your entered reading'),
            ('Change in 6 hours',f'{answer["change_6h_m"]:+.2f} m','Rising' if answer['change_6h_m']>0 else 'Falling or steady'),
            ('Distance to warning',f'{WARNING-current:+.2f} m','Warning threshold minus current level')],key='city_summary')
        st.download_button('Download assessment',pd.DataFrame([answer]).to_csv(index=False),
            file_name='kurundwad-flood-risk.csv',mime='text/csv',key='city_download')
    else:st.info('Enter your readings and select Predict flood risk. Changed inputs need a new prediction.')
    with st.container(border=True,key='prediction_rain_history'):
        section_title('','Your river readings')
        fig=go.Figure(go.Scatter(x=[issue-pd.Timedelta(hours=24),issue-pd.Timedelta(hours=6),issue],
            y=[previous24,previous6,current],mode='lines+markers',line=dict(color='#238b87',width=3),marker=dict(size=9),
            hovertemplate='%{x|%d %b %H:%M}<br>%{y:.3f} m<extra>Entered reading</extra>'))
        fig.add_hline(y=WARNING,line_color='#b5842b',line_dash='dash',annotation_text=f'Warning {WARNING:.2f} m')
        fig.add_hline(y=DANGER,line_color='#c65c4c',line_dash='dash',annotation_text=f'Danger {DANGER:.2f} m')
        chart(layout(fig,290).update_layout(yaxis_title='River elevation (m)',xaxis=dict(tickformat='%d %b %H:%M')),
            'city_inputs_chart',alt='Three manually entered river readings compared with official warning and danger levels')
    st.caption('Predicts river warning-level risk at this gauge; it does not predict street flooding or inundation. '
               'Lower risk does not establish safety. Manual scenario only; no live readings are fetched.')
    if issue.month not in [6,7,8,9,10]:
        st.info('The reported evaluation covers June–October. This date is outside that evaluated season.')
    with st.expander('Evaluation and model details'):
        result=json.loads((RUN/'final_results.json').read_text());protocol=json.loads((RUN/'protocol.json').read_text())
        metric=result['test'][result['selected_model']]
        st.write('Logistic Regression · trained on 2014–2020, selected on 2021–2022, evaluated on 2023–2025.')
        table=[]
        for name,label in [('logistic','Logistic Regression'),('decision_tree','Decision Tree'),
            ('random_forest','Random Forest'),('persistence','Current-level baseline'),('positive_trend','River-trend baseline')]:
            row=result['test'][name]
            table.append({'Model':label,**{key.title():f'{row[key]:.2%}' for key in ['accuracy','precision','recall','f1']}})
        st.dataframe(pd.DataFrame(table),hide_index=True,width='stretch')
        st.write(f'Test: {metric["n"]} daily predictions, {metric["positive"]} warning-level cases. '
                 'The selected model detected 13, missed 1, and raised 2 false alerts.')
        st.caption('Only one test crossing starts from below warning: ML detected it but made two false alerts '
            '(precision 33.33%, recall 100%, F1 50%). Most positive days are ongoing high water. '
            'The river-trend baseline performed better. These are retrospective results on few dependent flood days, '
            'not proof of reliable operational advance warning. Rainfall reports are sparse; missing reports are never treated as zero. '
            'The classifier score is not shown as a calibrated flood probability.')
