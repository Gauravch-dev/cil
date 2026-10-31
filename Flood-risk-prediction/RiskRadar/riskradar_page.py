"""Original RiskRadar rainfall/land input workflow, with a minimal public UI."""
import hashlib
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard_ui import chart,layout
from riskradar_predictor import DEFAULTS,EXAMPLES,category_choices,predict


def apply_example():
    name=st.session_state.get('radar_example')
    if name in EXAMPLES:
        values=dict(EXAMPLES[name]);st.session_state.radar_preferences=values
        for key,value in values.items():st.session_state['radar_'+key]=value


def mark_custom():
    st.session_state.radar_example=None


def number(label,key,values,minimum=None,maximum=None,step=1.,help=None):
    default=None if values[key] is None or 'radar_'+key in st.session_state else float(values[key])
    return st.number_input(label,min_value=minimum,max_value=maximum,value=default,
        step=float(step),key='radar_'+key,help=help,on_change=mark_custom)


def category(label,key,values,choices):
    initial=None if values[key] is None or 'radar_'+key in st.session_state else int(values[key])
    return st.selectbox(label,list(range(len(choices[key]))),index=initial,
        format_func=lambda code:choices[key][code],key='radar_'+key,on_change=mark_custom)


def result_gauge(score,level):
    colors={'LOW':'#237f79','MEDIUM':'#b18d57','HIGH':'#bd7054'}
    fig=go.Figure(go.Indicator(mode='gauge',value=score*100,
        gauge=dict(axis=dict(range=[0,100],tickvals=[20,55,85],ticktext=['Low','Medium','High'],
                   tickfont=dict(size=11,color='#83928b'),ticks=''),
            bar=dict(color=colors[level],thickness=.33),borderwidth=0,
            steps=[dict(range=[0,40],color='#e6f1ec'),dict(range=[40,70],color='#f5efdf'),dict(range=[70,100],color='#f4e4dd')],
            threshold=dict(value=score*100,line=dict(color=colors[level],width=3),thickness=.7))))
    return layout(fig,210).update_layout(margin=dict(l=32,r=32,t=12,b=10))


def render_flood():
    st.html('<div class="page-heading"><span class="eyebrow">FLOOD · RAINFALL &amp; AREA CONDITIONS</span>'
        '<h1>Check flood risk<span>.</span></h1><p>Set the rainfall, climate and land conditions for your area.</p></div>')
    try:choices=category_choices()
    except (OSError,ValueError):st.info('Risk checks are temporarily unavailable.');return
    previous=st.session_state.get('radar_preferences',dict(DEFAULTS))
    selected_example=st.session_state.get('radar_example')
    if selected_example in EXAMPLES and previous!=EXAMPLES[selected_example]:
        st.session_state.radar_example=None
    left,right=st.columns([1.08,1],gap='large')
    with left,st.container(border=True,key='flood_form'):
        st.pills('Try an example',list(EXAMPLES),default=None,key='radar_example',on_change=apply_example)
        values={}
        annual,wettest=st.columns(2,gap='medium')
        with annual:values['annual_precipitation']=number('Annual rain (mm)','annual_precipitation',previous,0.,step=50.)
        with wettest:values['precipitation_of_wettest_month']=number('Wettest month (mm)','precipitation_of_wettest_month',previous,0.,step=10.)
        values['precipitation_seasonality']=number('Rainfall variation (%)','precipitation_seasonality',previous,0.,step=1.,
            help='Variation between months, measured as coefficient of variation. Higher values mean stronger seasonal variation.')
        with st.expander('Land & weather details'):
            a,b=st.columns(2,gap='medium')
            with a:
                values['annual_mean_temperature']=number('Mean temperature (°C)','annual_mean_temperature',previous,-90.,60.)
                values['curve_number_amcii']=number('Runoff potential (0–100)','curve_number_amcii',previous,0.,100.,
                    help='Curve number: a higher value means more rain tends to run off rather than soak into the ground.')
                values['drainage_density']=number('Drainage density (km/km²)','drainage_density',previous,0.,step=.1)
                values['basin_relief']=number('Elevation difference (m)','basin_relief',previous,0.,step=50.)
            with b:
                values['temperature_seasonality']=number('Temperature variation','temperature_seasonality',previous,0.,step=10.,
                    help='Use the recorded seasonality index: temperature standard deviation multiplied by 100.')
                values['drainage_texture']=number('Drainage texture','drainage_texture',previous,0.,step=.5)
                values['ruggedness_number']=number('Terrain ruggedness','ruggedness_number',previous,0.,step=.1)
                values['infiltration_number']=number('Infiltration number','infiltration_number',previous,0.,step=.5)
            values['climate_type']=category('Climate','climate_type',previous,choices)
            c,d=st.columns(2,gap='medium')
            with c:values['landcover_type']=category('Land cover','landcover_type',previous,choices)
            with d:values['soil_type']=category('Soil class','soil_type',previous,choices)
        submitted=st.button('Check flood risk',type='primary',icon=':material/arrow_forward:',key='radar_predict',width='stretch')
    st.session_state.radar_preferences=values
    signature=hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()
    error=None
    if submitted:
        try:st.session_state.radar_result={'signature':signature,'inputs':values.copy(),'result':predict(values)}
        except ValueError as exc:error=str(exc)
    saved=st.session_state.get('radar_result')
    active=saved and saved['signature']==signature and error is None
    with right,st.container(border=True,key='flood_result'):
        if error:st.error(error)
        if active:
            answer=saved['result'];level=answer['risk_level'];color={'LOW':'#237f79','MEDIUM':'#b18d57','HIGH':'#bd7054'}[level]
            st.html(f'<div class="assessment-answer" style="--risk:{color}"><span class="result-label">YOUR AREA CONDITIONS</span>'
                f'<h2>{level.title()} risk</h2><p>Based on the rainfall and land conditions you entered.</p></div>')
            chart(result_gauge(answer['flood_risk_score'],level),'radar_gauge',toolbar=False,
                alt=f'{level.title()} risk on the original RiskRadar assessment scale')
            st.html(f'<div class="reading-summary"><div><small>Annual rainfall</small><strong>{values["annual_precipitation"]:,.0f} mm</strong></div>'
                f'<div><small>Wettest month</small><strong>{values["precipitation_of_wettest_month"]:,.0f} mm</strong></div></div>')
            export={**values,'risk_category':level.title()}
            for key in choices:export[key]=choices[key][int(values[key])]
            st.download_button('Save result',pd.DataFrame([export]).to_csv(index=False),file_name='riskradar-assessment.csv',
                mime='text/csv',key='radar_download')
        else:
            st.html('<div class="empty-assessment"><div class="water-icon">≈</div><strong>Your result, at a glance</strong>'
                '<p>Enter your area conditions or try an example, then select Check flood risk.</p></div>')
            st.html('<div class="risk-key"><span><i class="dot low"></i>Low</span>'
                '<span><i class="dot medium"></i>Medium</span><span><i class="dot high"></i>High</span></div>')
    st.caption('Long-term flood susceptibility from annual area conditions · Scenario estimate, not a live flood alert.')
