"""Simple user-facing historical river forecast and preserved rainfall history."""
from html import escape
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_ui import chart,layout,metric_cards,section_title
from river_forecast import (ROOT,EXPERIMENT,STATIONS,load_bundle,load_hourly,forecast_at,stage_category)
from historical_page import render_history

PALETTE={'Normal':'#21867e','Warning':'#be861f','Danger':'#c74f4b'}

@st.cache_resource
def forecast_bundle():return load_bundle()

@st.cache_data
def observations(station):return load_hourly(station)

@st.cache_data
def replay_dates():
    return pd.read_csv(EXPERIMENT/'test.csv',parse_dates=['time','label_end'])

def forecast_chart(hourly,result,show_actual=False):
    issue=pd.Timestamp(result['issue_time']);end=pd.Timestamp(result['horizon_end'])
    past=hourly.loc[(hourly.index>=issue-pd.Timedelta(hours=48))&(hourly.index<=issue)]
    values=[past.level.min(),past.level.max(),result['lower'],result['upper'],result['warning'],result['danger']]
    future=hourly.loc[(hourly.index>issue)&(hourly.index<=end)]
    if show_actual and not future.empty:values += [future.observed_max.min(),future.observed_max.max()]
    low=min(v for v in values if np.isfinite(v))-.45;high=max(v for v in values if np.isfinite(v))+.55
    fig=go.Figure()
    fig.add_hrect(y0=low,y1=result['warning'],fillcolor='#eaf5ef',line_width=0,layer='below')
    fig.add_hrect(y0=result['warning'],y1=result['danger'],fillcolor='#fbf1d9',line_width=0,layer='below')
    fig.add_hrect(y0=result['danger'],y1=high,fillcolor='#fbe9e6',line_width=0,layer='below')
    fig.add_trace(go.Scatter(x=past.index,y=past.level,name='Observed river level',mode='lines',
        line=dict(color='#217e91',width=3),connectgaps=False,
        hovertemplate='%{x|%d %b, %H:%M} IST<br>%{y:.3f} m<extra>Observed</extra>'))
    fig.add_trace(go.Scatter(x=[issue,end,end,issue,issue],
        y=[result['lower'],result['lower'],result['upper'],result['upper'],result['lower']],
        mode='lines',fill='toself',fillcolor='rgba(33,134,126,.14)',line=dict(width=0),name='Forecast range',hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[issue,end],y=[result['predicted_peak']]*2,name='Forecast 24-hour peak',
        mode='lines',line=dict(color='#173f55',width=3,dash='dash'),
        hovertemplate='Expected highest level during the next 24h<br>%{y:.3f} m<extra>Forecast peak</extra>'))
    if show_actual:
        fig.add_trace(go.Scatter(x=future.index,y=future.observed_max,mode='lines',name='Recorded future readings',
            line=dict(color='#8c6699',width=2),connectgaps=False,
            hovertemplate='%{x|%d %b, %H:%M} IST<br>%{y:.3f} m<extra>Recorded outcome</extra>'))
    for level,label,color in [(result['warning'],'Warning',PALETTE['Warning']),
                              (result['danger'],'Danger',PALETTE['Danger'])]:
        fig.add_hline(y=level,line_color=color,line_width=1.3,line_dash='dot',
                      annotation_text=f'{label} · {level:.3f} m',annotation_position='top left',
                      annotation_font_color=color)
    # A shape avoids Plotly date-annotation arithmetic inconsistencies.
    fig.add_shape(type='line',x0=issue,x1=issue,y0=0,y1=1,xref='x',yref='paper',
                  line=dict(color='#80999e',width=1,dash='dot'))
    layout(fig,425)
    fig.update_layout(yaxis=dict(title='River level (m)',range=[low,high],gridcolor='#dfe9e7'),
        xaxis=dict(title=None,type='date',range=[issue-pd.Timedelta(hours=48),end],tickformat='%d %b\n%H:%M'),
        legend=dict(orientation='h',y=-.2,x=0),margin=dict(l=45,r=16,t=12,b=80))
    return fig

def render_forecast():
    st.html('<div class="page-heading"><span class="eyebrow">MAHARASHTRA · NEXT 24 HOURS</span>'
            '<h1>Follow the river<span>.</span></h1>'
            '<p>Choose a gauge and replay a forecast from archived observations.</p></div>')
    if not (EXPERIMENT/'forecast_bundle.joblib').exists():
        st.info('River forecasts are being prepared. Rainfall history is available in the other tab.')
        return
    frame=replay_dates()
    with st.container(border=True,key='forecast_controls'):
        col1,col2,col3=st.columns([1.3,.7,1.5],gap='medium')
        with col1:
            station=st.selectbox('River gauge',list(STATIONS),format_func=lambda k:
                                 f"{STATIONS[k]['name']} · {STATIONS[k]['river']}",key='forecast_station')
        available=frame.loc[frame.station.eq(station)]
        years=sorted(available.time.dt.year.unique().tolist())
        with col2:
            year=st.selectbox('Year',years,index=len(years)-1,key='forecast_year_'+station)
        choices=available.loc[available.time.dt.year.eq(year),'time'].sort_values(ascending=False).tolist()
        with col3:
            date_key=f'forecast_date_{station}_{year}'
            issue=st.selectbox('Forecast date',choices,index=0,
                format_func=lambda t:t.strftime('%d %B %Y · 08:00 IST'),key=date_key)
        # Example dates are selected from past observations only, never outcomes.
        year_rows=available.loc[available.time.dt.year.eq(year)]
        examples={'Latest available':year_rows.time.max(),
                  'Rising river':year_rows.loc[year_rows.change_6h.idxmax(),'time'],
                  'Highest observed level':year_rows.loc[year_rows.current_level.idxmax(),'time']}
        def choose_example(value):
            st.session_state[f'forecast_date_{station}_{year}']=value
        buttons=st.columns(3)
        for col,(label,value) in zip(buttons,examples.items()):
            with col:st.button(label,key=f'example_{label}_{station}_{year}',
                               on_click=choose_example,args=(value,),width='stretch')
    info=STATIONS[station]
    st.html(f'<div class="forecast-context"><span class="demo-pill">Historical forecast</span>'
            f'<span>{escape(info["district"])} district · forecast window ends '
            f'{(issue+pd.Timedelta(hours=24)).strftime("%d %b, 08:00 IST")}</span></div>')
    hourly=observations(station)
    try:
        result=forecast_at(station,issue,bundle=forecast_bundle(),hourly=hourly)
    except (ValueError,KeyError,OSError) as error:
        st.error('The readings needed for this forecast are incomplete. Choose another available date.')
        return
    status=result['status'];color=PALETTE[status]
    gap=result['danger']-result['predicted_peak']
    trend=result['change_6h']
    label='Below warning' if status=='Normal' else status+' level'
    message=(f"{gap:.2f} m below danger" if gap>=0 else f"{abs(gap):.2f} m above danger")
    with st.container(key='forecast_answer'):
        st.html(f'<div class="forecast-verdict" style="--status:{color}"><div>'
                f'<span class="eyebrow">EXPECTED HIGHEST LEVEL · NEXT 24 HOURS</span>'
                f'<div class="forecast-number">{result["predicted_peak"]:.2f}<small> m</small></div>'
                f'<p>{message}</p></div><div class="stage-pill">{label}</div></div>')
    metric_cards([
        ('Level at forecast time',f'{result["current_level"]:.2f} m','Observed at 08:00 IST'),
        ('Change over 6 hours',f'{trend:+.2f} m','Rising' if trend>.005 else 'Falling' if trend<-.005 else 'Steady'),
        ('Forecast range',f'{result["lower"]:.2f}–{result["upper"]:.2f} m','Estimated from earlier forecast errors'),
        ('Official danger level',f'{result["danger"]:.3f} m',f'Warning begins at {result["warning"]:.3f} m'),
    ])
    lower_status=stage_category(result['lower'],station);upper_status=stage_category(result['upper'],station)
    if lower_status!=upper_status:
        st.warning('The forecast range crosses an official threshold. The status is uncertain.')
    with st.container(border=True,key='river_timeline'):
        heading,control=st.columns([2,1],vertical_alignment='center')
        with heading:section_title('','Past readings. Future peak.')
        with control:compare=st.toggle('Compare with recorded outcome',value=False,key='show_actual')
        chart(forecast_chart(hourly,result,compare),'river_forecast_chart',
              alt='Past river levels, forecast peak range and official warning/danger lines')
        st.caption('The dashed line is the predicted maximum during the window, not a predicted hour-by-hour path. '
                   'Levels use the gauge reference height, not river depth.')
        if compare:
            actual=float(available.loc[available.time.eq(issue),'target_peak'].iloc[0])
            actual_status=stage_category(actual,station)
            st.html(f'<div class="outcome-row"><span>Recorded peak <b>{actual:.3f} m</b></span>'
                    f'<span>Forecast error <b>{abs(actual-result["predicted_peak"]):.3f} m</b></span>'
                    f'<span>Recorded status <b>{"Below warning" if actual_status=="Normal" else actual_status}</b></span></div>')
    download={**result,'station_name':info['name'],'district':info['district'],'river':info['river']}
    st.download_button('Download this forecast',pd.DataFrame([download]).to_csv(index=False),
        file_name=f'{station}-forecast-{issue:%Y-%m-%d}.csv',mime='text/csv',icon=':material/download:')
    st.caption('Source: Central Water Commission / National Water Data Portal. '
               'Archived forecast demonstration · not a live emergency alert.')

def render_app():
    st.set_page_config(page_title='RiskRadar · Maharashtra River Forecast',page_icon='◉',
                       layout='wide',initial_sidebar_state='collapsed')
    st.html(f'<style>{(ROOT/"assets/dashboard.css").read_text(encoding="utf-8")}</style>')
    with st.container(key='app_navbar'):
        brand,nav=st.columns([1,1.3],gap='large',vertical_alignment='center')
        with brand:st.html('<div class="app-brand"><span class="app-mark"></span><strong>RiskRadar</strong>'
                          '<span class="brand-description">Maharashtra rivers</span></div>')
        with nav,st.container(key='main_navigation'):
            if st.session_state.get('workspace_page') not in ['River forecast','Rainfall history']:
                st.session_state.workspace_page='River forecast'
            page=st.radio('Navigate',['River forecast','Rainfall history'],horizontal=True,
                          key='workspace_page',label_visibility='collapsed',width='stretch')
    if page=='River forecast':render_forecast()
    else:render_history()
    st.html('<div class="page-footer"><span>RiskRadar</span><span>River observations. A clearer next day.</span></div>')
