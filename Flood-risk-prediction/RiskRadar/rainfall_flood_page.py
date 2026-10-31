"""A simple PS-aligned dashboard for rainfall and recorded flood risk analysis."""
from html import escape
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_ui import chart,layout,metric_cards,section_title
from historical_page import render_history
from city_flood_page import render_prediction
from rainfall_flood import (ROOT,DATA,RELEASE,filter_events,filter_inventory,load_events,
    load_inventory,load_bundle,daily_rainfall,rain_groups,severity_estimates,state_tokens,gauge_risk_summary)

COLORS={'Flood':'#238b87','Severe Flood':'#d46655'}


@st.cache_data
def events():return load_events()


@st.cache_data
def inventory():return load_inventory()


@st.cache_resource
def models():return load_bundle()


def counts_chart(frame,by_month=False):
    part=frame.assign(period=frame['Start Date'].dt.month if by_month else frame['Start Date'].dt.year)
    grouped=part.groupby(['period','Flood Type']).size().unstack(fill_value=0)
    if by_month:grouped=grouped.reindex(range(1,13),fill_value=0)
    fig=go.Figure()
    for label,color in COLORS.items():
        values=grouped[label] if label in grouped else np.zeros(len(grouped))
        fig.add_trace(go.Bar(x=grouped.index,y=values,name=label,marker_color=color,
            hovertemplate='%{x}<br>%{y:,} recorded events<extra>'+label+'</extra>'))
    layout(fig,315)
    fig.update_layout(barmode='stack',yaxis_title='Recorded flood events',legend=dict(orientation='h',y=1.18),
                      margin=dict(l=35,r=10,t=45,b=30),xaxis=dict(dtick=1,tickformat='d'))
    if by_month:
        fig.update_xaxes(tickvals=list(range(1,13)),ticktext=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    return fig


def rainfall_severity_chart(frame,window):
    labels=list(COLORS)
    values=[frame.loc[frame['Flood Type'].eq(label),window].median() for label in labels]
    counts=[int(frame['Flood Type'].eq(label).sum()) for label in labels]
    fig=go.Figure(go.Bar(x=labels,y=values,marker_color=list(COLORS.values()),marker_cornerradius=6,
        text=[f'{value:.1f} mm' if np.isfinite(value) else 'No events' for value in values],
        textposition='outside',cliponaxis=False,customdata=counts,
        hovertemplate='%{x}<br>Typical rainfall %{y:.1f} mm<br>%{customdata:,} recorded events<extra></extra>'))
    return layout(fig,315).update_layout(yaxis_title='Typical rainfall before event (mm)',showlegend=False,
                                         margin=dict(l=45,r=10,t=30,b=35))


def rainfall_group_chart(frame,window):
    group=rain_groups(frame,window)
    fig=go.Figure(go.Bar(x=group.rain_group.astype(str),y=group.severe_share*100,
        customdata=group[['events','severe']].to_numpy(),marker_color='#238b87',marker_cornerradius=5,
        text=[f'{value:.0%}' if pd.notna(value) else 'No events' for value in group.severe_share],
        textposition='outside',cliponaxis=False,
        hovertemplate='%{x} mm<br>%{customdata[1]} severe / %{customdata[0]} recorded floods<br>%{y:.1f}% severe<extra></extra>'))
    return layout(fig,300).update_layout(yaxis=dict(title='Severe share among recorded floods',range=[0,108],ticksuffix='%'),
        xaxis_title='Rainfall before event (mm)',margin=dict(l=45,r=10,t=15,b=45))


def render_analysis():
    st.html('<div class="page-heading"><span class="eyebrow">HISTORICAL WEATHER · COVERED INDIAN GAUGES</span>'
            '<h1>Rainfall &amp; flood risk<span>.</span></h1>'
            '<p>Explore the rain before recorded floods and how severe those floods became.</p></div>')
    data=events()
    with st.container(border=True,key='flood_filters'):
        first,second=st.columns([1,2],gap='large')
        with first:state=st.selectbox('State',['All covered states']+sorted(data.State.unique()),key='analysis_state')
        with second:years=st.slider('Event years',int(data['Start Date'].dt.year.min()),2019,(2000,2019),key='analysis_years')
    selected=filter_events(data,state,years)
    if selected.empty:
        st.info('No recorded events match these filters. Try a wider period or another state. Missing reports do not mean no floods.')
        return
    severe=int(selected.target.sum())
    metric_cards([('Recorded floods',f'{len(selected):,}',f'{selected.GaugeID.nunique()} covered gauges'),
                  ('Severe floods',f'{severe:,}',f'{severe/len(selected):.1%} of recorded floods'),
                  ('Median 3-day rainfall',f'{selected.T3d.median():.1f} mm','Before the recorded event began'),
                  ('Median flood duration',f'{selected["Event Duration (days)"].median():.0f} days','From the event record')])
    render_risk_map(selected,data.loc[data.split.eq('train')])
    left,right=st.columns(2,gap='large')
    with left,st.container(border=True,key='flood_frequency_panel'):
        section_title('','When were floods recorded?')
        by=st.radio('Group events',['By year','By month'],horizontal=True,key='event_grouping',label_visibility='collapsed')
        chart(counts_chart(selected,by=='By month'),'flood_event_counts',toolbar=False,
              alt='Recorded Flood and Severe Flood event counts by year or month')
    with right,st.container(border=True,key='rain_severity_panel'):
        section_title('','Typical rainfall before floods')
        window=st.radio('Rainfall window',['3 days','7 days','10 days'],horizontal=True,key='rain_window')
        window_key={'3 days':'T3d','7 days':'T7d','10 days':'T10d'}[window]
        chart(rainfall_severity_chart(selected,window_key),'rainfall_severity_boxes',toolbar=False,
              alt='Median antecedent rainfall amounts for Flood and Severe Flood events')
        st.caption('Typical means the median. Individual events can differ substantially.')
    with st.container(border=True,key='rain_risk_relationship'):
        section_title('','Does more rainfall always mean a severe flood?')
        chart(rainfall_group_chart(selected,window_key),'rainfall_severity_share',toolbar=False,
              alt='Severe share among recorded floods grouped by antecedent rainfall amount')
        st.caption('The percentages describe recorded floods only. They are not the probability of a flood on any day. '
                   'Rainfall groups use training-period boundaries; location and catchment conditions also differ.')
    render_event(selected)
    export=['EventID','Station','State','Start Date','End Date','Flood Type','T1d','T3d','T7d','T10d',
            'Peak Flood Level (m)','Warning Level','Danger Level','Event Duration (days)']
    st.download_button('Download matched rainfall and flood records',selected[export].to_csv(index=False),
        file_name='matched-rainfall-flood-events.csv',mime='text/csv',icon=':material/download:')
    st.caption('INDOFLOODS · matched recorded flood events and pre-event catchment rainfall · 1965–2019 rainfall coverage. '
               'Record counts reflect the available archive, not complete flood frequency.')


def render_risk_map(selected,reference):
    summary,limits=gauge_risk_summary(selected,reference)
    with st.container(border=True,key='regional_risk_map'):
        section_title('','Regional flood-severity map','Select a marker to explore its recorded gauge events below.')
        fig=go.Figure()
        colors={'Low':'#238b87','Medium':'#d59a38','High':'#d46655','Limited records':'#b2bfc4'}
        for band,color in colors.items():
            rows=summary.loc[summary.band.eq(band)]
            if rows.empty:continue
            custom=np.column_stack([rows.GaugeID,rows.Station,rows.events,rows.severe_share*100,rows.rain_3d_median])
            fig.add_trace(go.Scattermap(lat=rows.Latitude,lon=rows.Longitude,mode='markers',name=band,
                marker=dict(size=12,color=color,opacity=.9),customdata=custom,
                hovertemplate='%{customdata[1]}<br>%{customdata[2]} recorded floods<br>'
                '%{customdata[3]:.1f}% were severe<br>Median 3-day rainfall %{customdata[4]:.1f} mm<extra>'+band+'</extra>'))
        fig.update_layout(map=dict(style='carto-positron',center=dict(lat=float(summary.Latitude.mean()),
            lon=float(summary.Longitude.mean())),zoom=4),height=370,margin=dict(l=0,r=0,t=8,b=0),
            clickmode='event+select',
            legend=dict(orientation='h',y=1.05,x=0),paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Segoe UI',size=12,color='#173c4a'))
        selection=st.plotly_chart(fig,key='historical_risk_map',on_select='rerun',selection_mode='points',
            width='stretch',theme=None,config={'displayModeBar':False,'displaylogo':False},
            alt='Gauge locations coloured by relative historical severe-flood share, with sparse records shown in gray')
        points=selection.get('selection',{}).get('points',[]) if selection else []
        signature=json.dumps(points,sort_keys=True,default=str)
        if points and signature!=st.session_state.get('_consumed_map_selection'):
            fields=points[-1].get('customdata',[])
            if fields and fields[0] in set(selected.GaugeID):st.session_state['event_gauge']=fields[0]
        st.session_state['_consumed_map_selection']=signature
        st.caption(f'Low / Medium / High compare the observed severe share: up to {limits["low_max"]:.0%}, '
                   f'up to {limits["medium_max"]:.0%}, and above. Boundaries are training-gauge terciles. '
                   'Gray means fewer than 10 selected events. These are historical gauge patterns, not district-wide or future flood probabilities.')


def render_event(selected):
    with st.container(border=True,key='recorded_event_panel'):
        section_title('','Explore a recorded flood','See the rainfall leading up to one event.')
        col1,col2=st.columns([1,1.5],gap='large')
        gauges=selected[['GaugeID','Station','State']].drop_duplicates().sort_values(['Station','GaugeID']).set_index('GaugeID')
        with col1:
            gauge=st.selectbox('Gauge',gauges.index.tolist(),key='event_gauge',
                format_func=lambda key:f'{gauges.loc[key,"Station"]} · {gauges.loc[key,"State"]}')
        choices=selected.loc[selected.GaugeID.eq(gauge)].sort_values('Start Date',ascending=False).set_index('EventID')
        with col2:
            identifier=st.selectbox('Recorded event',choices.index.tolist(),key='selected_event',
                format_func=lambda key:choices.loc[key,'Start Date'].strftime('%d %B %Y'))
        row=choices.loc[identifier].copy();row['EventID']=identifier
        label=str(row['Flood Type']);color=COLORS[label]
        st.html(f'<div class="recorded-verdict"><div><span class="eyebrow">RECORDED OUTCOME</span>'
                f'<h2 style="color:{color}">{escape(label)}</h2>'
                f'<p>{escape(str(row["Station"]))} · {row["Start Date"]:%d %b %Y} to {row["End Date"]:%d %b %Y}</p>'
                '</div><div class="rain-window-values">'
                f'<span><b>{row["T3d"]:.1f} mm</b>previous 3 days</span>'
                f'<span><b>{row["T7d"]:.1f} mm</b>previous 7 days</span>'
                f'<span><b>{row["T10d"]:.1f} mm</b>previous 10 days</span></div></div>')
        rain=daily_rainfall(row)
        fig=go.Figure(go.Bar(x=rain.date,y=rain.rain_mm,marker_color='#238b87',marker_cornerradius=4,
            name='Daily rainfall',hovertemplate='%{x|%d %b %Y}<br>%{y:.2f} mm<extra>Before flood onset</extra>'))
        layout(fig,285)
        fig.update_layout(yaxis_title='Daily rainfall (mm)',xaxis=dict(tickformat='%d %b',dtick=86400000),
                          margin=dict(l=40,r=10,t=10,b=30))
        chart(fig,'pre_event_daily_rain',toolbar=True,
              alt='Ten days of daily catchment rainfall ending the day before the selected flood began')
        st.caption('Rainfall is measured before event onset. “Flood” and “Severe Flood” are the published event labels; '
                   'neither means “no flood”.')
        compare=st.toggle('Compare experimental severity estimates',value=False,key='compare_severity')
        if compare:
            if row['split']!='test':
                st.info('Choose an event from 2015–2019 for a held-out model comparison.')
            else:
                estimates=severity_estimates(row,bundle=models())
                comparison=pd.DataFrame([{'Information used':item['name'],
                    'Estimated severity':item['estimated_severity'],
                    'Matches recorded severity':item['estimated_severity']==row['Flood Type']} for item in estimates])
                st.dataframe(comparison,hide_index=True,width='stretch')
                st.caption('Severity among known floods, not flood/no-flood prediction. These experimental models have '
                           'limited reliability and are shown for historical comparison, not live alerts.')


def render_records():
    st.html('<div class="page-heading"><span class="eyebrow">INDIA FLOOD INVENTORY</span>'
            '<h1>Recorded floods<span>.</span></h1>'
            '<p>Browse reported events and where they occurred.</p></div>')
    frame=inventory()
    states=sorted({name for value in frame.State for name in state_tokens(value)})
    with st.container(border=True,key='inventory_filters'):
        c1,c2=st.columns([1,2],gap='large')
        with c1:state=st.selectbox('State',['All covered states']+states,key='inventory_state')
        minimum=int(frame['Start Date'].dt.year.min());maximum=int(frame['Start Date'].dt.year.max())
        with c2:years=st.slider('Reported event years',minimum,maximum,(2000,maximum),key='inventory_years')
    selected=filter_inventory(frame,state,years)
    metric_cards([('Reported events',f'{len(selected):,}','Unique source event records'),
                  ('Years with reports',str(selected['Start Date'].dt.year.nunique()),'Within the selected period'),
                  ('District details available',f'{selected.Districts.notna().sum():,}','Events with reported district names')])
    if selected.empty:
        st.info('No reports match these filters. This does not establish that no floods occurred.')
        return
    counts=selected.groupby(selected['Start Date'].dt.year).size()
    fig=go.Figure(go.Bar(x=counts.index,y=counts.values,marker_color='#238b87',marker_cornerradius=4,
        hovertemplate='%{x:.0f}<br>%{y:,} reported events<extra></extra>'))
    chart(layout(fig,310).update_layout(xaxis=dict(dtick=1,tickformat='d'),yaxis_title='Reported events'),
          'reported_event_chart',toolbar=False,alt='Number of unique flood inventory records by reported start year')
    columns=['UEI','Start Date','End Date','State','Districts','Main Cause','Duration(Days)']
    view=selected[columns].sort_values('Start Date',ascending=False).copy()
    for name in ['Start Date','End Date']:view[name]=view[name].dt.strftime('%d %b %Y')
    view=view.rename(columns={'UEI':'Event reference','Start Date':'Start','End Date':'End','Districts':'Reported districts',
                              'Main Cause':'Reported cause','Duration(Days)':'Duration (days)'})
    st.dataframe(view,hide_index=True,width='stretch')
    st.download_button('Download flood inventory',selected[columns].to_csv(index=False),
        file_name='reported-flood-events.csv',mime='text/csv',icon=':material/download:')
    st.caption('IFI-Impacts, 1967–2023. Reports can cover several states; each source event is counted once in this view. '
               'Missing reports are not verified non-flood observations. This inventory is separate from the matched gauge-event analysis.')


def render_app():
    st.set_page_config(page_title='RiskRadar · Rainfall and Flood Risk Analysis',page_icon='◉',
                       layout='wide',initial_sidebar_state='collapsed')
    st.html(f'<style>{(ROOT/"assets/dashboard.css").read_text(encoding="utf-8")}</style>')
    with st.container(key='app_navbar'):
        brand,nav=st.columns([.9,1.8],gap='large',vertical_alignment='center')
        with brand:st.html('<div class="app-brand"><span class="app-mark"></span><strong>RiskRadar</strong>'
                          '<span class="brand-description">Rainfall &amp; floods</span></div>')
        with nav,st.container(key='main_navigation'):
            pages=['Rainfall & floods','Prediction','Rainfall history','Flood records']
            if st.session_state.get('workspace_page') not in pages:
                st.session_state.workspace_page='Prediction'
            page=st.radio('Navigate',pages,horizontal=True,key='workspace_page',label_visibility='collapsed',width='stretch')
    if not (DATA/'manifest.json').exists():
        st.info('The historical analysis is being prepared. Please try again shortly.')
        return
    if page=='Rainfall & floods':render_analysis()
    elif page=='Prediction':render_prediction()
    elif page=='Rainfall history':render_history(country_scope='India')
    else:render_records()
    st.html('<div class="page-footer"><span>RiskRadar</span><span>Rainfall and flood risk analysis using historical weather data</span></div>')
