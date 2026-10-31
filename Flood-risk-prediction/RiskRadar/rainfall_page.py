"""One rainfall chart at a time, with simple location and period context."""
import calendar
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard_ui import chart,layout,metric_cards
from historical_data import aggregate_rainfall,default_station,read_weather,station_catalog
from rainfall_risk import historical_indicators


@st.cache_data
def saved_rainfall():
    identifier=default_station(station_catalog())
    frame,_=read_weather(identifier)
    return frame


def render_rainfall():
    st.html('<div class="page-heading"><span class="eyebrow">RAINFALL · NEAR MUMBAI</span>'
        '<h1>Explore rainfall<span>.</span></h1>'
        '<p>Compare seasons and spot unusually wet days.</p></div>')
    try:frame=saved_rainfall()
    except (ValueError,KeyError,json.JSONDecodeError,OSError):
        st.info('Rainfall history is temporarily unavailable. Please try again later.');return
    if frame is None or frame.empty:
        st.info('Rainfall history is temporarily unavailable. Please try again later.');return
    preferences=st.session_state.get('rain_preferences',{'years':(2016,2025),'view':'Yearly','daily_year':2025})
    analysis=historical_indicators(frame.set_index('date')[['rain_mm']])
    with st.container(border=True,key='rain_controls'):
        place,period=st.columns([1,1.7],gap='large',vertical_alignment='center')
        with place:
            st.html('<div class="location-label"><span class="location-dot"></span><div><strong>Near Mumbai</strong>'
                    '<small>Historical rainfall · 2000–2025</small></div></div>')
        with period:years=st.slider('Period',2000,2025,tuple(preferences['years']),key='rain_years')
    selected=frame.loc[frame.date.dt.year.between(*years)].copy()
    total=selected.rain_mm.sum(min_count=1)
    peak=selected.loc[selected.rain_mm.idxmax()] if selected.rain_mm.notna().any() else None
    threshold=analysis['heavy_threshold_mm']
    heavy=selected.rain_mm.gt(threshold)
    metric_cards([('Rainfall in this period',f'{total:,.0f} mm' if pd.notna(total) else '—',f'{years[0]}–{years[1]}'),
        ('Wettest day',f'{peak.rain_mm:,.1f} mm' if peak is not None else '—',peak.date.strftime('%d %b %Y') if peak is not None else 'No reading'),
        ('Unusually wet days',f'{int(heavy.sum()):,}',f'More than {threshold:.1f} mm in a day')],key='rain_summary')
    with st.container(border=True,key='rain_chart_panel'):
        control,year_control=st.columns([2,1],gap='large',vertical_alignment='bottom')
        with control:
            views=['Daily','Monthly','Yearly','Monsoon']
            view=st.radio('View',views,index=views.index(preferences['view']),horizontal=True,
                          key='rain_view',label_visibility='collapsed')
        with year_control:
            preferred_year=min(years[1],max(years[0],preferences['daily_year']))
            chosen=st.selectbox('Year',list(range(years[0],years[1]+1)),index=preferred_year-years[0],
                                key='rain_daily_year') if view=='Daily' else None
        st.session_state.rain_preferences={'years':years,'view':view,'daily_year':chosen or preferences['daily_year']}
        fig=go.Figure();height=345
        if view=='Daily':
            daily=selected.loc[selected.date.dt.year.eq(chosen)]
            fig.add_trace(go.Scatter(x=daily.date,y=daily.rain_mm,mode='lines',name='Rainfall',
                line=dict(color='#238b87',width=2),fill='tozeroy',fillcolor='rgba(35,139,135,.09)',
                connectgaps=False,hovertemplate='%{x|%d %b %Y}<br>%{y:.1f} mm<extra></extra>'))
            unusual=daily.loc[daily.rain_mm.gt(threshold)]
            fig.add_trace(go.Scatter(x=unusual.date,y=unusual.rain_mm,mode='markers',name='Unusually wet day',
                marker=dict(color='#c77558',size=7),hovertemplate='%{x|%d %b %Y}<br>%{y:.1f} mm<extra>Unusually wet day</extra>'))
            fig.update_layout(xaxis=dict(rangeslider=dict(visible=True,thickness=.07)),
                              legend=dict(orientation='h',y=1.18,x=0))
            title=f'Daily rainfall · {chosen}'
        elif view=='Monthly':
            monthly=aggregate_rainfall(selected,'MS').rename('rain_mm').reset_index()
            monthly['year']=monthly.date.dt.year;monthly['month']=monthly.date.dt.month
            grid=monthly.pivot(index='year',columns='month',values='rain_mm').reindex(columns=range(1,13))
            fig.add_trace(go.Heatmap(z=grid.to_numpy(),x=list(calendar.month_abbr)[1:],y=grid.index,
                colorscale=['#f0f7f5','#b7dcd2','#5ba89b','#227f79','#193f4c'],
                colorbar=dict(title='mm',thickness=8,len=.8),xgap=4,ygap=4,
                hovertemplate='%{x} %{y:.0f}<br>%{z:,.1f} mm<extra></extra>'))
            fig.update_layout(yaxis=dict(dtick=1,tickformat='d',autorange='reversed'),xaxis=dict(side='top'))
            height=max(310,min(570,len(grid)*26+90));title='A year of rain, month by month'
        else:
            series=selected.set_index('date').rain_mm
            if view=='Monsoon':
                wet=series.loc[series.index.month.isin([6,7,8,9])]
                totals=wet.groupby(wet.index.year).sum(min_count=122)
                reference=frame.loc[frame.date.dt.year.le(2015)&frame.date.dt.month.isin([6,7,8,9])]
                normal=reference.groupby(reference.date.dt.year).rain_mm.sum(min_count=122).mean()
                title='Monsoon rainfall · June to September'
            else:
                totals=aggregate_rainfall(selected,'YS');totals.index=totals.index.year
                normal=aggregate_rainfall(frame.loc[frame.date.dt.year.le(2015)],'YS').mean()
                title='Rainfall through the years'
            fig.add_trace(go.Bar(x=totals.index,y=totals.values,marker_color='#238b87',marker_cornerradius=5,
                hovertemplate='%{x:.0f}<br>%{y:,.1f} mm<extra></extra>'))
            if np.isfinite(normal):fig.add_hline(y=normal,line_dash='dot',line_color='#b48c4d',
                annotation_text='Historical average',annotation_position='top left')
            fig.update_xaxes(dtick=1,tickformat='d')
        layout(fig,height)
        if view!='Monthly':fig.update_yaxes(title='Rainfall (mm)')
        chart(fig,'rainfall_explorer',toolbar=False,alt=title)
        st.caption('Hover for rainfall amounts. Drag across the chart to explore a smaller period.' if view=='Daily'
                   else 'Darker squares mean more rainfall. Hover to explore.' if view=='Monthly'
                   else 'June–September rainfall. The dotted line shows the 2000–2015 average.' if view=='Monsoon'
                   else 'Yearly rainfall. The dotted line shows the 2000–2015 average.')
    monthly=analysis['monthly']
    monthly=monthly.loc[monthly.index.year>=years[0]].loc[lambda rows:rows.index.year<=years[1]]
    unusual=int(monthly.z_score.ge(2).sum())
    st.html(f'<div class="rain-insight"><span class="insight-mark">↗</span><span><strong>{unusual} unusually wet months</strong>'
        ' in this period, compared with the historical seasonal pattern.</span></div>')
    footer,download=st.columns([2,1],vertical_alignment='center')
    with footer:st.caption('Historical rainfall estimates · Near Mumbai · Open-Meteo / ERA5')
    with download:st.download_button('Save rainfall data',selected.to_csv(index=False),
        file_name=f'mumbai-rainfall-{years[0]}-{years[1]}.csv',mime='text/csv',key='rain_download',width='stretch')
