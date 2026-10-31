"""A bounded historical-rainfall page with explicit sources and no invented labels."""
import calendar
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from dashboard_ui import COLORS, chart, layout, metric_cards, section_title
from historical_data import (ROOT, aggregate_rainfall, default_station, fetch_weather,
                              read_weather, station_catalog)


def render_history(country_scope=None):
    st.html('<div class="page-heading"><h1>Rainfall history<span>.</span></h1>'
            '<p>Choose a location. Discover its wetter years and rainy seasons.</p></div>')
    if not (ROOT / 'data/processed/stations.csv').exists():
        st.info('Rainfall history is temporarily unavailable. Please try again later.')
        return
    catalog = station_catalog()
    if country_scope is not None:
        catalog=catalog.loc[catalog.country.eq(country_scope)].copy()
    preferred = default_station(catalog)
    countries = sorted(catalog.country.unique())
    with st.container(border=True, key='history_filters'):
        left, right, period = st.columns([1, 1.65, 1.8], gap='large')
        with left:
            country = st.selectbox('Country', countries, index=countries.index('India') if 'India' in countries else 0,
                                   key='history_country')
        choices = catalog.loc[catalog.country.eq(country)].set_index('station_id')
        identifiers = choices.index.tolist()
        def location_label(code):
            latitude, longitude = choices.loc[code, ['latitude', 'longitude']]
            coordinates = f'{abs(latitude):.2f}° {"N" if latitude >= 0 else "S"}, {abs(longitude):.2f}° {"E" if longitude >= 0 else "W"}'
            return f'Near Mumbai · {coordinates}' if code == preferred else coordinates
        with right:
            selected = st.selectbox('Location', identifiers,
                index=identifiers.index(preferred) if preferred in identifiers else 0,
                format_func=location_label,
                key='history_station')
        with period:
            years = st.slider('Years to explore', 2000, 2025, (2016, 2025), key='history_years')
    reload_needed = False
    try:
        frame, document = read_weather(selected)
    except (ValueError, KeyError, json.JSONDecodeError):
        st.error('We couldn’t read the rainfall for this location. Load it again to retry.')
        frame, document = None, None
        reload_needed = True
    if frame is None:
        st.info('Load rainfall history for this location to begin.')
        if st.button('Load rainfall history', type='primary', key='load_history'):
            try:
                with st.spinner('Loading rainfall history…'):
                    fetch_weather(selected, overwrite=reload_needed)
                st.rerun()
            except (requests.RequestException, ValueError, KeyError):
                st.error('Rainfall couldn’t be loaded. Please try again later or choose another location.')
        return
    filtered = frame.loc[frame.date.dt.year.between(*years)].copy()
    annual = aggregate_rainfall(filtered, 'YS')
    monthly = aggregate_rainfall(filtered, 'MS')
    valid = int(filtered.rain_mm.notna().sum())
    metric_cards([
        ('Average yearly rainfall', f'{annual.mean():,.0f}' if annual.notna().any() else '—', 'mm per year'),
        ('Wettest day', f'{filtered.rain_mm.max():,.1f}' if valid else '—', 'mm of rainfall'),
        ('Wettest month', f'{monthly.max():,.0f}' if monthly.notna().any() else '—', 'mm in one month'),
        ('Days with rain', f'{filtered.rain_mm.gt(0).sum():,}', 'in the selected years'),
    ])
    if valid < len(filtered):
        st.info('Some rainfall readings are missing. Incomplete monthly and yearly totals are not shown.')
    yearly_tab, monthly_tab, daily_tab, monsoon_tab = st.tabs(['Year by year', 'Monthly pattern', 'Daily rainfall','Monsoon seasons'])
    with yearly_tab, st.container(border=True, key='annual_history'):
        section_title('', 'Which years were wetter?')
        view = st.radio('Chart view', ['Rainfall total', 'Above / below average'], horizontal=True, key='annual_view')
        mean = annual.mean()
        values = annual if view == 'Rainfall total' else annual - mean
        fig = go.Figure(go.Bar(x=annual.index.year, y=values,
            marker_color=[COLORS['teal'] if v >= 0 else '#aabcc4' for v in values],
            marker_cornerradius=5, hovertemplate='%{x:.0f}<br>%{y:,.1f} mm<extra></extra>'))
        if np.isfinite(mean) and view == 'Rainfall total':
            fig.add_hline(y=mean, line_dash='dot', line_color='#d59a38',
                          annotation_text='Period average', annotation_position='top left')
        elif view != 'Rainfall total':
            fig.add_hline(y=0, line_color='#8b9da5')
        chart(layout(fig, 330).update_layout(xaxis=dict(dtick=1, tickformat='d'),
              yaxis_title='Rainfall (mm)' if view == 'Rainfall total' else 'Difference from period average (mm)'),
              'annual_rainfall', toolbar=False, alt='Annual rainfall totals or differences from the selected period average')
    with monthly_tab, st.container(border=True, key='monthly_history'):
        section_title('', 'When does the rain arrive?', 'Darker squares mean more rain. Hover for the amount.')
        grid = monthly.rename('rain_mm').reset_index()
        grid['year'], grid['month'] = grid.date.dt.year, grid.date.dt.month
        grid = grid.pivot(index='year', columns='month', values='rain_mm').reindex(columns=range(1, 13))
        fig = go.Figure(go.Heatmap(z=grid.to_numpy(), x=list(calendar.month_abbr)[1:], y=grid.index,
            colorscale=['#edf6f3', '#9bcfc1', '#238b87', '#173c4a'],
            colorbar=dict(title='mm', thickness=10),
            hovertemplate='%{x} %{y:.0f}<br>%{z:,.1f} mm<extra></extra>'))
        chart(layout(fig, max(300, min(620, len(grid) * 27 + 90))).update_layout(
            yaxis=dict(dtick=1, tickformat='d', autorange='reversed'), xaxis=dict(side='top')),
            'monthly_rainfall', toolbar=False, alt='Monthly rainfall heatmap, with years as rows and months as columns')
    with daily_tab, st.container(border=True, key='daily_history'):
        section_title('', 'Find the rainiest days', 'Drag to zoom in on a date range.')
        selected_year = st.selectbox('Year', list(range(years[0], years[1] + 1)),
                                     index=years[1] - years[0], key='daily_year')
        daily = filtered.loc[filtered.date.dt.year.eq(selected_year)]
        fig = go.Figure(go.Scatter(x=daily.date, y=daily.rain_mm, mode='lines',
            line=dict(color=COLORS['teal'], width=1.8), fill='tozeroy', fillcolor='rgba(35,139,135,.13)',
            connectgaps=False, hovertemplate='%{x|%d %b %Y}<br>%{y:.1f} mm<extra></extra>'))
        chart(layout(fig, 360).update_layout(yaxis_title='Rainfall (mm)',
            xaxis=dict(rangeslider=dict(visible=True, thickness=.12),
                       rangeselector=dict(buttons=[dict(count=1, label='1 month', step='month', stepmode='backward'),
                                                  dict(count=3, label='3 months', step='month', stepmode='backward'),
                                                  dict(step='all', label='Full year')]))),
            'daily_rainfall', toolbar=False, alt=f'Daily historical rainfall for {selected_year}')
    with monsoon_tab,st.container(border=True,key='monsoon_history'):
        section_title('','Compare monsoon seasons','June–September rainfall in each selected year.')
        series=filtered.set_index('date').rain_mm
        summer=series.loc[series.index.month.isin([6,7,8,9])]
        totals=summer.groupby(summer.index.year).sum(min_count=122)
        original=frame.set_index('date').rain_mm
        reference=original.loc[(original.index.year<=2015)&original.index.month.isin([6,7,8,9])]
        normal=reference.groupby(reference.index.year).sum(min_count=122).mean()
        fig=go.Figure(go.Bar(x=totals.index,y=totals.values,marker_color=COLORS['teal'],marker_cornerradius=5,
                            hovertemplate='%{x:.0f}<br>Monsoon rainfall %{y:,.1f} mm<extra></extra>'))
        if np.isfinite(normal):fig.add_hline(y=normal,line_dash='dot',line_color='#d59a38',annotation_text='2000–2015 average')
        chart(layout(fig,320).update_layout(xaxis=dict(dtick=1,tickformat='d'),yaxis_title='Monsoon rainfall (mm)'),
              'monsoon_rainfall',toolbar=False,alt='June to September rainfall totals by year compared with the fixed historical average')
        st.caption('All 122 daily readings are required for a monsoon total; incomplete seasons are not filled with zero.')
    st.download_button('Download rainfall', filtered.to_csv(index=False),
        file_name=f'{selected}_rainfall_{years[0]}_{years[1]}.csv', mime='text/csv', icon=':material/download:')
    from prediction_page import render_anomalies
    render_anomalies(frame,filtered)
    st.caption('Historical rainfall estimates · [Open-Meteo / ERA5](https://open-meteo.com/en/docs/historical-weather-api) · '
               '[Locations](https://zenodo.org/records/7158028)')
