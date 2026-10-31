"""Archived Risk Map; not imported or routed by the main application.

Preserved for reference. The unchanged source CSV remains at the repository root.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dashboard_ui import chart, layout, metric_cards, section_title

ROOT = Path(__file__).resolve().parents[1]

def map_chart(frame):
    plotted = frame.assign(score_pct=frame.predicted_risk * 100)
    fig = px.scatter_map(plotted, lat='latitude', lon='longitude', color='score_pct',
        color_continuous_scale=['#efc78e', '#dc9464', '#c6574c'], range_color=[70, 100],
        custom_data=['latitude', 'longitude', 'score_pct', 'records'],
        map_style='carto-positron', zoom=.65, center=dict(lat=18, lon=8))
    fig.update_traces(marker=dict(size=9, opacity=.8),
        hovertemplate='<b>Model score %{customdata[2]:.1f}%</b><br>Latitude %{customdata[0]:.4f}<br>Longitude %{customdata[1]:.4f}<br>Source records %{customdata[3]}<extra></extra>')
    layout(fig, 440)
    return fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title='Score %', orientation='h', y=-.05, len=.6, thickness=8,
                               outlinewidth=0, tickvals=[70, 80, 90, 100]), uirevision='risk-atlas')



def histogram(frame):
    counts, _ = np.histogram(frame.predicted_risk * 100, bins=[70, 80, 90, 95, 100.0001])
    fig = go.Figure(go.Bar(y=['70–80%', '80–90%', '90–95%', '95–100%'], x=counts,
        orientation='h', marker_color=['#efc78e', '#e4ad76', '#d78662', '#c6574c'],
        marker_cornerradius=5, text=counts, textposition='outside', cliponaxis=False,
        hovertemplate='%{y}<br>%{x:,} source records<extra></extra>'))
    layout(fig, 340)
    return fig.update_layout(margin=dict(l=0, r=35, t=35, b=30),
        xaxis=dict(title='Source records', showgrid=True, gridcolor=COLORS['grid']),
        yaxis=dict(autorange='reversed', showgrid=False))



@st.cache_data(show_spinner=False)
def locations(modified_ns):
    frame = pd.read_csv(ROOT / 'high_risk_locations.csv')
    frame = frame[['latitude', 'longitude', 'predicted_risk']].apply(pd.to_numeric, errors='coerce')
    valid = (frame.latitude.between(-90, 90) & frame.longitude.between(-180, 180) &
             frame.predicted_risk.between(0, 1))
    return frame.loc[valid].copy()



def render_map():
    st.html('<div class="page-heading"><span class="eyebrow">SAVED RISK MAP</span><h1>See where higher scores appear<span>.</span></h1><p>Zoom into a location. Hover for its score. Filter to narrow the view.</p></div>')
    path = ROOT / 'high_risk_locations.csv'
    if not path.exists():
        st.info('The supplied high-risk locations file is unavailable.')
        return
    frame = locations(path.stat().st_mtime_ns)
    with st.container(border=True, key='map_filters'):
        left, right = st.columns([1.6, 1], gap='large')
        with left:
            minimum = st.slider('Show scores at or above · %', 70, 100, 70, key='map_minimum')
        with right:
            mode = st.selectbox('Show on map', ['Unique coordinates', 'All source records'], key='map_mode')
    filtered = frame.loc[frame.predicted_risk >= minimum / 100].copy()
    points = filtered.groupby(['latitude', 'longitude'], as_index=False).agg(
        predicted_risk=('predicted_risk', 'max'), records=('predicted_risk', 'size'))
    metric_cards([('Matching records', f'{len(filtered):,}', f'of {len(frame):,} supplied records'),
                  ('Locations on map', f'{len(points):,}', 'distinct coordinates'),
                  ('Highest score', f'{filtered.predicted_risk.max():.1%}' if len(filtered) else '—', 'in the current filter'),
                  ('Median score', f'{filtered.predicted_risk.median():.1%}' if len(filtered) else '—', 'in the current filter')])
    if filtered.empty:
        st.info('No records match this score. Lower the minimum to see locations again.')
        return
    left, right = st.columns([2.4, 1], gap='large')
    with left, st.container(border=True, key='map_panel'):
        section_title('EXPLORE LOCATIONS', 'Where are the higher scores?', 'Drag to move · scroll to zoom · hover for details')
        plotted = points if mode == 'Unique coordinates' else filtered.assign(records=1)
        chart(map_chart(plotted), 'locations_map', alt=f'{len(plotted)} markers from supplied high-risk records')
        st.caption('Each unique-coordinate marker shows the highest supplied score there. These are saved model outputs, not live flood observations.')
    with right, st.container(border=True, key='distribution_panel'):
        section_title('READ THE PATTERN', 'How many records in each band?', 'Longer bars mean more saved records.')
        chart(histogram(filtered), 'score_distribution', toolbar=False, alt='Record counts in four high-risk score bands')
        st.caption('The supplied CSV contains high-risk records only; it does not show all assessed locations.')
    with st.expander('Explore and download the filtered records'):
        st.dataframe(filtered.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude', 'predicted_risk': 'Model score'}),
                     hide_index=True, width='stretch', height=280)
        st.download_button('Download filtered CSV', filtered.to_csv(index=False), file_name='riskradar-locations.csv',
                           mime='text/csv', icon=':material/download:')

