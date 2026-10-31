"""Two pages for the rainfall and flood monitor."""
from pathlib import Path
import streamlit as st
from rainfall_page import render_rainfall
from riskradar_page import render_flood

ROOT=Path(__file__).resolve().parent


def render_app():
    st.set_page_config(page_title='RiskRadar · Rainfall & Flood',page_icon='◉',layout='wide',initial_sidebar_state='collapsed')
    st.html('<style>'+(ROOT/'assets/dashboard.css').read_text(encoding='utf-8')+
            (ROOT/'assets/monitor.css').read_text(encoding='utf-8')+'</style>')
    with st.container(key='app_navbar'):
        brand,nav=st.columns([1,1],vertical_alignment='center')
        with brand:
            st.html('<div class="app-brand"><span class="app-mark"></span><strong>RiskRadar</strong>'
                    '<span class="brand-description">Rainfall &amp; flood monitor</span></div>')
        with nav,st.container(key='main_navigation'):
            if st.session_state.get('workspace_page') not in ['Rainfall','Flood']:
                st.session_state.workspace_page='Rainfall'
            page=st.radio('Navigate',['Rainfall','Flood'],horizontal=True,key='workspace_page',
                          label_visibility='collapsed',width='stretch')
    if page=='Rainfall':render_rainfall()
    else:render_flood()
    st.html('<div class="page-footer"><span>RiskRadar</span><span>Rainfall &amp; flood risk</span></div>')
