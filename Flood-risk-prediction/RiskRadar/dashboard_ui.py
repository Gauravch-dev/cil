"""Data-backed visual components for the existing Streamlit dashboard."""
from html import escape

import plotly.graph_objects as go
import streamlit as st

COLORS = {'ink': '#173c4a', 'muted': '#74858d', 'teal': '#238b87', 'LOW': '#238b87',
          'MEDIUM': '#d59a38', 'HIGH': '#d46655', 'grid': '#edf1f2'}


def section_title(kicker, title, note=''):
    st.html(f'<div class="section-title"><span class="eyebrow">{escape(kicker)}</span><h3>{escape(title)}</h3>'
            f'{"<p>" + escape(note) + "</p>" if note else ""}</div>')


def metric_cards(items,key='summary_stats'):
    with st.container(key=key):
        columns = st.columns(len(items), gap='medium')
        for col, (label, value, detail) in zip(columns, items):
            with col:
                st.html(f'<div class="stat-card"><span>{escape(label)}</span><strong>{escape(str(value))}</strong>'
                        f'<small>{escape(detail)}</small></div>')


def chart(figure, key, toolbar=True, alt=''):
    st.plotly_chart(figure, key=key, width='stretch', theme=None, alt=alt,
                   config={'displayModeBar': toolbar, 'displaylogo': False, 'responsive': True,
                           'modeBarButtonsToRemove': ['lasso2d', 'select2d']})


def layout(figure, height=300):
    return figure.update_layout(height=height, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=16, t=20, b=16),
        font=dict(family='Segoe UI, sans-serif', size=12, color=COLORS['muted']),
        hoverlabel=dict(bgcolor=COLORS['ink'], font_color='white', bordercolor=COLORS['ink']),
        xaxis=dict(showgrid=False, zeroline=False, automargin=True),
        yaxis=dict(gridcolor=COLORS['grid'], zeroline=False, automargin=True))


def gauge(score, level):
    fig = go.Figure(go.Indicator(mode='gauge+number', value=score * 100,
        number=dict(suffix='%', font=dict(size=54, color=COLORS['ink'])),
        title=dict(text='RISK SCORE', font=dict(size=12, color=COLORS['muted'])),
        gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 40, 70, 100], ticks='',
                            tickfont=dict(size=11, color=COLORS['muted'])),
                   borderwidth=0, bgcolor='white',
                   bar=dict(color=COLORS[level], thickness=.56),
                   steps=[dict(range=[0, 40], color='#dcedea'),
                          dict(range=[40, 70], color='#f6ebd5'),
                          dict(range=[70, 100], color='#f6e0dc')],
                   threshold=dict(value=score * 100, thickness=.9,
                                  line=dict(color=COLORS['ink'], width=2)))))
    return layout(fig, 220).update_layout(margin=dict(l=26, r=26, t=36, b=8))


def risk_scale(score):
    st.html(f'''<div class="risk-scale" aria-label="Risk categories: Low up to 40 percent, Medium above 40 to 70 percent, High above 70 percent">
        <div class="scale-track"><span></span><span></span><span></span><i style="left:clamp(4px,{score * 100:.3f}%,calc(100% - 4px))"></i></div>
        <div class="scale-labels"><span>Low<small>0–40%</small></span><span>Medium<small>&gt;40–70%</small></span><span>High<small>&gt;70–100%</small></span></div></div>''')


def what_if_chart(frame, selected_mm, score, trial_mm=None, trial_score=None):
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=40, fillcolor='#eff7f5', line_width=0, layer='below')
    fig.add_hrect(y0=40, y1=70, fillcolor='#fcf7ec', line_width=0, layer='below')
    fig.add_hrect(y0=70, y1=100, fillcolor='#fcf0ed', line_width=0, layer='below')
    fig.add_trace(go.Scatter(x=frame.annual_mm, y=frame.score_pct, mode='lines+markers',
        line=dict(color=COLORS['teal'], width=3), marker=dict(size=7, color=COLORS['teal']),
        name='Rainfall scenarios', hovertemplate='%{x:,.0f} mm / year<br>Risk score %{y:.1f}%<extra></extra>'))
    fig.add_trace(go.Scatter(x=[selected_mm], y=[score * 100], mode='markers',
        marker=dict(color=COLORS['ink'], size=14, line=dict(color='white', width=3)),
        name='Your assessment', hovertemplate='Your submitted scenario<br>%{x:,.0f} mm / year · %{y:.1f}%<extra></extra>'))
    fig.add_vline(x=selected_mm, line_dash='dot', line_color='#a9bcc0', layer='below')
    if trial_mm is not None and trial_mm != selected_mm:
        fig.add_trace(go.Scatter(x=[trial_mm], y=[trial_score * 100], mode='markers',
            marker=dict(color=COLORS['HIGH'], size=15, symbol='diamond', line=dict(color='white', width=2)),
            name='Slider scenario', hovertemplate='Slider scenario<br>%{x:,.0f} mm / year · %{y:.1f}%<extra></extra>'))
    for y, label, color in [(20, 'LOW', COLORS['LOW']), (55, 'MEDIUM', COLORS['MEDIUM']), (85, 'HIGH', COLORS['HIGH'])]:
        fig.add_annotation(x=1, xref='paper', y=y, text=label, showarrow=False,
                           xanchor='right', font=dict(size=10, color=color))
    layout(fig, 290)
    return fig.update_layout(yaxis=dict(range=[0, 105], ticksuffix='%', dtick=20, title=None),
        xaxis=dict(title='Annual rainfall (mm)', tickformat=',.0f'),
        legend=dict(orientation='h', y=1.22, x=0, font_size=11), margin=dict(l=45, r=15, t=55, b=42))


def history_chart(entries):
    values = [entry['result']['flood_risk_score'] * 100 for entry in entries]
    annual = [entry['inputs']['annual_precipitation'] for entry in entries]
    fig = go.Figure(go.Bar(x=[f'Scenario {i + 1}' for i in range(len(entries))], y=values,
        customdata=[[a, e.get('name', 'Custom conditions'), e['result']['risk_level'].title()] for a, e in zip(annual, entries)],
        marker_color=[COLORS[e['result']['risk_level']] for e in entries],
        marker_cornerradius=6, width=.45,
        text=[f'{v:.1f}%' for v in values], textposition='outside', cliponaxis=False,
        hovertemplate='%{x} · %{customdata[1]}<br>Annual rainfall %{customdata[0]:,} mm<br>%{customdata[2]} risk · %{y:.1f}%<extra></extra>'))
    return layout(fig, 235).update_layout(yaxis=dict(range=[0, 100], ticksuffix='%'))




