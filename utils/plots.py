import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Any

def plot_prediction_gauge(probability: float) -> go.Figure:
    """Membuat chart gauge minimalis dengan indikator warna risiko standar UI/UX."""
    prob_percentage = probability * 100
    
    # Warna risiko standar UI/UX (Hijau, Kuning/Oranye, Merah)
    if probability < 0.35:
        risk_color = '#10B981' # Emerald Green
        risk_text = 'RISIKO RENDAH'
    elif probability < 0.70:
        risk_color = '#F59E0B' # Amber
        risk_text = 'RISIKO SEDANG'
    else:
        risk_color = '#EF4444' # Rose/Red
        risk_text = 'RISIKO TINGGI'
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Diagnosis: {risk_text}", 'font': {'size': 14, 'color': '#475569', 'family': 'Plus Jakarta Sans, sans-serif'}},
        number={'suffix': '%', 'font': {'size': 38, 'color': '#0F172A', 'family': 'Plus Jakarta Sans, sans-serif'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': risk_color, 'thickness': 0.25},
            'bgcolor': "#F1F5F9",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 100], 'color': '#F8FAFC'}
            ]
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=260,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def plot_feature_importance_bar(df_contrib: pd.DataFrame) -> go.Figure:
    """Membuat grafik batang horizontal kontribusi fitur dengan label bahasa Indonesia."""
    df_plot = df_contrib.copy()
    
    # Positif (meningkatkan risiko attrisi) = Coral Red, Negatif (faktor protektif) = Biru Royal
    df_plot['Color'] = df_plot['Contribution'].apply(lambda x: '#F43F5E' if x > 0 else '#3B82F6')
    df_plot = df_plot.sort_values(by='Contribution', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_plot['Label'],
        x=df_plot['Contribution'],
        orientation='h',
        marker_color=df_plot['Color'],
        text=df_plot['Contribution'].apply(lambda x: f"+{x:.3f}" if x > 0 else f"{x:.3f}"),
        textposition='outside',
        hoverinfo='text',
        hovertext=df_plot.apply(
            lambda r: f"<b>{r['Label']}</b><br>Kontribusi: {r['Contribution']:.4f}",
            axis=1
        )
    ))
    
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Nilai Kontribusi Terhadap Risiko",
                font=dict(size=10, color='#64748B')
            ),
            tickfont=dict(size=9, color='#64748B'),
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='#CBD5E1',
            showgrid=False
        ),
        yaxis=dict(
            tickfont=dict(size=10, color='#475569'),
            showgrid=False
        ),
        margin=dict(l=140, r=40, t=10, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=320
    )
    return fig

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Membuat heatmap korelasi interaktif dengan color scale diverging RdBu."""
    df_numeric = df.select_dtypes(include=['int64', 'float64']).copy()
    df_numeric = df_numeric.loc[:, df_numeric.nunique() > 1]
    
    corr = df_numeric.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        zmin=-1.0,
        zmax=1.0,
        colorbar=dict(
            title=dict(text='Korelasi', font=dict(size=10)),
            tickfont=dict(size=9)
        )
    ))
    
    fig.update_layout(
        margin=dict(l=100, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(size=8, color='#64748B')),
        yaxis=dict(tickfont=dict(size=8, color='#64748B')),
        height=450
    )
    return fig
