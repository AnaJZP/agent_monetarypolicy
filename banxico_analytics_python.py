# banxico_analytics_python_fixed.py
# Versión corregida con colores IPN consistentes

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime
from pathlib import Path
import re
from collections import Counter

# Configurar Plotly para generar imágenes sin navegador
pio.kaleido.scope.mathjax = None

# Paleta de colores del IPN - Profesional
IPN_COLORS = {
    "Restrictiva": "#722F37",      # Guinda oscuro
    "Expansiva": "#4A4A4A",        # Gris oscuro
    "Neutral": "#A8A8A8",          # Plateado
    "primary": "#722F37",          # Guinda principal
    "secondary": "#B8860B",        # Dorado/plateado
    "accent": "#2F2F2F",           # Negro suave
    "background": "#FFFFFF",       # Blanco
    "light_guinda": "#8B4A52",     # Guinda claro
    "dark_silver": "#708090",      # Plateado oscuro
    "white": "#FFFFFF",
    "black": "#000000"
}

def load_json_data(json_path='banxico_analysis.json'):
    """Carga datos desde JSON y convierte a DataFrame"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convertir a DataFrame
        df = pd.DataFrame(data)
        
        # Procesar fechas
        if 'report_date' in df.columns:
            df['report_date'] = pd.to_datetime(df['report_date'])
        elif 'processing_timestamp' in df.columns:
            # Usar timestamp si no hay report_date
            df['report_date'] = pd.to_datetime(df['processing_timestamp'])
        
        df = df.sort_values('report_date').reset_index(drop=True)
        
        print(f"✅ JSON cargado: {len(df)} informes")
        print(f"📅 Período: {df['report_date'].min().strftime('%Y-%m')} a {df['report_date'].max().strftime('%Y-%m')}")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ No se encontró {json_path}")
        # Buscar archivos JSON alternativos
        json_files = list(Path('.').glob('*banxico*.json'))
        if json_files:
            print("Archivos JSON disponibles:")
            for f in json_files:
                print(f"   • {f.name}")
        return None
    except Exception as e:
        print(f"❌ Error cargando JSON: {e}")
        return None

def create_stance_evolution_png(df, output_dir):
    """Evolución temporal - PNG profesional con colores IPN"""
    fig = plt.figure(figsize=(16, 10), dpi=150)
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    # Panel principal - Stance Score
    ax1 = fig.add_subplot(gs[0])
    
    # Línea principal
    ax1.plot(df['report_date'], df['stance_score'], 
             color=IPN_COLORS['primary'], linewidth=3, marker='o', markersize=8, 
             markerfacecolor='white', markeredgecolor=IPN_COLORS['primary'], markeredgewidth=2)
    
    # Colorear por postura
    for i in range(len(df)-1):
        stance = df['monetary_stance'].iloc[i]
        color = IPN_COLORS.get(stance, IPN_COLORS['Neutral'])
        ax1.axvspan(df['report_date'].iloc[i], df['report_date'].iloc[i+1], 
                   alpha=0.2, facecolor=color)
    
    # Bandas de referencia
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax1.axhspan(-0.5, 0.5, alpha=0.1, color=IPN_COLORS['Neutral'], label='Zona Neutral')
    ax1.axhspan(0.5, 2, alpha=0.1, color=IPN_COLORS['Restrictiva'], label='Zona Restrictiva')
    ax1.axhspan(-2, -0.5, alpha=0.1, color=IPN_COLORS['Expansiva'], label='Zona Expansiva')
    
    ax1.set_title('Evolución de la Postura Monetaria de Banxico\nAnálisis Cuantitativo de Comunicación (2018-2025)', 
                 fontsize=20, fontweight='bold', pad=20, color=IPN_COLORS['primary'])
    ax1.set_ylabel('Puntuación de Postura\n(-2: Muy Expansiva, +2: Muy Restrictiva)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Panel 2 - Confianza
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(df['report_date'], df['confidence_level'], 
             color=IPN_COLORS['secondary'], linewidth=2, marker='s', markersize=6)
    ax2.set_ylabel('Nivel de\nConfianza', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Panel 3 - Señales
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(df['report_date'], df['senales_restrictivas'], 
             color=IPN_COLORS['accent'], linewidth=2, label='Señales Restrictivas')
    ax3.plot(df['report_date'], df['senales_expansivas'], 
             color=IPN_COLORS['Expansiva'], linewidth=2, label='Señales Expansivas')
    ax3.set_ylabel('Intensidad\nde Señales', fontsize=12)
    ax3.set_xlabel('Fecha del Informe', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Formato de fechas
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    output_path = output_dir / "banxico_stance_evolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Gráfico de evolución guardado: {output_path}")

def create_temporal_stance_timeline(df, output_dir):
    """Gráfico temporal de posturas vs fecha - Línea de tiempo"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), dpi=150)
    fig.suptitle('Línea de Tiempo de Posturas Monetarias de Banxico\nEvolución Cronológica y Cambios de Régimen', 
                 fontsize=18, fontweight='bold', color=IPN_COLORS['primary'])
    
    # Panel superior - Timeline con stance score
    dates = df['report_date']
    scores = df['stance_score']
    stances = df['monetary_stance']
    
    # Línea principal
    ax1.plot(dates, scores, color=IPN_COLORS['primary'], linewidth=4, alpha=0.8, zorder=10)
    
    # Marcadores por tipo de postura
    for stance in stances.unique():
        mask = stances == stance
        color = IPN_COLORS[stance]
        ax1.scatter(dates[mask], scores[mask], 
                   c=color, s=120, alpha=0.8, edgecolor='white', linewidth=2,
                   label=f'Postura {stance}', zorder=15)
    
    # Bandas de contexto
    ax1.axhspan(-2, -0.5, alpha=0.15, color=IPN_COLORS['Expansiva'], label='Zona Expansiva')
    ax1.axhspan(-0.5, 0.5, alpha=0.10, color=IPN_COLORS['Neutral'], label='Zona Neutral')
    ax1.axhspan(0.5, 2, alpha=0.15, color=IPN_COLORS['Restrictiva'], label='Zona Restrictiva')
    ax1.axhline(0, color=IPN_COLORS['black'], linestyle='--', alpha=0.6, linewidth=1)
    
    ax1.set_ylabel('Puntuación de Postura Monetaria', fontsize=14, fontweight='bold', color=IPN_COLORS['primary'])
    ax1.set_title('Evolución del Stance Score a lo Largo del Tiempo', fontsize=16, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, color=IPN_COLORS['dark_silver'])
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.set_ylim(-2.2, 2.2)
    
    # Panel inferior - Distribución temporal de posturas (barras)
    df['year_quarter'] = df['report_date'].dt.year.astype(str) + '-Q' + df['report_date'].dt.quarter.astype(str)
    
    stance_by_period = df.groupby(['year_quarter', 'monetary_stance']).size().unstack(fill_value=0)
    
    # Crear barras apiladas
    bottom = np.zeros(len(stance_by_period))
    
    for stance in ['Expansiva', 'Neutral', 'Restrictiva']:
        if stance in stance_by_period.columns:
            bars = ax2.bar(range(len(stance_by_period)), stance_by_period[stance], 
                          bottom=bottom, color=IPN_COLORS[stance], alpha=0.8, 
                          label=f'Postura {stance}', edgecolor='white', linewidth=1)
            bottom += stance_by_period[stance]
    
    ax2.set_ylabel('Número de Informes', fontsize=14, fontweight='bold', color=IPN_COLORS['primary'])
    ax2.set_xlabel('Período', fontsize=14, fontweight='bold', color=IPN_COLORS['primary'])
    ax2.set_title('Distribución de Posturas por Período', fontsize=16, fontweight='bold', pad=15)
    ax2.set_xticks(range(len(stance_by_period)))
    ax2.set_xticklabels(stance_by_period.index, rotation=45, ha='right')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, color=IPN_COLORS['dark_silver'])
    
    # Formato de fechas para panel superior
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    output_path = output_dir / "banxico_temporal_stance_timeline.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor=IPN_COLORS['primary'])
    plt.close()
    
    print(f"📅 Timeline de posturas guardado: {output_path}")

def create_main_concerns_analysis(df, output_dir):
    """Análisis de preocupaciones principales del reporte"""
    fig = plt.figure(figsize=(18, 14), dpi=150)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1], hspace=0.35, wspace=0.25)
    
    fig.suptitle('Análisis de Preocupaciones Principales en Informes de Banxico\nIdentificación de Temas Centrales por Período', 
                 fontsize=18, fontweight='bold', color=IPN_COLORS['primary'])
    
    # 1. Heatmap de preocupaciones a lo largo del tiempo (panel principal)
    ax1 = fig.add_subplot(gs[0, :])
    
    concern_metrics = ['preocupacion_inflacion', 'preocupacion_crecimiento', 'fortaleza_empleo', 'incertidumbre_politica']
    concern_labels = ['Preocupación\nInflación', 'Preocupación\nCrecimiento', 'Fortaleza\nEmpleo', 'Incertidumbre\nPolítica']
    
    # Crear matriz de datos para heatmap
    concern_data = df[concern_metrics].T
    concern_data.index = concern_labels
    concern_data.columns = [d.strftime('%Y-%m') for d in df['report_date']]
    
    # Crear colormap personalizado IPN
    ipn_cmap = LinearSegmentedColormap.from_list('ipn', 
                                                [IPN_COLORS['white'], IPN_COLORS['light_guinda'], IPN_COLORS['primary']])
    
    sns.heatmap(concern_data, 
                ax=ax1, 
                cmap=ipn_cmap,
                cbar_kws={'label': 'Nivel de Intensidad', 'shrink': 0.6},
                linewidths=1, 
                linecolor='white',
                square=False,
                fmt='.2f')
    
    ax1.set_title('Mapa de Calor: Intensidad de Preocupaciones por Período', 
                 fontsize=16, fontweight='bold', pad=15, color=IPN_COLORS['primary'])
    ax1.set_xlabel('Fecha del Informe', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Tipo de Preocupación', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Preocupación dominante por período (barras)
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Identificar preocupación principal en cada período
    dominant_concerns = []
    for _, row in df.iterrows():
        concerns = {
            'Inflación': row['preocupacion_inflacion'],
            'Crecimiento': row['preocupacion_crecimiento'],
            'Empleo': row['fortaleza_empleo'],
            'Incertidumbre': row['incertidumbre_politica']
        }
        dominant = max(concerns.keys(), key=lambda k: concerns[k])
        dominant_concerns.append(dominant)
    
    df['dominant_concern'] = dominant_concerns
    concern_counts = df['dominant_concern'].value_counts()
    
    colors = [IPN_COLORS['primary'], IPN_COLORS['light_guinda'], IPN_COLORS['secondary'], IPN_COLORS['dark_silver']]
    bars = ax2.bar(concern_counts.index, concern_counts.values, 
                   color=colors[:len(concern_counts)], alpha=0.8, edgecolor='white', linewidth=2)
    
    ax2.set_title('Preocupación Dominante\nFrecuencia por Tipo', fontweight='bold', color=IPN_COLORS['primary'])
    ax2.set_ylabel('Número de Informes', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, color=IPN_COLORS['dark_silver'])
    
    # Agregar valores en las barras
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Evolución de la preocupación principal
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Crear serie temporal de la métrica más alta en cada momento
    max_concerns = df[concern_metrics].max(axis=1)
    ax3.plot(df['report_date'], max_concerns, 
             color=IPN_COLORS['primary'], linewidth=3, marker='o', markersize=8,
             markerfacecolor='white', markeredgecolor=IPN_COLORS['primary'], markeredgewidth=2)
    ax3.fill_between(df['report_date'], max_concerns, alpha=0.3, color=IPN_COLORS['light_guinda'])
    
    ax3.set_title('Intensidad de Preocupación\nPrincipal en el Tiempo', fontweight='bold', color=IPN_COLORS['primary'])
    ax3.set_ylabel('Nivel Máximo', fontweight='bold')
    ax3.grid(True, alpha=0.3, color=IPN_COLORS['dark_silver'])
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Análisis de cambios de enfoque (panel inferior)
    ax4 = fig.add_subplot(gs[2, :])
    
    # Detectar cambios en preocupación dominante
    concern_changes = df['dominant_concern'] != df['dominant_concern'].shift(1)
    change_points = df[concern_changes]
    
    if len(change_points) > 1:
        # Crear timeline de cambios
        y_positions = {'Inflación': 3, 'Crecimiento': 2, 'Empleo': 1, 'Incertidumbre': 0}
        
        for i, (_, row) in enumerate(change_points.iterrows()):
            concern = row['dominant_concern']
            y_pos = y_positions[concern]
            
            # Punto de cambio
            ax4.scatter(row['report_date'], y_pos, s=150, 
                       color=IPN_COLORS['primary'], edgecolor='white', linewidth=2, zorder=10)
            
            # Línea conectora horizontal
            if i < len(change_points) - 1:
                next_date = change_points.iloc[i + 1]['report_date']
                ax4.hlines(y_pos, row['report_date'], next_date, 
                          colors=IPN_COLORS['light_guinda'], linewidth=4, alpha=0.7)
        
        ax4.set_yticks(list(y_positions.values()))
        ax4.set_yticklabels(list(y_positions.keys()))
        ax4.set_title('Timeline de Cambios en Preocupación Dominante', 
                     fontsize=16, fontweight='bold', color=IPN_COLORS['primary'])
        ax4.set_xlabel('Fecha', fontweight='bold')
        ax4.set_ylabel('Tipo de Preocupación', fontweight='bold')
        ax4.grid(True, alpha=0.3, color=IPN_COLORS['dark_silver'])
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    output_path = output_dir / "banxico_main_concerns_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white',
                edgecolor=IPN_COLORS['primary'])
    plt.close()
    
    print(f"🎯 Análisis de preocupaciones principales guardado: {output_path}")

def create_policy_radar_png(df, output_dir):
    """Radar de políticas comparativo"""
    fig = plt.figure(figsize=(15, 10), dpi=150)
    
    # Seleccionar períodos
    latest = df.iloc[-1]
    earliest = df.iloc[0] 
    mid_point = df.iloc[len(df)//2]
    
    metrics = ['preocupacion_inflacion', 'preocupacion_crecimiento', 'fortaleza_empleo',
               'incertidumbre_politica', 'senales_expansivas', 'senales_restrictivas']
    labels = ['Preocupación\nInflación', 'Preocupación\nCrecimiento', 'Fortaleza\nEmpleo',
              'Incertidumbre\nPolítica', 'Señales\nExpansivas', 'Señales\nRestrictivas']
    
    # Crear 3 subplots de radar
    fig.suptitle('Evolución de Patrones de Señales de Política Monetaria\nComparación por Intensidad en Diferentes Dimensiones', 
                 fontsize=18, fontweight='bold', y=0.95, color=IPN_COLORS['primary'])
    
    periods = [
        ('Período Inicial', earliest, IPN_COLORS['secondary'], 0),
        ('Período Medio', mid_point, IPN_COLORS['Neutral'], 1), 
        ('Período Reciente', latest, IPN_COLORS['primary'], 2)
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    
    for name, data, color, idx in periods:
        ax = fig.add_subplot(1, 3, idx+1, projection='polar')
        
        values = [data[metric] for metric in metrics]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=name)
        ax.fill(angles, values, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Título con fecha
        date_str = data['report_date'].strftime('%Y-%m')
        ax.set_title(f'{name}\n({date_str})', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_path = output_dir / "banxico_policy_radar_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"🎯 Radar comparativo guardado: {output_path}")

def create_correlation_matrix_png(df, output_dir):
    """Matriz de correlaciones profesional"""
    # Variables para correlación
    corr_vars = ['stance_score', 'confidence_level', 'preocupacion_inflacion',
                'preocupacion_crecimiento', 'fortaleza_empleo', 'incertidumbre_politica',
                'senales_expansivas', 'senales_restrictivas']
    
    corr_matrix = df[corr_vars].corr()
    
    # Labels más descriptivos
    labels = ['Postura\nMonetaria', 'Nivel de\nConfianza', 'Preocup.\nInflación',
              'Preocup.\nCrecimiento', 'Fortaleza\nEmpleo', 'Incertidumbre\nPolítica',
              'Señales\nExpansivas', 'Señales\nRestrictivas']
    
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    
    # Crear heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    heatmap = sns.heatmap(corr_matrix, 
                         annot=True, 
                         fmt='.2f',
                         cmap='RdBu_r',
                         center=0,
                         mask=mask,
                         square=True,
                         xticklabels=labels,
                         yticklabels=labels,
                         cbar_kws={'shrink': 0.8},
                         ax=ax)
    
    ax.set_title('Matriz de Correlaciones de Métricas de Política Monetaria\nAnálisis de Interdependencias entre Señales de Comunicación', 
                fontsize=16, fontweight='bold', pad=20, color=IPN_COLORS['primary'])
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = output_dir / "banxico_correlation_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"🔥 Matriz de correlaciones guardada: {output_path}")

def create_stance_distribution_png(df, output_dir):
    """Distribución y transiciones de posturas"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    fig.suptitle('Análisis de Distribución y Transiciones de Posturas Monetarias', 
                 fontsize=18, fontweight='bold', color=IPN_COLORS['primary'])
    
    # 1. Distribución de posturas
    stance_counts = df['monetary_stance'].value_counts()
    colors = [IPN_COLORS[stance] for stance in stance_counts.index]
    
    wedges, texts, autotexts = ax1.pie(stance_counts.values, 
                                      labels=stance_counts.index,
                                      colors=colors,
                                      autopct='%1.1f%%',
                                      startangle=90)
    ax1.set_title('Distribución de Posturas\nMonetarias (%)', fontweight='bold')
    
    # 2. Evolución por año
    df['year'] = df['report_date'].dt.year
    yearly_stance = df.groupby(['year', 'monetary_stance']).size().unstack(fill_value=0)
    
    yearly_stance.plot(kind='bar', stacked=True, ax=ax2, 
                      color=[IPN_COLORS[col] for col in yearly_stance.columns])
    ax2.set_title('Evolución Anual de Posturas', fontweight='bold')
    ax2.set_xlabel('Año')
    ax2.set_ylabel('Número de Informes')
    ax2.legend(title='Postura', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Histograma de stance scores
    ax3.hist(df['stance_score'], bins=15, alpha=0.7, color=IPN_COLORS['primary'], edgecolor='black')
    ax3.axvline(df['stance_score'].mean(), color=IPN_COLORS['accent'], linestyle='--', 
               linewidth=2, label=f'Promedio: {df["stance_score"].mean():.2f}')
    ax3.set_title('Distribución de Puntuaciones\nde Postura', fontweight='bold')
    ax3.set_xlabel('Stance Score')
    ax3.set_ylabel('Frecuencia')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Volatilidad temporal
    df['stance_volatility'] = df['stance_score'].rolling(window=3, center=True).std()
    ax4.plot(df['report_date'], df['stance_volatility'], 
            color=IPN_COLORS['accent'], linewidth=2, marker='o')
    ax4.set_title('Volatilidad de la Postura\n(Desviación Estándar Móvil)', fontweight='bold')
    ax4.set_xlabel('Fecha')
    ax4.set_ylabel('Volatilidad (3 períodos)')
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    
    output_path = output_dir / "banxico_stance_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📈 Análisis de posturas guardado: {output_path}")

def generate_summary_report(df, output_dir):
    """Genera reporte ejecutivo en texto"""
    report = f"""
=== REPORTE EJECUTIVO: ANÁLISIS DE POLÍTICA MONETARIA DE BANXICO ===

📊 RESUMEN DEL PERÍODO:
• Período analizado: {df['report_date'].min().strftime('%B %Y')} - {df['report_date'].max().strftime('%B %Y')}
• Total de informes: {len(df)}
• Duración: {(df['report_date'].max() - df['report_date'].min()).days / 365.25:.1f} años

🎯 ESTADÍSTICAS DE POSTURA:
• Postura promedio: {df['stance_score'].mean():.2f}
• Desviación estándar: {df['stance_score'].std():.2f}
• Postura más restrictiva: {df['stance_score'].max():.2f} ({df.loc[df['stance_score'].idxmax(), 'report_date'].strftime('%B %Y')})
• Postura más expansiva: {df['stance_score'].min():.2f} ({df.loc[df['stance_score'].idxmin(), 'report_date'].strftime('%B %Y')})

📋 DISTRIBUCIÓN DE POSTURAS:
"""
    
    for stance, count in df['monetary_stance'].value_counts().items():
        percentage = (count / len(df)) * 100
        report += f"• {stance}: {count} informes ({percentage:.1f}%)\n"
    
    report += f"""
📈 ANÁLISIS DE TENDENCIAS:
• Tendencia reciente (últimos 5): {"Más restrictiva" if df['stance_score'].tail(5).mean() > df['stance_score'].head(5).mean() else "Más expansiva"}
• Volatilidad promedio: {df['stance_score'].std():.2f}
• Nivel de confianza promedio: {df['confidence_level'].mean():.2f}

🔑 MÉTRICAS CLAVE (PROMEDIOS):
• Preocupación por inflación: {df['preocupacion_inflacion'].mean():.2f}
• Preocupación por crecimiento: {df['preocupacion_crecimiento'].mean():.2f}
• Fortaleza del empleo: {df['fortaleza_empleo'].mean():.2f}
• Incertidumbre política: {df['incertidumbre_politica'].mean():.2f}

Reporte generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(output_dir / "reporte_ejecutivo.txt", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📋 Reporte ejecutivo guardado: reporte_ejecutivo.txt")

def generate_key_signals_report(df_signals, output_dir):
    """Genera reporte de las señales más relevantes por fecha"""
    
    # Seleccionar señales más significativas (por longitud y stance extremo)
    df_signals['abs_stance'] = df_signals['stance_score'].abs()
    df_signals['signal_length'] = df_signals['key_policy_signal'].astype(str).str.len()
    
    # Top 10 señales más significativas
    top_signals = df_signals.nlargest(10, 'abs_stance')
    
    report = f"""
=== REPORTE DE SEÑALES CLAVE DE POLÍTICA MONETARIA ===

📅 ANÁLISIS DE COMUNICACIÓN BANXICO
Período: {df_signals['report_date'].min().strftime('%B %Y')} - {df_signals['report_date'].max().strftime('%B %Y')}

🔑 TOP 10 SEÑALES MÁS SIGNIFICATIVAS (por intensidad de postura):

"""
    
    for i, (_, signal) in enumerate(top_signals.iterrows(), 1):
        date_str = signal['report_date'].strftime('%B %Y')
        stance_str = signal['monetary_stance']
        score_str = f"{signal['stance_score']:+.2f}"
        
        report += f"{i:2d}. [{date_str}] {stance_str} ({score_str})\n"
        report += f"    \"{signal['key_policy_signal']}\"\n\n"
    
    # Estadísticas adicionales
    avg_length = df_signals['signal_length'].mean()
    most_complex = df_signals.loc[df_signals['signal_length'].idxmax()]
    
    report += f"""
📊 ESTADÍSTICAS DE SEÑALES:
• Total de señales analizadas: {len(df_signals)}
• Longitud promedio: {avg_length:.0f} caracteres
• Señal más compleja: {most_complex['report_date'].strftime('%B %Y')} ({most_complex['signal_length']} caracteres)

🎯 PATRONES IDENTIFICADOS:
"""
    
    # Analizar patrones por postura
    for stance in ['Restrictiva', 'Expansiva', 'Neutral']:
        stance_data = df_signals[df_signals['monetary_stance'] == stance]
        if not stance_data.empty:
            avg_len = stance_data['signal_length'].mean()
            count = len(stance_data)
            report += f"• {stance}: {count} señales, {avg_len:.0f} caracteres promedio\n"
    
    report += f"""
Reporte generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
════════════════════════════════════════════════════════
"""
    
    # Guardar reporte
    with open(output_dir / "key_policy_signals_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📋 Reporte de señales clave guardado: key_policy_signals_report.txt")

def create_key_policy_signals_analysis(df, output_dir):
    """Análisis de señales clave de política monetaria por fecha"""
    fig = plt.figure(figsize=(18, 12), dpi=150)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1], hspace=0.35, wspace=0.25)
    
    fig.suptitle('Análisis de Señales Clave de Política Monetaria de Banxico\nExtracción de Frases Relevantes por Período', 
                 fontsize=18, fontweight='bold', color=IPN_COLORS['primary'])
    
    # 1. Timeline de señales clave (panel principal)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Preparar datos para timeline
    df_signals = df[['report_date', 'key_policy_signal', 'stance_score', 'monetary_stance']].copy()
    df_signals = df_signals.dropna(subset=['key_policy_signal'])
    
    if len(df_signals) == 0:
        # Si no hay señales, crear un gráfico simple
        ax1.text(0.5, 0.5, 'No hay señales clave disponibles en los datos', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=16)
        ax1.set_title('Timeline de Señales Clave de Política Monetaria (Sin Datos)', 
                     fontsize=16, fontweight='bold', color=IPN_COLORS['primary'])
    else:
        # Crear colores por postura
        colors = [IPN_COLORS.get(stance, IPN_COLORS['Neutral']) for stance in df_signals['monetary_stance']]
        
        # Scatter plot con señales en el eje Y (usando índice) y fechas en X
        y_positions = range(len(df_signals))
        scatter = ax1.scatter(df_signals['report_date'], y_positions, 
                             c=colors, s=100, alpha=0.7, edgecolor='white', linewidth=2)
        
        # Agregar texto de señales (truncado para legibilidad)
        for i, (_, row) in enumerate(df_signals.iterrows()):
            signal_text = str(row['key_policy_signal'])[:50] + "..." if len(str(row['key_policy_signal'])) > 50 else str(row['key_policy_signal'])
            ax1.text(row['report_date'], i, f"  {signal_text}", 
                    fontsize=9, va='center', ha='left',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, 
                             edgecolor=IPN_COLORS.get(row['monetary_stance'], 'gray')))
        
        ax1.set_title('Timeline de Señales Clave de Política Monetaria', 
                     fontsize=16, fontweight='bold', pad=15, color=IPN_COLORS['primary'])
        ax1.set_xlabel('Fecha del Informe', fontweight='bold')
        ax1.set_ylabel('Orden Cronológico de Informes', fontweight='bold')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.tick_params(axis='x', rotation=45)
    
    ax1.grid(True, alpha=0.3, color=IPN_COLORS['dark_silver'])
    
    # 2. Análisis de frecuencia de palabras clave (izquierda inferior)
    ax2 = fig.add_subplot(gs[1, 0])
    
    if len(df_signals) > 0:
        # Extraer palabras clave más comunes
        all_signals = ' '.join(df_signals['key_policy_signal'].astype(str))
        # Palabras relevantes para política monetaria (filtrar stop words)
        words = re.findall(r'\b[a-záéíóúñ]{4,}\b', all_signals.lower())
        
        # Filtrar palabras específicas de política monetaria
        monetary_keywords = ['inflación', 'tasa', 'tasas', 'crecimiento', 'riesgo', 'riesgos', 
                            'objetivo', 'estabilidad', 'política', 'monetaria', 'banxico',
                            'economía', 'económica', 'precios', 'expectativas', 'decisión']
        
        filtered_words = [word for word in words if word in monetary_keywords]
        word_counts = Counter(filtered_words)
        
        if word_counts:
            top_words = word_counts.most_common(8)
            words_labels, word_frequencies = zip(*top_words)
            
            bars = ax2.barh(words_labels, word_frequencies, color=IPN_COLORS['secondary'], alpha=0.7)
            ax2.set_title('Palabras Clave Más Frecuentes\nen Señales de Política', fontweight='bold')
            ax2.set_xlabel('Frecuencia')
            
            # Agregar valores en barras
            for bar in bars:
                width = bar.get_width()
                ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{int(width)}', ha='left', va='center', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No se encontraron\npalabras clave relevantes', 
                    ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'Sin datos disponibles', 
                ha='center', va='center', transform=ax2.transAxes)
    
    ax2.set_title('Palabras Clave Más Frecuentes\nen Señales de Política', fontweight='bold')
    
    # 3. Intensidad de señales por postura (derecha inferior)
    ax3 = fig.add_subplot(gs[1, 1])
    
    if len(df_signals) > 0:
        # Calcular longitud promedio de señales por postura
        signal_lengths = df.groupby('monetary_stance')['key_policy_signal'].apply(
            lambda x: x.astype(str).str.len().mean()
        ).dropna()
        
        if len(signal_lengths) > 0:
            colors_stance = [IPN_COLORS.get(stance, IPN_COLORS['Neutral']) for stance in signal_lengths.index]
            bars = ax3.bar(signal_lengths.index, signal_lengths.values, 
                           color=colors_stance, alpha=0.7, edgecolor='white', linewidth=2)
            
            # Agregar valores en barras
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            
            ax3.set_ylabel('Caracteres Promedio')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'Sin datos suficientes\npara análisis', 
                    ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'Sin datos disponibles', 
                ha='center', va='center', transform=ax3.transAxes)
    
    ax3.set_title('Longitud Promedio de Señales\npor Tipo de Postura', fontweight='bold')
    
    # 4. Evolución de complejidad de señales (panel inferior completo)
    ax4 = fig.add_subplot(gs[2, :])
    
    # Calcular complejidad de señales (número de palabras)
    df['signal_complexity'] = df['key_policy_signal'].astype(str).str.split().str.len()
    df['signal_complexity'] = df['signal_complexity'].fillna(0)  # Llenar NaN con 0
    
    # Crear gráfico de línea con barras de complejidad
    ax4_twin = ax4.twinx()
    
    # Línea de stance score
    line = ax4.plot(df['report_date'], df['stance_score'], 
                   color=IPN_COLORS['primary'], linewidth=3, marker='o', markersize=6,
                   label='Stance Score')
    
    # Barras de complejidad
    bars = ax4_twin.bar(df['report_date'], df['signal_complexity'], 
                       alpha=0.3, color=IPN_COLORS['secondary'], width=20,
                       label='Complejidad Señal (# palabras)')
    
    ax4.set_xlabel('Fecha del Informe', fontweight='bold')
    ax4.set_ylabel('Stance Score', fontweight='bold', color=IPN_COLORS['primary'])
    ax4_twin.set_ylabel('Complejidad de Señal (palabras)', fontweight='bold', color=IPN_COLORS['secondary'])
    
    ax4.set_title('Evolución de Stance Score vs Complejidad de Señales Clave', 
                 fontsize=16, fontweight='bold', color=IPN_COLORS['primary'])
    
    # Leyendas combinadas
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax4.grid(True, alpha=0.3, color=IPN_COLORS['dark_silver'])
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    output_path = output_dir / "banxico_key_policy_signals.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"🔍 Análisis de señales clave guardado: {output_path}")
    
    # Generar reporte de señales destacadas solo si hay datos
    if len(df_signals) > 0:
        generate_key_signals_report(df_signals, output_dir)

def main():
    """Función principal"""
    print("=== 📊 Análisis Visual Avanzado de Banxico (JSON → PNG) ===")
    
    # Cargar datos del JSON
    df = load_json_data()
    
    if df is None:
        return
    
    # Crear directorio de salida
    output_dir = Path("banxico_visualizaciones")
    output_dir.mkdir(exist_ok=True)
    
    # Generar todas las visualizaciones
    try:
        create_stance_evolution_png(df, output_dir)
        create_temporal_stance_timeline(df, output_dir)
        create_main_concerns_analysis(df, output_dir)
        create_policy_radar_png(df, output_dir)
        create_correlation_matrix_png(df, output_dir)
        create_stance_distribution_png(df, output_dir)
        create_key_policy_signals_analysis(df, output_dir)  # Función corregida
        generate_summary_report(df, output_dir)
        
        print(f"\n🎉 ¡Análisis completado!")
        print(f"📁 Archivos guardados en: {output_dir.absolute()}")
        print("📊 Gráficos generados:")
        print("   • banxico_stance_evolution.png")
        print("   • banxico_temporal_stance_timeline.png")
        print("   • banxico_main_concerns_analysis.png")
        print("   • banxico_policy_radar_comparison.png") 
        print("   • banxico_correlation_matrix.png")
        print("   • banxico_stance_analysis.png")
        print("   • banxico_key_policy_signals.png")
        print("   • reporte_ejecutivo.txt")
        print("   • key_policy_signals_report.txt (si hay datos disponibles)")
        print("\n✨ Todos los gráficos están listos para presentaciones!")
        
    except Exception as e:
        print(f"❌ Error durante la generación: {e}")
        print("🔧 Revisa que el archivo JSON tenga las columnas necesarias")

if __name__ == "__main__":
    main()