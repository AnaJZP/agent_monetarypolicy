# simple_validation.py
# Versión simplificada con similitud coseno y métricas avanzadas

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Colores IPN
IPN_COLORS = {
    "Restrictiva": "#722F37",
    "Expansiva": "#4A4A4A", 
    "Neutral": "#A8A8A8",
    "primary": "#722F37",
    "secondary": "#B8860B",
    "light": "#8B4A52"
}

def calculate_similarity_metrics(df):
    """Calcula métricas de similitud entre informes"""
    
    # Seleccionar columnas numéricas para análisis
    numeric_cols = ['stance_score', 'confidence_level', 'preocupacion_inflacion', 
                   'preocupacion_crecimiento', 'fortaleza_empleo', 'incertidumbre_politica',
                   'senales_expansivas', 'senales_restrictivas']
    
    # Filtrar columnas que existen
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) < 3:
        return None, None, None
    
    # Preparar datos
    data_matrix = df[available_cols].fillna(df[available_cols].mean())
    
    # Estandarizar para similitud coseno
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_matrix)
    
    # Calcular similitud coseno entre informes
    cosine_sim_matrix = cosine_similarity(data_scaled)
    
    # Métricas de similitud
    similarity_metrics = {
        'similitud_promedio': np.mean(cosine_sim_matrix[np.triu_indices_from(cosine_sim_matrix, k=1)]),
        'similitud_max': np.max(cosine_sim_matrix[np.triu_indices_from(cosine_sim_matrix, k=1)]),
        'similitud_min': np.min(cosine_sim_matrix[np.triu_indices_from(cosine_sim_matrix, k=1)]),
        'consistencia': 1 - np.std(cosine_sim_matrix[np.triu_indices_from(cosine_sim_matrix, k=1)])
    }
    
    # Encontrar informes más similares
    upper_triangle = np.triu_indices_from(cosine_sim_matrix, k=1)
    max_sim_idx = np.argmax(cosine_sim_matrix[upper_triangle])
    most_similar_pair = (upper_triangle[0][max_sim_idx], upper_triangle[1][max_sim_idx])
    
    return cosine_sim_matrix, similarity_metrics, most_similar_pair

def analyze_stance_patterns(df):
    """Analiza patrones en las posturas monetarias"""
    patterns = {}
    
    if 'monetary_stance' in df.columns:
        patterns['distribución_posturas'] = df['monetary_stance'].value_counts().to_dict()
        
        # Calcular entropía (diversidad de posturas)
        stance_probs = df['monetary_stance'].value_counts(normalize=True)
        patterns['entropia_posturas'] = -np.sum(stance_probs * np.log2(stance_probs))
    
    if 'stance_score' in df.columns:
        patterns['stance_stats'] = {
            'media': df['stance_score'].mean(),
            'mediana': df['stance_score'].median(),
            'desviacion': df['stance_score'].std(),
            'rango': df['stance_score'].max() - df['stance_score'].min(),
            'asimetria': stats.skew(df['stance_score']),
            'curtosis': stats.kurtosis(df['stance_score'])
        }
        
        # Detectar cambios de tendencia
        if len(df) > 5:
            # Correlación con índice temporal (tendencia)
            patterns['tendencia_temporal'] = stats.pearsonr(range(len(df)), df['stance_score'])[0]
    
    return patterns

def simple_validate_and_visualize():
    """Validación básica con análisis de similitud"""
    
    # Buscar archivo JSON
    json_files = list(Path('.').glob('*banxico*.json'))
    if not json_files:
        print("❌ No se encontró archivo JSON de Banxico")
        return False
    
    json_file = json_files[0]
    print(f"🔍 Validando: {json_file}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) == 0:
            print("❌ JSON vacío o formato incorrecto")
            return False
        
        df = pd.DataFrame(data)
        print(f"✅ Cargados {len(df)} registros")
        print(f"📊 Columnas: {list(df.columns)}")
        
        # Validaciones básicas
        errors = validate_basic_requirements(df)
        
        # Análisis de similitud
        cosine_matrix, sim_metrics, similar_pair = calculate_similarity_metrics(df)
        
        # Análisis de patrones
        patterns = analyze_stance_patterns(df)
        
        # Mostrar resultados
        print_validation_results(errors, sim_metrics, patterns)
        
        # Crear visualizaciones
        create_enhanced_dashboard(df, cosine_matrix, sim_metrics, similar_pair)
        
        return len(errors) == 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def validate_basic_requirements(df):
    """Validaciones básicas requeridas"""
    errors = []
    
    # Verificar campos clave
    required_fields = ['monetary_stance', 'stance_score', 'confidence_level']
    for field in required_fields:
        if field not in df.columns:
            errors.append(f"Campo faltante: {field}")
        elif df[field].isnull().any():
            errors.append(f"Valores nulos en: {field}")
    
    # Verificar rangos
    if 'stance_score' in df.columns:
        out_of_range = ~df['stance_score'].between(-2, 2)
        if out_of_range.any():
            errors.append(f"Stance scores fuera de rango: {out_of_range.sum()} casos")
    
    if 'confidence_level' in df.columns:
        out_of_range = ~df['confidence_level'].between(0, 1)
        if out_of_range.any():
            errors.append(f"Confidence levels fuera de rango: {out_of_range.sum()} casos")
    
    return errors

def print_validation_results(errors, sim_metrics, patterns):
    """Imprime resultados de validación y análisis"""
    
    if errors:
        print(f"⚠️ {len(errors)} errores encontrados:")
        for error in errors:
            print(f"   • {error}")
    else:
        print("✅ Validación básica exitosa")
    
    # Mostrar métricas de similitud
    if sim_metrics:
        print(f"\n📊 MÉTRICAS DE SIMILITUD:")
        print(f"   • Similitud coseno promedio: {sim_metrics['similitud_promedio']:.3f}")
        print(f"   • Similitud máxima: {sim_metrics['similitud_max']:.3f}")
        print(f"   • Similitud mínima: {sim_metrics['similitud_min']:.3f}")
        print(f"   • Índice de consistencia: {sim_metrics['consistencia']:.3f}")
    
    # Mostrar patrones de stance
    if 'stance_stats' in patterns:
        stats = patterns['stance_stats']
        print(f"\n🎯 ANÁLISIS DE POSTURAS:")
        print(f"   • Stance promedio: {stats['media']:.2f}")
        print(f"   • Volatilidad (std): {stats['desviacion']:.2f}")
        print(f"   • Asimetría: {stats['asimetria']:.2f}")
        
        if 'tendencia_temporal' in patterns:
            tendencia = patterns['tendencia_temporal']
            direccion = "restrictiva" if tendencia > 0 else "expansiva"
            print(f"   • Tendencia temporal: {direccion} ({tendencia:.3f})")

def create_enhanced_dashboard(df, cosine_matrix, sim_metrics, similar_pair):
    """Crea dashboard mejorado con análisis de similitud"""
    
    fig = plt.figure(figsize=(16, 12), dpi=150)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Dashboard de Validación y Similitud - Análisis Banxico', 
                 fontsize=16, fontweight='bold', color=IPN_COLORS['primary'])
    
    # 1. Distribución de posturas (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    if 'monetary_stance' in df.columns:
        stance_counts = df['monetary_stance'].value_counts()
        colors = [IPN_COLORS.get(s, IPN_COLORS['Neutral']) for s in stance_counts.index]
        ax1.pie(stance_counts.values, labels=stance_counts.index, colors=colors, 
               autopct='%1.1f%%', startangle=90)
        ax1.set_title('Distribución de Posturas', fontweight='bold')
    
    # 2. Stance scores vs tiempo (top-center)
    ax2 = fig.add_subplot(gs[0, 1])
    if 'stance_score' in df.columns:
        ax2.plot(range(len(df)), df['stance_score'], 
                color=IPN_COLORS['primary'], linewidth=2, marker='o', markersize=4)
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('Evolución Stance Score', fontweight='bold')
        ax2.set_xlabel('Índice de Informe')
        ax2.set_ylabel('Stance Score')
        ax2.grid(True, alpha=0.3)
    
    # 3. Matriz de similitud coseno (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    if cosine_matrix is not None:
        im = ax3.imshow(cosine_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
        ax3.set_title('Matriz Similitud Coseno', fontweight='bold')
        ax3.set_xlabel('Informe')
        ax3.set_ylabel('Informe')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax3, shrink=0.6)
        cbar.set_label('Similitud', rotation=270, labelpad=15)
    
    # 4. Distribución de similitudes (middle-left)
    ax4 = fig.add_subplot(gs[1, 0])
    if cosine_matrix is not None:
        # Extraer triangular superior (sin diagonal)
        upper_tri = cosine_matrix[np.triu_indices_from(cosine_matrix, k=1)]
        ax4.hist(upper_tri, bins=15, color=IPN_COLORS['secondary'], alpha=0.7, edgecolor='black')
        ax4.axvline(np.mean(upper_tri), color=IPN_COLORS['primary'], linestyle='--', linewidth=2)
        ax4.set_title('Distribución de Similitudes', fontweight='bold')
        ax4.set_xlabel('Similitud Coseno')
        ax4.set_ylabel('Frecuencia')
        ax4.grid(True, alpha=0.3)
    
    # 5. Confidence levels (middle-center)
    ax5 = fig.add_subplot(gs[1, 1])
    if 'confidence_level' in df.columns:
        ax5.hist(df['confidence_level'], bins=12, color=IPN_COLORS['light'], 
                alpha=0.7, edgecolor='black')
        ax5.axvline(df['confidence_level'].mean(), color='red', linestyle='--', linewidth=2)
        ax5.set_title('Niveles de Confianza', fontweight='bold')
        ax5.set_xlabel('Confidence Level')
        ax5.set_ylabel('Frecuencia')
        ax5.grid(True, alpha=0.3)
    
    # 6. Scatter: Stance vs Confidence (middle-right)
    ax6 = fig.add_subplot(gs[1, 2])
    if 'stance_score' in df.columns and 'confidence_level' in df.columns:
        colors = [IPN_COLORS.get(s, IPN_COLORS['Neutral']) for s in df['monetary_stance']]
        scatter = ax6.scatter(df['stance_score'], df['confidence_level'], 
                             c=colors, alpha=0.7, s=60, edgecolor='white', linewidth=1)
        
        # Línea de tendencia
        z = np.polyfit(df['stance_score'], df['confidence_level'], 1)
        p = np.poly1d(z)
        ax6.plot(df['stance_score'], p(df['stance_score']), 
                color=IPN_COLORS['secondary'], linestyle='--', linewidth=2)
        
        ax6.set_title('Stance vs Confianza', fontweight='bold')
        ax6.set_xlabel('Stance Score')
        ax6.set_ylabel('Confidence Level')
        ax6.grid(True, alpha=0.3)
        
        # Correlación
        corr = df['stance_score'].corr(df['confidence_level'])
        ax6.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax6.transAxes,
                fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # 7. Resumen estadístico (bottom span)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Crear texto de resumen
    if sim_metrics:
        stance_mean = df['stance_score'].mean() if 'stance_score' in df.columns and not df['stance_score'].isnull().all() else 0
        conf_mean = df['confidence_level'].mean() if 'confidence_level' in df.columns and not df['confidence_level'].isnull().all() else 0
        
        stats_text = f"""
REPORTE EJECUTIVO DE VALIDACIÓN - POLÍTICA MONETARIA BANXICO

📊 ESTADÍSTICAS BÁSICAS:                           🔍 ANÁLISIS DE SIMILITUD:
• Total de informes: {len(df)}                    • Similitud coseno promedio: {sim_metrics['similitud_promedio']:.3f}
• Posturas únicas: {df['monetary_stance'].nunique() if 'monetary_stance' in df.columns else 'N/A'}                        • Similitud máxima: {sim_metrics['similitud_max']:.3f}
• Stance promedio: {stance_mean:.2f}                       • Similitud mínima: {sim_metrics['similitud_min']:.3f}
• Confianza promedio: {conf_mean:.2f}                   • Consistencia: {sim_metrics['consistencia']:.3f}

🎯 INTERPRETACIÓN:
• Similitud > 0.8: Informes muy consistentes    • Similitud 0.5-0.8: Consistencia moderada
• Similitud < 0.5: Alta variabilidad            • Consistencia > 0.7: Análisis estable
        """
    else:
        stats_text = f"Error en cálculo de similitud - datos insuficientes"
    
    ax7.text(0.02, 0.5, stats_text, fontsize=11, va='center', ha='left',
             bbox=dict(boxstyle="round,pad=0.5", facecolor=IPN_COLORS['Neutral'], alpha=0.1))
    
    plt.tight_layout()
    
    output_path = Path("enhanced_validation_dashboard.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Dashboard mejorado guardado: {output_path}")

if __name__ == "__main__":
    print("=== 🔍 Validación Avanzada con Similitud Coseno ===")
    simple_validate_and_visualize()