# analyzer.py
# VERSIÓN ASÍNCRONA para análisis en paralelo con orden cronológico

import json
import re
import asyncio
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback

def extract_date_from_filename(filename: str):
    """
    Extrae fecha del nombre del archivo de informe de Banxico para ordenamiento cronológico.
    Formatos esperados: 'abril-junio 2024.pdf', 'enero-marzo 2025.pdf', etc.
    """
    # Diccionario de meses en español
    months_map = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
        'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
        'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    
    # Extraer año
    year_match = re.search(r'(\d{4})', filename)
    if not year_match:
        return datetime(1900, 1, 1)  # Fecha por defecto para archivos sin año
    
    year = int(year_match.group(1))
    
    # Extraer trimestre basado en el patrón de nombres
    if 'enero-marzo' in filename.lower():
        month = 3  # Q1
    elif 'abril-junio' in filename.lower():
        month = 6  # Q2
    elif 'julio-septiembre' in filename.lower():
        month = 9  # Q3
    elif 'octubre-diciembre' in filename.lower():
        month = 12 # Q4
    else:
        # Buscar cualquier mes individual
        for month_name, month_num in months_map.items():
            if month_name in filename.lower():
                month = month_num
                break
        else:
            month = 1  # Por defecto enero
    
    return datetime(year, month, 1)

async def analyze_theme_research_grade(theme_name: str, search_query: str, vector_store, llm, report_filename: str):
    """
    Análisis temático asíncrono de grado investigativo.
    """
    print(f"--> Analizando {theme_name} para {report_filename}...")
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 25, 'filter': {'source_document': report_filename}}
    )
    
    try:
        # ASYNC: Usamos ainvoke para la recuperación de documentos
        context_docs = await retriever.ainvoke(search_query)
    except Exception as e:
        print(f"    ⚠️  Error en recuperación con filtro: {e}")
        context_docs = []
    
    # Búsqueda de respaldo si es necesario
    if not context_docs:
        print(f"    ⚠️  No se encontraron documentos con filtro para {report_filename}. Intentando búsqueda general...")
        try:
            retriever_general = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 40})
            all_docs = await retriever_general.ainvoke(search_query)
            context_docs = [doc for doc in all_docs if doc.metadata.get('source_document') == report_filename]
        except Exception as e:
            print(f"    ❌ Error en búsqueda general: {e}")
            return f"• Error recuperando información sobre {theme_name} en este informe: {str(e)}", 0.0

    if not context_docs:
        return f"• No se encontró información relevante sobre {theme_name} en este informe.", 0.0
    
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    
    # === PROMPTS ADAPTADOS PARA BANXICO - VERSIÓN CONCISA ===
    theme_prompts = {
        "inflacion": """
        Analiza este texto de informe trimestral de Banxico sobre inflación. Sé CONCISO y directo.
        Proporciona un análisis en máximo 200 palabras cubriendo:
        
        **Evaluación Actual:**
        • Nivel actual de inflación (general y subyacente) vs objetivo 3% +/- 1%
        • Factores clave que impulsan la dinámica inflacionaria
        
        **Perspectivas:**
        • Pronósticos y balance de riesgos
        • Implicaciones para política monetaria
        
        Texto: {context}
        
        ANÁLISIS DE INFLACIÓN (MÁXIMO 200 PALABRAS):
        """,
        
        "tasas_de_interes": """
        Analiza este texto de informe trimestral de Banxico sobre política monetaria. Sé CONCISO.
        Análisis en máximo 200 palabras cubriendo:
        
        **Decisión de Política:**
        • Decisión sobre Tasa Objetivo y justificación
        • Factores económicos clave que motivaron la decisión
        
        **Perspectivas:**
        • Guía a futuro sobre trayectoria de tasas
        • Postura relativa México vs Estados Unidos
        
        Texto: {context}
        
        ANÁLISIS DE TASAS (MÁXIMO 200 PALABRAS):
        """,
        
        "actividad_economica_empleo": """
        Analiza este texto de informe trimestral de Banxico sobre economía real. Sé CONCISO.
        Análisis en máximo 200 palabras cubriendo:
        
        **Condiciones Actuales:**
        • Crecimiento del PIB y componentes principales
        • Mercado laboral: desocupación, empleo formal, salarios
        
        **Implicaciones:**
        • Condiciones de holgura económica
        • Impacto en presiones inflacionarias futuras
        
        Texto: {context}
        
        ANÁLISIS ECONÓMICO (MÁXIMO 200 PALABRAS):
        """
    }
    
    prompt_template = theme_prompts.get(theme_name, theme_prompts["inflacion"])
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    
    try:
        with get_openai_callback() as cb:
            # ASYNC: Usamos ainvoke para la llamada al LLM
            analysis = await chain.ainvoke({"context": context_text})
            cost = cb.total_cost
        
        print(f"    ✅ {theme_name}: Análisis generado para {report_filename}.")
        return analysis.strip(), cost
        
    except Exception as e:
        print(f"    ❌ Error procesando {theme_name} en {report_filename}: {e}")
        return f"• Error analizando {theme_name}: {str(e)}", 0.0

async def get_quantitative_scores(context_text: str, llm):
    """
    Extrae puntajes cuantitativos de forma asíncrona.
    """
    scoring_prompt = ChatPromptTemplate.from_template(
        """
        Como analista cuantitativo de Banxico, extrae puntajes numéricos de este contexto de informe trimestral.
        Devuelve ÚNICAMENTE un objeto JSON con estas claves exactas y valores float:
        {{
            "preocupacion_inflacion": 0.0, "preocupacion_crecimiento": 0.0,
            "fortaleza_empleo": 0.0, "incertidumbre_politica": 0.0,
            "senales_expansivas": 0.0, "senales_restrictivas": 0.0
        }}
        
        Guía de puntuación (0.0 = bajo/débil, 1.0 = alto/fuerte):
        - preocupacion_inflacion: Qué tan preocupado está Banxico por los riesgos inflacionarios.
        - preocupacion_crecimiento: Nivel de preocupación sobre la actividad económica/desaceleración.
        - fortaleza_empleo: Qué tan fuerte es la situación del mercado laboral y la economía en general.
        - incertidumbre_politica: Grado de incertidumbre en las señales de política (riesgos, dudas, factores externos/internos).
        - senales_expansivas: Fuerza del lenguaje que sugiere una política monetaria más laxa (Dovish).
        - senales_restrictivas: Fuerza del lenguaje que sugiere una política monetaria más apretada (Hawkish).
        
        Contexto: {context}
        
        JSON:
        """
    )
    
    try:
        chain = scoring_prompt | llm | StrOutputParser()
        with get_openai_callback() as cb:
            # ASYNC: Usamos ainvoke
            scores_str = await chain.ainvoke({"context": context_text})
            cost = cb.total_cost
        
        # Extraer JSON del string de respuesta
        json_match = re.search(r'\{.*\}', scores_str, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
        else:
            scores = json.loads(scores_str.strip())
            
        return scores, cost
        
    except Exception as e:
        print(f"    Advertencia: No se pudieron extraer los puntajes cuantitativos: {e}")
        return {
            "preocupacion_inflacion": 0.0, "preocupacion_crecimiento": 0.0, 
            "fortaleza_empleo": 0.0, "incertidumbre_politica": 0.0,
            "senales_expansivas": 0.0, "senales_restrictivas": 0.0
        }, 0.0

async def get_report_analysis(report_filename: str, vector_store, llm):
    """
    Análisis completo asíncrono para un único informe de Banxico.
    """
    print(f"-> Iniciando análisis para: {report_filename}...")
    total_cost = 0.0
    
    # Recuperar todo el contenido del informe
    try:
        full_report_retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={'k': 100}
        )
        global_docs = await full_report_retriever.ainvoke(
            "política monetaria Banco de México decisión perspectiva Junta de Gobierno"
        )
        filtered_docs = [doc for doc in global_docs if doc.metadata.get('source_document') == report_filename]
    except Exception as e:
        print(f"!! Error recuperando documentos para {report_filename}: {e}")
        return None, 0.0
    
    if not filtered_docs:
        print(f"!! No se encontró contenido para {report_filename}")
        return None, 0.0
        
    context_text = "\n\n".join([doc.page_content for doc in filtered_docs])
    
    # Prompt mejorado para análisis global - VERSION CONCISA
    global_analysis_prompt = ChatPromptTemplate.from_template(
        """
        Analiza este informe trimestral de Banxico. Devuelve SOLO el JSON, sin explicaciones adicionales:
        {{
            "monetary_stance": "Restrictiva" o "Expansiva" o "Neutral",
            "stance_score": float entre -2.0 y +2.0,
            "confidence_level": float entre 0.0 y 1.0,
            "forward_guidance": "Guía futura de Banxico en 1 frase (máx 100 caracteres)",
            "key_policy_signal": "Señal principal en 1 frase (máx 80 caracteres)"
        }}
        
        DEFINICIONES:
        **RESTRICTIVA**: AUMENTOS de tasas o mantenimiento alto + lenguaje anti-inflacionario
        **EXPANSIVA**: REDUCCIONES de tasas + lenguaje pro-crecimiento  
        **NEUTRAL**: MANTENIMIENTO sin sesgo claro + comunicación dependiente de datos
        
        PUNTAJES:
        +2.0: Muy restrictiva | +1.0: Restrictiva | 0.0: Neutral | -1.0: Expansiva | -2.0: Muy expansiva
        
        Contexto: {context}
        
        JSON:
        """
    )
    
    stage_1_chain = global_analysis_prompt | llm | StrOutputParser()
    
    try:
        with get_openai_callback() as cb:
            global_analysis_str = await stage_1_chain.ainvoke({"context": context_text})
            total_cost += cb.total_cost
        
        # Extraer JSON del string de respuesta
        json_match = re.search(r'\{.*\}', global_analysis_str, re.DOTALL)
        if json_match:
            final_analysis = json.loads(json_match.group())
        else:
            final_analysis = json.loads(global_analysis_str.strip())
            
    except Exception as e:
        print(f"!! Error en análisis global para {report_filename}: {e}")
        return None, total_cost

    # Obtener puntajes cuantitativos
    quant_scores, quant_cost = await get_quantitative_scores(context_text, llm)
    final_analysis.update(quant_scores)
    total_cost += quant_cost
    
    # Definir temas y sus queries de búsqueda
    themes = {
        "inflacion": "inflación INPC subyacente no subyacente objetivo precios trayectoria pronóstico riesgos",
        "tasas_de_interes": "política monetaria tasa de interés objetivo Junta de Gobierno postura decisión comunicado guía",
        "actividad_economica_empleo": "actividad económica PIB holgura mercado laboral empleo desocupación salarios crecimiento inversión consumo"
    }
    
    # ASYNC: Ejecutamos los análisis temáticos en paralelo para este informe
    theme_tasks = []
    for theme, query in themes.items():
        task = analyze_theme_research_grade(theme, query, vector_store, llm, report_filename)
        theme_tasks.append(task)
    
    # Esperar a que todos los análisis temáticos terminen
    theme_results = await asyncio.gather(*theme_tasks)
    
    # Agregar los resultados temáticos al análisis final
    for i, theme in enumerate(themes.keys()):
        summary_key = f"{theme}_summary"
        analysis, cost = theme_results[i]
        final_analysis[summary_key] = analysis
        total_cost += cost
    
    # Agregar metadata adicional
    final_analysis['processing_cost'] = total_cost
    final_analysis['report_filename'] = report_filename
    final_analysis['processing_timestamp'] = datetime.now().isoformat()
    
    print(f"✅ Análisis completo para {report_filename}. Costo: ${total_cost:.4f}")
    return final_analysis, total_cost

async def analyze_all_reports(vector_store, llm, report_filenames=None):
    """
    Función principal para analizar todos los informes de Banxico de forma asíncrona,
    respetando el orden cronológico.
    
    Args:
        vector_store: Vector store con los documentos procesados
        llm: Modelo de lenguaje para el análisis
        report_filenames: Lista opcional de nombres de archivos específicos a analizar.
                         Si es None, obtendrá todos los archivos únicos del vector store.
    
    Returns:
        tuple: (lista de análisis, costo total)
    """
    
    if report_filenames is None:
        # Obtener todos los nombres de archivos únicos del vector store
        print("-> Obteniendo lista de informes del vector store...")
        try:
            # Hacer una búsqueda amplia para obtener todos los documentos
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 1000})
            all_docs = await retriever.ainvoke("Banxico política monetaria informe trimestral")
            
            # Extraer nombres únicos de archivos
            unique_filenames = set()
            for doc in all_docs:
                source_doc = doc.metadata.get('source_document')
                if source_doc:
                    unique_filenames.add(source_doc)
            
            report_filenames = list(unique_filenames)
            print(f"-> Encontrados {len(report_filenames)} informes únicos")
            
        except Exception as e:
            print(f"!! Error obteniendo lista de informes: {e}")
            return [], 0.0
    
    if not report_filenames:
        print("!! No se encontraron informes para analizar")
        return [], 0.0
    
    # ORDENAR CRONOLÓGICAMENTE los informes
    print("-> Ordenando informes cronológicamente...")
    report_filenames_sorted = sorted(report_filenames, key=extract_date_from_filename)
    
    print("-> Orden cronológico de procesamiento:")
    for i, filename in enumerate(report_filenames_sorted, 1):
        date = extract_date_from_filename(filename)
        print(f"   {i:2d}. {filename} -> {date.strftime('%Y-%m')}")
    
    # Procesar los informes de forma SECUENCIAL para mantener el orden cronológico
    # (pero cada informe individual usa análisis asíncrono internamente)
    all_analyses = []
    total_cost = 0.0
    
    print(f"\n-> Iniciando análisis secuencial de {len(report_filenames_sorted)} informes...")
    
    for i, filename in enumerate(report_filenames_sorted, 1):
        print(f"\n=== PROCESANDO INFORME {i}/{len(report_filenames_sorted)}: {filename} ===")
        
        try:
            analysis, cost = await get_report_analysis(filename, vector_store, llm)
            
            if analysis is not None:
                # Agregar información de orden cronológico
                analysis['chronological_order'] = i
                analysis['total_reports'] = len(report_filenames_sorted)
                
                all_analyses.append(analysis)
                total_cost += cost
                
                print(f"✅ Completado {filename} (#{i}/{len(report_filenames_sorted)}) - Costo: ${cost:.4f}")
            else:
                print(f"❌ Falló el análisis de {filename}")
                
        except Exception as e:
            print(f"❌ Error procesando {filename}: {e}")
            continue
    
    print(f"\n🎯 ANÁLISIS COMPLETO:")
    print(f"   • Informes procesados: {len(all_analyses)}/{len(report_filenames_sorted)}")
    print(f"   • Costo total: ${total_cost:.4f}")
    
    return all_analyses, total_cost

# Función de conveniencia para análisis de un solo informe
async def analyze_single_report(report_filename: str, vector_store, llm):
    """
    Conveniencia para analizar un solo informe específico.
    
    Args:
        report_filename: Nombre del archivo a analizar
        vector_store: Vector store con los documentos
        llm: Modelo de lenguaje
    
    Returns:
        tuple: (análisis, costo)
    """
    print(f"-> Análisis individual de: {report_filename}")
    return await get_report_analysis(report_filename, vector_store, llm)

# Función para guardar resultados
def save_analysis_results(analyses: list, output_filename: str = "banxico_analysis_results.json"):
    """
    Guarda los resultados de análisis en un archivo JSON.
    
    Args:
        analyses: Lista de análisis generados
        output_filename: Nombre del archivo de salida
    """
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(analyses, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Resultados guardados en: {output_filename}")
        print(f"   • Total de análisis: {len(analyses)}")
        
    except Exception as e:
        print(f"❌ Error guardando resultados: {e}")

# Ejemplo de uso
if __name__ == "__main__":
    # Este código es solo para referencia, no se ejecutará automáticamente
    print("Módulo analyzer.py cargado correctamente.")
    print("Funciones disponibles:")
    print("  - analyze_all_reports(vector_store, llm)")
    print("  - analyze_single_report(filename, vector_store, llm)")
    print("  - save_analysis_results(analyses, filename)")