# main.py
# VERSIÓN MEJORADA CON JSON INDIVIDUAL POR INFORME

import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import asyncio
from datetime import datetime

# Importamos las nuevas funciones asíncronas del analyzer mejorado
from analyzer import analyze_all_reports, save_analysis_results, extract_date_from_filename
from config import SETTINGS
from pdf_processor import extract_and_prepare_text
from rag_prepper import chunk_document

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ----------------------- Utilidades -----------------------

def ensure_parent_dir_exists(filepath: str):
    """Asegura que el directorio padre de un archivo exista."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

def list_pdfs(pdf_dir: str) -> list[Path]:
    """Lista todos los archivos PDF en un directorio."""
    p = Path(pdf_dir)
    if not p.is_dir():
        print(f"⚠️  No existe el directorio de PDFs: {p.resolve()}")
        return []
    return sorted(p.glob("*.pdf"))

def get_report_metadata(pdf_path: Path):
    """Extrae metadata detallada del informe basada en el nombre del archivo."""
    filename = pdf_path.name
    date = extract_date_from_filename(filename)
    
    # Determinar trimestre
    month = date.month
    if month <= 3:
        quarter = "Q1"
        quarter_name = "enero-marzo"
    elif month <= 6:
        quarter = "Q2" 
        quarter_name = "abril-junio"
    elif month <= 9:
        quarter = "Q3"
        quarter_name = "julio-septiembre"
    else:
        quarter = "Q4"
        quarter_name = "octubre-diciembre"
    
    return {
        "filename": filename,
        "date": date.isoformat(),
        "year": date.year,
        "quarter": quarter,
        "quarter_name": quarter_name,
        "sort_date": date.strftime("%Y-%m")
    }

# ----------------------- FASE 1A: Crear JSON Individual por Informe -----------------------

def process_single_pdf_to_json(pdf_path: Path, individual_reports_dir: str, chunk_settings: dict):
    """
    Procesa un PDF individual y crea su JSON correspondiente.
    Retorna True si se procesó correctamente, False si ya existía.
    """
    metadata = get_report_metadata(pdf_path)
    
    # Crear nombre de archivo JSON basado en el PDF
    json_filename = f"{pdf_path.stem}.json"
    json_path = Path(individual_reports_dir) / json_filename
    
    # Si ya existe el JSON, saltamos el procesamiento
    if json_path.exists():
        return False, metadata
    
    print(f"   -> Procesando: {pdf_path.name}")
    
    try:
        # Extraer texto del PDF
        text = extract_and_prepare_text(str(pdf_path))
        if not text.strip():
            print(f"      ⚠️  No se extrajo texto de {pdf_path.name}")
            return False, metadata
        
        # Crear chunks
        chunks = chunk_document(
            text, 
            chunk_size=chunk_settings['chunk_size'], 
            chunk_overlap=chunk_settings['chunk_overlap']
        )
        
        # Crear estructura JSON del informe individual
        report_data = {
            "metadata": metadata,
            "processing_info": {
                "processed_at": datetime.now().isoformat(),
                "total_chunks": len(chunks),
                "original_text_length": len(text),
                "chunk_settings": chunk_settings
            },
            "chunks": []
        }
        
        # Agregar cada chunk con su metadata
        for i, chunk_content in enumerate(chunks):
            chunk_data = {
                "chunk_id": f"{pdf_path.stem}_chunk_{i:03d}",
                "chunk_index": i,
                "content": chunk_content,
                "content_length": len(chunk_content)
            }
            report_data["chunks"].append(chunk_data)
        
        # Guardar JSON individual
        ensure_parent_dir_exists(str(json_path))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"      ✅ Creado: {json_filename} ({len(chunks)} chunks)")
        return True, metadata
        
    except Exception as e:
        print(f"      ❌ Error procesando {pdf_path.name}: {e}")
        return False, metadata

def run_phase_1a_individual_jsons(pdf_dir: str, individual_reports_dir: str, chunk_settings: dict):
    """
    Procesa todos los PDFs a JSONs individuales en orden cronológico.
    """
    print("\n--- FASE 1A: CREAR JSONS INDIVIDUALES POR INFORME ---")
    
    pdf_files = list_pdfs(pdf_dir)
    if not pdf_files:
        print(f"❌ No se encontraron PDFs en '{pdf_dir}'")
        return []

    # Ordenar cronológicamente
    pdf_files_sorted = sorted(pdf_files, key=lambda x: extract_date_from_filename(x.name))
    
    print(f"-> Encontrados {len(pdf_files_sorted)} informes para procesar:")
    
    processed_reports = []
    new_reports = 0
    
    for i, pdf_path in enumerate(pdf_files_sorted, 1):
        metadata = get_report_metadata(pdf_path)
        print(f"   {i:2d}. {pdf_path.name} -> {metadata['quarter']} {metadata['year']}")
        
        was_processed, report_metadata = process_single_pdf_to_json(
            pdf_path, individual_reports_dir, chunk_settings
        )
        
        if was_processed:
            new_reports += 1
            
        processed_reports.append(report_metadata)
    
    print(f"✅ Fase 1A completada:")
    print(f"   • Total de informes: {len(processed_reports)}")
    print(f"   • Nuevos procesados: {new_reports}")
    print(f"   • Ya existían: {len(processed_reports) - new_reports}")
    
    return processed_reports

# ----------------------- FASE 1B: Consolidar Base de Conocimiento -----------------------

def run_phase_1b_consolidate_kb(individual_reports_dir: str, knowledge_base_path: str):
    """
    Consolida todos los JSONs individuales en una base de conocimiento unificada.
    """
    print("\n--- FASE 1B: CONSOLIDAR BASE DE CONOCIMIENTO ---")
    
    individual_dir = Path(individual_reports_dir)
    if not individual_dir.exists():
        print(f"❌ Directorio de informes individuales no existe: {individual_dir}")
        return
    
    # Buscar todos los archivos JSON
    json_files = sorted(individual_dir.glob("*.json"))
    if not json_files:
        print(f"❌ No se encontraron archivos JSON en {individual_dir}")
        return
    
    print(f"-> Consolidando {len(json_files)} informes individuales...")
    
    consolidated_chunks = []
    report_summaries = []
    
    # Ordenar archivos cronológicamente
    json_files_sorted = sorted(json_files, key=lambda x: extract_date_from_filename(x.name))
    
    for json_file in json_files_sorted:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                report_data = json.load(f)
            
            metadata = report_data["metadata"]
            chunks = report_data["chunks"]
            
            print(f"   -> {metadata['filename']}: {len(chunks)} chunks")
            
            # Agregar chunks a la base consolidada
            for chunk in chunks:
                consolidated_chunk = {
                    "source_document": metadata["filename"],
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"],
                    "report_metadata": metadata  # Incluir toda la metadata del informe
                }
                consolidated_chunks.append(consolidated_chunk)
            
            # Resumen del informe para tracking
            report_summary = {
                "filename": metadata["filename"],
                "date": metadata["date"],
                "year": metadata["year"],
                "quarter": metadata["quarter"],
                "total_chunks": len(chunks),
                "processed_at": report_data["processing_info"]["processed_at"]
            }
            report_summaries.append(report_summary)
            
        except Exception as e:
            print(f"   ❌ Error consolidando {json_file.name}: {e}")
            continue
    
    # Crear la base de conocimiento consolidada
    knowledge_base = {
        "creation_info": {
            "created_at": datetime.now().isoformat(),
            "total_reports": len(report_summaries),
            "total_chunks": len(consolidated_chunks),
            "date_range": {
                "from": min(r["date"] for r in report_summaries) if report_summaries else None,
                "to": max(r["date"] for r in report_summaries) if report_summaries else None
            }
        },
        "report_summaries": report_summaries,
        "chunks": consolidated_chunks
    }
    
    # Guardar base de conocimiento consolidada
    ensure_parent_dir_exists(knowledge_base_path)
    with open(knowledge_base_path, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Base de conocimiento consolidada:")
    print(f"   • Archivo: {knowledge_base_path}")
    print(f"   • Informes: {len(report_summaries)}")
    print(f"   • Chunks totales: {len(consolidated_chunks)}")
    
    if report_summaries:
        date_range = knowledge_base["creation_info"]["date_range"]
        print(f"   • Período: {date_range['from'][:7]} a {date_range['to'][:7]}")

# ----------------------- FASE 2: Crear Vector Store (FAISS) -----------------------

def run_phase_2_build_faiss(knowledge_base_path: str, vector_store_path: str, embeddings_model: OpenAIEmbeddings):
    print("\n--- FASE 2: CREACIÓN DE ÍNDICE VECTORIAL (FAISS) ---")
    ensure_parent_dir_exists(Path(vector_store_path) / "index.faiss")
    
    with open(knowledge_base_path, "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)

    chunks = knowledge_base["chunks"]
    if not chunks:
        print("❌ No hay chunks en la base de conocimiento.")
        return

    print(f"-> Creando documentos para {len(chunks)} chunks...")
    documents = []
    
    for chunk in chunks:
        doc_metadata = {
            "source_document": chunk["source_document"],
            "chunk_id": chunk["chunk_id"]
        }
        
        # Agregar metadata del informe si está disponible
        if "report_metadata" in chunk:
            report_meta = chunk["report_metadata"]
            doc_metadata.update({
                "report_date": report_meta.get("date"),
                "report_year": report_meta.get("year"),
                "report_quarter": report_meta.get("quarter"),
                "sort_date": report_meta.get("sort_date")
            })
        
        doc = Document(
            page_content=chunk["content"],
            metadata=doc_metadata
        )
        documents.append(doc)

    batch_size = 256
    print(f"-> Generando embeddings en lotes de {batch_size}...")
    
    try:
        first_batch = documents[:batch_size]
        vector_store = FAISS.from_documents(first_batch, embeddings_model)
        print(f"   -> Lote inicial: {len(first_batch)} documentos")
        
        for i in range(batch_size, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vector_store.add_documents(batch)
            progress = min(i + len(batch), len(documents))
            print(f"   -> Progreso: {progress}/{len(documents)} documentos")

        vector_store.save_local(vector_store_path)
        print(f"✅ Vector store guardado en: '{vector_store_path}'")
        
    except Exception as e:
        print(f"❌ Error creando vector store: {e}")
        raise

# ----------------------- FASE 3: Análisis Cronológico -----------------------

async def run_chronological_analysis(vector_store_path: str, knowledge_base_path: str, embeddings_model: OpenAIEmbeddings, llm: ChatOpenAI, output_paths: dict):
    """
    Ejecuta el análisis cronológico usando la base de conocimiento consolidada.
    """
    print("\n--- FASE 3: ANÁLISIS CRONOLÓGICO DE INFORMES DE BANXICO ---")
    
    for path in output_paths.values():
        ensure_parent_dir_exists(path)

    print("-> Cargando vector store...")
    try:
        vector_store = FAISS.load_local(
            vector_store_path, 
            embeddings_model, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"❌ Error cargando vector store: {e}")
        return None, 0.0
    
    # Obtener lista de informes de la base de conocimiento consolidada
    with open(knowledge_base_path, "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
    
    # Extraer informes únicos de los summaries (ya vienen ordenados cronológicamente)
    report_summaries = knowledge_base.get("report_summaries", [])
    unique_reports = [summary["filename"] for summary in report_summaries]
    
    if not unique_reports:
        print("❌ No hay informes para analizar en la base de conocimiento.")
        return None, 0.0

    print(f"-> Analizando {len(unique_reports)} informes en orden cronológico:")
    for i, filename in enumerate(unique_reports, 1):
        report_meta = next((s for s in report_summaries if s["filename"] == filename), {})
        quarter_info = report_meta.get("quarter", "N/A")
        year_info = report_meta.get("year", "N/A")
        print(f"   {i:2d}. {filename} -> {quarter_info} {year_info}")

    try:
        print("\n-> Iniciando análisis con analyzer mejorado...")
        all_analyses, total_cost = await analyze_all_reports(
            vector_store=vector_store,
            llm=llm,
            report_filenames=unique_reports
        )

        if not all_analyses:
            print("❌ No se generaron análisis.")
            return None, total_cost

        print(f"✅ Análisis completado para {len(all_analyses)} informes")

        # Enriquecer análisis con metadata de informes
        for analysis in all_analyses:
            filename = analysis.get("report_filename")
            report_meta = next((s for s in report_summaries if s["filename"] == filename), {})
            if report_meta:
                analysis["report_year"] = report_meta.get("year")
                analysis["report_quarter"] = report_meta.get("quarter")
                analysis["report_date"] = report_meta.get("date")

        # Guardar resultados
        json_output_path = output_paths.get('json', 'banxico_analysis_complete.json')
        save_analysis_results(all_analyses, json_output_path)

        # CSV con columnas ordenadas
        csv_output_path = output_paths.get('csv', 'banxico_analysis_summary.csv')
        df = pd.DataFrame(all_analyses)
        
        column_order = [
            'report_filename', 'report_year', 'report_quarter', 'chronological_order',
            'monetary_stance', 'stance_score', 'confidence_level',
            'forward_guidance', 'key_policy_signal',
            'preocupacion_inflacion', 'preocupacion_crecimiento', 'fortaleza_empleo',
            'incertidumbre_politica', 'senales_expansivas', 'senales_restrictivas',
            'processing_timestamp', 'processing_cost'
        ]
        
        available_columns = [col for col in column_order if col in df.columns]
        remaining_columns = [col for col in df.columns if col not in available_columns]
        final_column_order = available_columns + remaining_columns
        
        df = df[final_column_order]
        df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ Resultados guardados en:")
        print(f"   • JSON completo: {json_output_path}")
        print(f"   • CSV resumen: {csv_output_path}")

        return df, total_cost

    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        return None, 0.0

# ----------------------- FASE 4: Síntesis -----------------------

def run_synthesis_analysis(df: pd.DataFrame):
    """Genera síntesis de alto nivel."""
    print("\n--- FASE 4: SÍNTESIS DE RESULTADOS ---")
    
    if df is None or df.empty:
        print("-> No hay datos para la síntesis.")
        return
        
    num_reports = len(df)
    
    # Información temporal
    date_range = "N/A"
    if 'report_year' in df.columns and 'report_quarter' in df.columns:
        try:
            years = df['report_year'].dropna()
            if len(years) > 0:
                min_year, max_year = int(years.min()), int(years.max())
                date_range = f"{min_year} - {max_year}"
        except:
            pass

    print(f"\n📊 RESUMEN EJECUTIVO:")
    print(f"   • Período analizado: {date_range}")
    print(f"   • Total de informes: {num_reports}")
    
    # Estadísticas de postura monetaria
    if 'stance_score' in df.columns:
        stance_stats = df['stance_score'].describe()
        print(f"\n🎯 POSTURA MONETARIA:")
        print(f"   • Promedio: {stance_stats['mean']:.2f}")
        print(f"   • Mediana: {stance_stats['50%']:.2f}")
        print(f"   • Rango: {stance_stats['min']:.2f} a {stance_stats['max']:.2f}")
        
        if 'monetary_stance' in df.columns:
            stance_counts = df['monetary_stance'].value_counts()
            print(f"   • Distribución: {dict(stance_counts)}")

    # Evolución temporal
    if all(col in df.columns for col in ['chronological_order', 'stance_score', 'report_year']):
        df_sorted = df.sort_values('chronological_order')
        
        print(f"\n📈 EVOLUCIÓN TEMPORAL:")
        # Últimos 3 vs primeros 3 informes
        if len(df_sorted) >= 6:
            early_avg = df_sorted['stance_score'].head(3).mean()
            recent_avg = df_sorted['stance_score'].tail(3).mean()
            trend = "más restrictiva" if recent_avg > early_avg else "más expansiva"
            
            print(f"   • Postura inicial (primeros 3): {early_avg:.2f}")
            print(f"   • Postura reciente (últimos 3): {recent_avg:.2f}")
            print(f"   • Tendencia: {trend}")
        
        # Por año si hay múltiples años
        if 'report_year' in df.columns:
            yearly_avg = df.groupby('report_year')['stance_score'].mean()
            if len(yearly_avg) > 1:
                print(f"   • Postura promedio por año:")
                for year, avg_score in yearly_avg.items():
                    print(f"     - {int(year)}: {avg_score:.2f}")

    print(f"\n✅ SÍNTESIS COMPLETADA")

# ----------------------- Orquestador Principal -----------------------

async def main():
    """Función principal mejorada con estructura JSON individual."""
    load_dotenv()
    print("🏦 === ANÁLISIS DE POLÍTICA MONETARIA DE BANXICO v2.1 ===")
    print(f"🕒 Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not SETTINGS:
        print("❌ Error: No se pudo cargar config.yaml")
        return

    paths = SETTINGS['paths']
    chunking = SETTINGS['chunking']

    # Crear estructura de directorios mejorada
    individual_reports_dir = paths.get('individual_reports_dir', 'banxico_output/individual_reports')
    
    # Inicializar modelos de IA
    try:
        print("-> Inicializando modelos OpenAI...")
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.1, 
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        print("✅ Modelos inicializados")
        
    except Exception as e:
        print(f"❌ Error inicializando modelos: {e}")
        return

    try:
        # FASE 1A: Crear JSONs individuales
        processed_reports = run_phase_1a_individual_jsons(
            paths['pdf_reports_dir'], 
            individual_reports_dir, 
            chunking
        )

        if not processed_reports:
            print("❌ No se procesaron informes")
            return

        # FASE 1B: Consolidar base de conocimiento
        run_phase_1b_consolidate_kb(
            individual_reports_dir,
            paths['knowledge_base_file']
        )

        # FASE 2: Vector Store
        vector_store_path = Path(paths['vector_store_dir'])
        if not vector_store_path.exists() or not any(vector_store_path.iterdir()):
            run_phase_2_build_faiss(
                paths['knowledge_base_file'], 
                paths['vector_store_dir'], 
                embeddings_model
            )
        else:
            print(f"✅ Vector store ya existe: {paths['vector_store_dir']}")

        # FASES 3 y 4: Análisis y síntesis
        output_paths = {
            'csv': paths.get('analysis_output_csv', 'banxico_analysis.csv'),
            'json': paths.get('analysis_output_json', 'banxico_analysis.json')
        }

        analysis_df, total_cost = await run_chronological_analysis(
            vector_store_path=paths['vector_store_dir'],
            knowledge_base_path=paths['knowledge_base_file'],
            embeddings_model=embeddings_model,
            llm=llm,
            output_paths=output_paths
        )
        
        if analysis_df is not None:
            run_synthesis_analysis(analysis_df)
        
        print(f"\n💰 COSTO TOTAL: ${total_cost:.4f}")
        print(f"🕒 Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n🏆 ¡ANÁLISIS COMPLETADO!")
        
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())