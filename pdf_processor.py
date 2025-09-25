# pdf_processor.py
import os
import pdfplumber
import pandas as pd
import re

def extract_and_prepare_text(pdf_path: str) -> str:
    """
    Extrae texto de un PDF, manejando el diseño de dos columnas y convirtiendo
    tablas a formato Markdown. Incluye diagnósticos detallados.
    """
    print(f"-> Extrayendo texto de: {os.path.basename(pdf_path)}")
    full_text = ""
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            # DIAGNÓSTICO: Informar si se encontraron páginas.
            print(f"   -> PDF abierto. Número de páginas encontradas: {num_pages}")
            
            if num_pages == 0:
                print("   ⚠️  Advertencia: El PDF no contiene páginas.")
                return ""

            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text(x_tolerance=2, layout=True) or ""
                
                # DIAGNÓSTICO: Mostrar cuántos caracteres se extrajeron por página.
                print(f"   -> Procesando página {page_num + 1}/{num_pages} | Caracteres extraídos: {len(page_text)}")

                if page_text:
                    cleaned_page_text = re.sub(r'\s\s+', ' ', page_text.replace('\r', ''))
                    full_text += f"\n\n--- Page {page_num + 1} ---\n{cleaned_page_text}"

                tables = page.extract_tables()
                if tables:
                    full_text += f"\n\n--- Tables on Page {page_num + 1} ---\n"
                    for table in tables:
                        if not table or not table[0]: continue
                        header = [str(h) if h is not None else '' for h in table[0]]
                        df = pd.DataFrame(table[1:], columns=header)
                        full_text += df.to_markdown(index=False) + "\n\n"
                        
    except Exception as e:
        print(f"   ‼️ Error crítico procesando el archivo {pdf_path}: {e}")
        return ""

    # DIAGNÓSTICO: Informar el tamaño total del texto extraído.
    print(f"   -> Extracción finalizada. Total de caracteres: {len(full_text)}")
    return full_text