# rag_prepper.py
# Responsable de dividir el texto limpio en chunks para el RAG.

from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_document(document_text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Divide un documento de texto largo en fragmentos más pequeños y manejables.

    Args:
        document_text: El texto completo del documento.
        chunk_size: El tamaño máximo de cada chunk.
        chunk_overlap: La superposición de caracteres entre chunks.

    Returns:
        Una lista de cadenas de texto, donde cada cadena es un chunk.
    """
    print("-> Dividiendo el documento en chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        # Usa los argumentos recibidos en lugar de valores fijos
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n--- Page", "\n\n--- Table", "\n\n", "\n", ". ", " "]
    )
    
    chunks = text_splitter.split_text(document_text)
    print(f"-> Documento dividido en {len(chunks)} chunks.")
    return chunks