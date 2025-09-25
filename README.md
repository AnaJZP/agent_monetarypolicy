# agent_monetarypolicy

Análisis automatizado de **postura de política monetaria** a partir de informes trimestrales de Banxico (RAG + LLM). El pipeline extrae texto de PDFs, crea una base de conocimiento, construye un índice vectorial FAISS y corre un análisis cronológico con métricas cuantitativas y resúmenes temáticos. También incluye módulos de **validación** y **gráficos profesionales**.

## 🗂️ Estructura principal

```
agent_monetarypolicy/
├─ analyzer.py                # Lógica de análisis con LLM (asíncrono, orden cronológico)
├─ banxico_analytics_python.py# Gráficos "fancys" y reportes visuales
├─ config.py                  # Carga config.yaml
├─ config.yaml                # Rutas y parámetros de chunking
├─ main.py                    # Orquestador por fases (1A, 1B, 2, 3, 4)
├─ pdf_processor.py           # Extracción de texto desde PDF (tablas → markdown)
├─ rag_prepper.py             # Split del texto en chunks para RAG
├─ validation.py              # Validación y dashboard de similitud
└─ banxico_informes/          # (coloca aquí tus PDFs)
```

## ⚙️ Requisitos

- Python 3.10+
- Paquetes (sugerencia):
  ```bash
  pip install python-dotenv pyyaml pandas pdfplumber langchain langchain-openai langchain-community faiss-cpu scikit-learn scipy matplotlib seaborn
  ```
- **OpenAI API key** en un archivo `.env`:
  ```env
  OPENAI_API_KEY=tu_api_key
  ```

## 🔧 Configuración (`config.yaml`)

Ajusta rutas y parámetros de chunking. Ejemplo:

```yaml
paths:
  pdf_reports_dir: 'banxico_informes'
  knowledge_base_file: 'banxico_output/knowledge_base_banxico.json'
  vector_store_dir: 'banxico_vectordb'
  analysis_output_csv: 'banxico_output/analisis_postura_banxico.csv'

chunking:
  chunk_size: 1500
  chunk_overlap: 250
```

> `config.py` carga automáticamente este YAML cuando importas `SETTINGS`.

## ▶️ Ejecución rápida

1) Crea carpetas y coloca tus **PDFs** en `banxico_informes/`  
2) Ejecuta el orquestador:
```bash
python main.py
```
El flujo realiza:
- **Fase 1A**: crea un JSON por cada informe (chunks + metadatos)
- **Fase 1B**: consolida la **base de conocimiento**
- **Fase 2**: construye el índice **FAISS**
- **Fase 3**: análisis cronológico (resúmenes, señales, métricas)
- **Fase 4**: síntesis (estadísticos y tendencias)

### Salidas esperadas
- JSON completo del análisis (p. ej. `banxico_output/banxico_analysis.json`)
- CSV resumen (p. ej. `banxico_output/analisis_postura_banxico.csv`)
- Mensajes de consola con **costo total** y **período analizado**

## ✅ Validación y dashboard

Para validar consistencia y generar un dashboard con **similitud coseno** y métricas adicionales:

```bash
python validation.py
```
Genera `enhanced_validation_dashboard.png` con:
- Distribución de posturas
- Evolución del *stance score*
- Matriz y distribución de similitudes
- Resumen ejecutivo de validación

## 📊 Gráficos “fancys” (colores IPN)

Desde `banxico_analytics_python.py` puedes producir PNGs como:
- **Evolución temporal** del *stance score*
- **Radar comparativo** de señales por período (inicial / medio / reciente)
- **Matriz de correlaciones** entre métricas de política

Ejemplo (dentro de ese módulo): llamar funciones `create_stance_evolution_png`, `create_policy_radar_png`, `create_correlation_matrix_png` pasando un `DataFrame` del JSON y un `output_dir`.

## 🧪 Notas útiles

- Los nombres de archivos PDF (e.g., `enero-marzo 2024.pdf`) se usan para inferir **trimestres** y **orden cronológico**.
- Si ya existe el **vector store**, se reutiliza para acelerar la ejecución.
- Si cambias PDFs o parámetros de chunking, regenerar la base de conocimiento y FAISS.

## 🛠️ Troubleshooting

- **`config.yaml` no encontrado** → verifica ubicación y nombre.
- **`OPENAI_API_KEY` vacío** → crea `.env` y reinicia terminal/VS Code.
- **Fallo cargando FAISS** → elimina `banxico_vectordb/` y corre de nuevo Fases 1B–2.
- **PDF sin texto** → confirma que no sea imagen escaneada sin OCR.

## 📜 Licencia

SEPI ESE IPN
---

### 🚀 Push del repo con Visual Studio Code (UI)

1. Abre la carpeta del proyecto en VS Code.  
2. **Source Control** (icono de rama) → *Initialize Repository* (si aún no está).  
3. Escribe un **mensaje de commit** (por ej. “feat: primera versión”) y pulsa **Commit**.  
4. **Publish to GitHub** (botón en la vista de Source Control) o:  
   - Crea el repo vacío en GitHub (sin README si ya lo tienes).
   - En VS Code, **Add Remote** → pega la URL del repo (SSH o HTTPS).
5. Pulsa **Sync/Push** para enviar `main` (o `master`) al remoto.

### 💻 Equivalente en terminal (opcional)

```bash
git init
git add .
git commit -m "feat: primera versión del agente Banxico"
git branch -M main
git remote add origin <URL-DEL-REPO>
git push -u origin main
```

## 📄 .gitignore recomendado

Incluye en el repo un `.gitignore` que excluya:
- Entornos y cachés: `venv/`, `.venv/`, `__pycache__/`, `.pytest_cache/`
- Salidas: `banxico_output/`, `banxico_vectordb/`
- Claves y variables: `.env`
- Datos locales: `*.pdf` (si no deseas versionarlos), `*.csv`, `*.json` generados
```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/

# VS Code
.vscode/

# Datos y salidas
banxico_output/
banxico_vectordb/
*.csv
*.json

# PDFs de entrada (opcional)
banxico_informes/

# Credenciales
.env
```

---

**Créditos**: Pipeline de extracción (`pdf_processor.py`), split por chunks (`rag_prepper.py`), análisis asíncrono y ordenado (`analyzer.py`), orquestación por fases (`main.py`), configuración (`config.py` + `config.yaml`), validación y gráficos (`validation.py`, `banxico_analytics_python.py`).
