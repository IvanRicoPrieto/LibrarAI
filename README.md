# � LibrarAI - Tu Biblioteca Inteligente

Sistema RAG (Retrieval-Augmented Generation) para consultar tu biblioteca de Física, Matemáticas y cualquier otra área del conocimiento.

## 📋 Características

- **🔍 Búsqueda híbrida**: Vector (semántica) + BM25 (léxica) + Grafo (relaciones)
- **🎯 Re-ranking**: Cross-Encoder opcional que mejora precisión +15-25%
- **� Evaluación RAGAS**: Pipeline de evaluación con métricas de calidad RAG- **💾 Cache de Embeddings**: Reduce costes 70-90% y elimina latencia en queries repetidas- **�📚 Chunking jerárquico**: 3 niveles (Macro/Meso/Micro) con auto-merge inteligente
- **📝 Citas precisas**: Referencias `[n]` a fuentes específicas con ubicación
- **🤖 Multi-LLM**: Claude Sonnet 4.5, GPT-4.1, modelos locales (Ollama)
- **⚡ Indexación incremental**: Solo procesa documentos nuevos/modificados
- **🕸️ Grafo de conocimiento**: Extracción automática de entidades y relaciones
- **🔬 Deep Research**: Descomposición de queries complejas con búsqueda iterativa
- **✅ Validación de citas**: Critic que verifica que las citas tienen soporte real
- **🖥️ Code Sandbox**: Ejecución segura de código Python para cálculos y gráficas

## 🚀 Instalación

### 1. Clonar/Copiar el proyecto

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI"
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o: .venv\Scripts\activate  # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar API keys

```bash
cp .env.example .env
# Editar .env con tus API keys
```

Contenido de `.env`:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Indexar la biblioteca

```bash
python -m src.cli.ingest_library
```

## 📖 Uso

### Consulta simple

```bash
python -m src.cli.ask_library "¿Qué es el algoritmo de Shor?"
```

### Modo interactivo

```bash
python -m src.cli.ask_library --interactive
```

### Opciones avanzadas

```bash
# Usar GPT-4.1 en lugar de Claude
python -m src.cli.ask_library "Explica BB84" --model gpt-4.1

# Solo ver fuentes (sin generar respuesta, ahorra costes)
python -m src.cli.ask_library "BB84" --sources

# Deep Research para queries complejas
python -m src.cli.ask_library "Compara BB84 con E91" --deep

# Con validación de citas (Critic)
python -m src.cli.ask_library "¿Qué es un qubit?" --critic

# Re-ranking con Cross-Encoder (+15-25% precisión)
python -m src.cli.ask_library "Teorema de No-Clonación" --rerank

# Re-ranking con preset de máxima calidad
python -m src.cli.ask_library "Ecuación de Schrödinger" --rerank --rerank-preset quality

# Ejecutar código de la respuesta
python -m src.cli.ask_library "Calcula entropía de von Neumann" --exec

# Más contexto (15 chunks)
python -m src.cli.ask_library "Compara protocolos QKD" --top-k 15

# Streaming de respuesta
python -m src.cli.ask_library "¿Qué es el entrelazamiento?" --stream

# Guardar sesión
python -m src.cli.ask_library "Deriva la ecuación de Schrödinger" --save

# Salida JSON
python -m src.cli.ask_library "¿Qué es un qubit?" --json
```

### Evaluación de Calidad (RAGAS)

```bash
# Evaluar una query individual
python -m src.cli.evaluate --query "¿Qué es el entrelazamiento cuántico?"

# Ejecutar benchmark completo
python -m src.cli.evaluate --suite default

# Comparar con baseline anterior
python -m src.cli.evaluate --suite default --baseline benchmark_results/baseline.json

# Benchmark sin reranking (para comparación A/B)
python -m src.cli.evaluate --suite default --no-rerank
```

### Cache de Embeddings

```bash
# Ver estadísticas del cache
python -m src.cli.ask_library --cache-stats

# Deshabilitar cache (útil para debugging)
python -m src.cli.ask_library "Pregunta" --no-cache
```

## 🏗️ Arquitectura

```
LibrarAI/
├── config/
│   ├── settings.yaml      # Configuración principal
│   └── ontology.yaml      # Ontología del dominio
├── data/
│   ├── books/             # Libros organizados por temática
│   │   ├── computacion_cuantica/
│   │   ├── computacion_neuromorfica/
│   │   ├── comunicacion_cuantica/
│   │   ├── espacios_de_hilbert/
│   │   ├── estructuras_algebraicas/
│   │   ├── geometrias_lineales/
│   │   ├── informacion_cuantica/
│   │   ├── mecanica_cuantica/
│   │   ├── teoria_informacion/
│   │   └── topologia/
│   └── papers/            # Papers organizados por temática
│       ├── computacion_neuromorfica/
│       └── qkd/
├── docs/                  # Documentación
│   ├── CLI_AGENT_MANUAL.md
│   └── ADDING_DOCUMENTS.md
├── indices/               # Índices generados
│   ├── qdrant/           # Base de datos vectorial
│   ├── bm25_index.pkl    # Índice BM25
│   ├── chunks.pkl        # Almacén de chunks
│   └── manifest.json     # Tracking de documentos
├── logs/                  # Logs y costes
│   └── cost_tracking.csv
├── src/
│   ├── ingestion/        # Parsing y chunking
│   │   ├── parser.py     # Parser de Markdown
│   │   ├── chunker.py    # Chunking jerárquico
│   │   └── indexer.py    # Indexación
│   ├── retrieval/        # Recuperación
│   │   ├── vector_retriever.py
│   │   ├── bm25_retriever.py
│   │   ├── graph_retriever.py
│   │   └── fusion.py     # Fusión híbrida (RRF)
│   ├── generation/       # Generación
│   │   ├── prompt_builder.py
│   │   ├── synthesizer.py
│   │   └── citation_injector.py
│   ├── agents/           # Agentes inteligentes
│   │   ├── router.py     # Routing de queries
│   │   ├── planner.py    # Planificación multi-hop
│   │   └── critic.py     # Crítica de respuestas
│   ├── execution/        # Ejecución de código
│   │   └── sandbox.py    # Sandbox seguro
│   ├── utils/            # Utilidades
│   │   └── cost_tracker.py
│   └── cli/              # Interfaz de línea de comandos
│       ├── ask_library.py
│       └── ingest_library.py
├── outputs/              # Sesiones guardadas
├── .venv/                # Entorno virtual Python
├── .env                  # API keys (no commitear)
└── README.md
```

## ⚙️ Configuración

### settings.yaml

```yaml
embedding:
  provider: openai
  model: text-embedding-3-large
  dimensions: 3072

chunking:
  micro_size: 200
  meso_size: 512
  macro_size: 2048

retrieval:
  vector_top_k: 30
  bm25_top_k: 30
  rrf_k: 60
  final_top_k: 10

generation:
  provider: anthropic
  model: claude-sonnet-4-5-20250929
  temperature: 0.3
  max_tokens: 2000
```

## 📊 Comandos de Indexación

```bash
# Indexación incremental (solo nuevos/modificados)
python -m src.cli.ingest_library

# Reindexar todo
python -m src.cli.ingest_library --force

# Ver estadísticas
python -m src.cli.ingest_library --stats

# Construir grafo de conocimiento
python -m src.cli.ingest_library --build-graph

# Dry run (ver qué se procesaría)
python -m src.cli.ingest_library --dry-run
```

## 💰 Estimación de Costes

### Indexación (una vez)

- **Embeddings**: ~$0.13 / 1M tokens (text-embedding-3-large)
- Biblioteca típica (50 papers): ~$0.50-1.00 total

### Consultas

- **Claude Sonnet 4.5**: $3/1M input, $15/1M output
- **GPT-4.1-mini**: $0.40/1M input, $1.60/1M output
- Consulta típica: $0.01-0.05

## 🔧 Troubleshooting

### Error: "OPENAI_API_KEY no configurada"

```bash
cp .env.example .env
# Editar .env con tu API key
```

### Error: "No se encontraron índices"

```bash
python -m src.cli.ingest_library
```

### Error: "qdrant-client no instalado"

```bash
pip install qdrant-client
```

### Memoria insuficiente

- Reducir `batch_size` en settings.yaml
- Usar embeddings de menor dimensión

## 📚 Añadir documentos

1. Convierte tus PDFs a Markdown (usando herramientas como `marker`, `nougat`, etc.)
2. Crea una subcarpeta temática si no existe:
   - `data/books/tu_tematica/nombre_libro/`
   - `data/papers/tu_tematica/nombre_paper/`
3. Coloca el `.md` y opcionalmente el `.pdf` original
4. Ejecuta `python -m src.cli.ingest_library`

### Estructura recomendada

```
data/books/
├── algebra_lineal/
│   └── mi_libro_algebra/
│       ├── mi_libro_algebra.md
│       └── mi_libro_algebra.pdf  (opcional)
├── analisis_matematico/
├── fisica_clasica/
└── ...
```

## 🤝 Estructura de Markdown recomendada

```markdown
# Título del Documento

## Capítulo 1: Introducción

### 1.1 Conceptos básicos

Contenido...

### 1.2 Fórmulas importantes

La ecuación de Schrödinger:

$$i\hbar\frac{\partial}{\partial t}\Psi = \hat{H}\Psi$$

## Capítulo 2: ...
```

## 📝 Licencia

Uso personal/educativo. Los contenidos de la biblioteca son propiedad de sus respectivos autores.

---

## 📊 Estadísticas de la Biblioteca (Enero 2026)

| Categoría  | Contenido                     | Palabras       |
| ---------- | ----------------------------- | -------------- |
| **Libros** | 33 libros en 10 categorías    | ~5.3M          |
| **Papers** | 10 papers en 2 categorías     | ~160K          |
| **Total**  | 43 documentos, ~56,000 chunks | ~5.5M palabras |

### Desglose por área:

- 🔢 **Estructuras Algebraicas**: 8 libros (~1.6M palabras)
- 📐 **Geometrías Lineales**: 6 libros (~788K palabras)
- 🔵 **Topología**: 3 libros (~554K palabras)
- ⚛️ **Computación Cuántica**: 4 libros (~713K palabras)
- 🌀 **Mecánica Cuántica**: 2 libros (~419K palabras)
- 📡 **Información Cuántica**: 2 libros (~463K palabras)
- 🧠 **Computación Neuromórfica**: 1 libro + 3 papers
- 🔐 **QKD/Criptografía Cuántica**: 7 papers (~129K palabras)

---

**Autor**: Desarrollado para el Máster en Computación Cuántica - UNIR  
**Última actualización**: Enero 2026
