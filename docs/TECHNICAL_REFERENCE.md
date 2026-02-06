# 📚 LibrarAI - Documentación Técnica del Sistema

**Versión:** 2.0
**Última actualización:** 7 de enero de 2026
**Estado:** ✅ Sistema completamente implementado y operativo

---

## 📋 Índice

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Características Implementadas](#2-características-implementadas)
3. [Arquitectura del Sistema](#3-arquitectura-del-sistema)
4. [Stack Tecnológico](#4-stack-tecnológico)
5. [Componentes del Sistema](#5-componentes-del-sistema)
6. [Guía de Uso](#6-guía-de-uso)
7. [Configuración](#7-configuración)
8. [Despliegue con Docker](#8-despliegue-con-docker)
9. [Testing](#9-testing)
10. [Métricas y Observabilidad](#10-métricas-y-observabilidad)
11. [Estructura del Proyecto](#11-estructura-del-proyecto)
12. [Estimación de Costes](#12-estimación-de-costes)
13. [Roadmap y Trabajo Futuro](#13-roadmap-y-trabajo-futuro)

---

## 1. Resumen Ejecutivo

**LibrarAI** es un sistema RAG (Retrieval-Augmented Generation) agéntico diseñado para consultar bibliotecas técnicas de Física, Matemáticas y Computación Cuántica. El sistema permite:

- 🔍 **Recuperar información** relevante de documentos Markdown usando búsqueda híbrida
- 📝 **Generar respuestas** fundamentadas con citas precisas a las fuentes
- 🤖 **Integración con agentes** como Claude Code para redacción asistida
- 💰 **Optimización de costes** mediante múltiples capas de caché

### Estado de Implementación

| Componente             | Estado | Descripción                                |
| ---------------------- | :----: | ------------------------------------------ |
| Búsqueda Híbrida       |   ✅   | Vector + BM25 + GraphRAG con fusión RRF    |
| Re-ranking             |   ✅   | Cross-Encoder con 4 presets de calidad     |
| HyDE                   |   ✅   | Query expansion con documentos hipotéticos |
| Evaluación RAGAS       |   ✅   | Pipeline completo con 6 métricas           |
| Cache Embeddings       |   ✅   | Reduce costes 70-90%                       |
| Cache Semántico        |   ✅   | Respuestas cacheadas por similitud         |
| Compresión Contexto    |   ✅   | Reduce tokens 30-60%                       |
| Memoria Conversacional |   ✅   | Sesiones multi-turno                       |
| Filtrado Metadata      |   ✅   | Por categoría/dominio                      |
| Chunking Semántico     |   ✅   | Detecta teoremas, definiciones, etc.       |
| Code Sandbox           |   ✅   | Ejecución segura con validación AST        |
| Dockerización          |   ✅   | docker-compose con Qdrant + App            |
| Logging Estructurado   |   ✅   | structlog con tracing                      |
| Tests                  |   ✅   | 66 tests pasando                           |

---

## 2. Características Implementadas

### 2.1 Recuperación de Información

| Característica         | Mejora  | Descripción                                            |
| ---------------------- | :-----: | ------------------------------------------------------ |
| **Búsqueda Vectorial** |  Base   | Embeddings OpenAI text-embedding-3-large (3072 dims)   |
| **Búsqueda Léxica**    |  Base   | BM25 para coincidencias exactas                        |
| **GraphRAG**           | +10-15% | Grafo de conocimiento con 18 entidades y 19 relaciones |
| **Fusión RRF**         | +5-10%  | Reciprocal Rank Fusion con k=60                        |
| **Re-ranking**         | +15-25% | Cross-Encoder ms-marco-MiniLM                          |
| **HyDE**               | +10-20% | Hypothetical Document Embeddings                       |
| **Pesos Dinámicos**    | +5-10%  | Adapta pesos según tipo de query                       |
| **Filtrado Metadata**  |    -    | Filtra por categoría/dominio                           |

### 2.2 Optimización de Costes

| Característica          |  Ahorro  | Descripción                                  |
| ----------------------- | :------: | -------------------------------------------- |
| **Cache Embeddings**    |  70-90%  | LRU cache con persistencia en disco          |
| **Cache Semántico**     | 100%/hit | Reutiliza respuestas similares (umbral 0.92) |
| **Compresión Contexto** |  30-60%  | 3 niveles: light/medium/aggressive           |
| **Indexación Paralela** |   3-5x   | Embeddings en paralelo (10 workers)          |

### 2.3 Calidad y Verificación

| Característica          | Descripción                                                                      |
| ----------------------- | -------------------------------------------------------------------------------- |
| **Evaluación RAGAS**    | 6 métricas: faithfulness, relevancy, precision, recall, harmfulness, correctness |
| **Critic**              | Validación de citas antes de entregar respuesta                                  |
| **Chunking Semántico**  | Preserva definiciones, teoremas, demostraciones                                  |
| **Ontología Extendida** | 18 tipos de entidades, 19 relaciones matemáticas/cuánticas                       |

### 2.4 Ejecución de Código

| Característica         | Descripción                                                                |
| ---------------------- | -------------------------------------------------------------------------- |
| **Sandbox Seguro**     | Subprocess aislado o Docker container                                      |
| **Validación AST**     | Detecta bucles infinitos, recursión, código peligroso                      |
| **Whitelist Ampliada** | numpy, scipy, sympy, matplotlib, qutip, pennylane, cirq, sklearn, networkx |
| **Captura de Figuras** | Gráficas matplotlib embebidas en respuesta                                 |

### 2.5 Robustez

| Característica             | Descripción                                  |
| -------------------------- | -------------------------------------------- |
| **Dockerización**          | docker-compose con Qdrant, App y Sandbox     |
| **Logging Estructurado**   | structlog con JSON/console, trace context    |
| **Tests**                  | 66 tests unitarios + integración             |
| **Memoria Conversacional** | Sesiones persistentes, detección de followup |

---

## 3. Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────┐
│                           USUARIO                                   │
│                      (Terminal / VS Code / Agente)                  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        CAPA DE INTERFAZ                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ CLI Parser   │  │ Session      │  │ Output       │               │
│  │ (click)      │  │ Manager      │  │ Formatter    │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        CAPA DE CACHÉ                                │
│  ┌──────────────┐  ┌──────────────┐                                 │
│  │ Semantic     │  │ Embedding    │  ← 100% ahorro si hit           │
│  │ Cache        │  │ Cache        │  ← 70-90% ahorro embeddings     │
│  └──────────────┘  └──────────────┘                                 │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        CAPA AGÉNTICA                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Router       │  │ Planner      │  │ Critic       │               │
│  │ (GPT-4.1-m)  │  │ (Deep Res.)  │  │ (Verify)     │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     CAPA DE RECUPERACIÓN                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Vector       │  │ BM25         │  │ Graph        │               │
│  │ Retriever    │  │ Retriever    │  │ Retriever    │               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
│         │                 │                 │                       │
│         └─────────────────┼─────────────────┘                       │
│                           ▼                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ HyDE         │  │ RRF Fusion   │  │ Re-Ranker    │               │
│  │ (opcional)   │  │ + Auto-Merge │  │ (opcional)   │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CAPA DE GENERACIÓN                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Context      │  │ LLM          │  │ Citation     │               │
│  │ Compressor   │  │ (Claude)     │  │ Injector     │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CAPA DE ALMACENAMIENTO                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Qdrant       │  │ BM25 Index   │  │ Knowledge    │               │
│  │ (Docker)     │  │ (Pickle)     │  │ Graph (NX)   │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Stack Tecnológico

### 4.1 Componentes Principales

| Componente    | Tecnología            | Versión | Justificación                          |
| ------------- | --------------------- | ------- | -------------------------------------- |
| Framework RAG | LlamaIndex            | 0.10+   | Chunking jerárquico, citations nativas |
| Vector DB     | Qdrant                | latest  | Escalable, filtrado metadata, Docker   |
| Índice Léxico | rank_bm25             | 0.2.2   | Ligero, sin servidor                   |
| Grafos        | NetworkX              | 3.2+    | En memoria, suficiente para ~500 docs  |
| Re-ranker     | sentence-transformers | 2.2+    | ms-marco-MiniLM local                  |
| Evaluación    | RAGAS                 | custom  | LLM-as-judge                           |

### 4.2 Modelos de IA

| Uso             | Modelo                 | Provider  | Precio             |
| --------------- | ---------------------- | --------- | ------------------ |
| **Generación**  | Claude Sonnet 4.5      | Anthropic | $3/$15 per 1M      |
| **Alternativa** | GPT-4.1                | OpenAI    | $2.50/$10 per 1M   |
| **Ruteo**       | GPT-4.1-mini           | OpenAI    | $0.15/$0.60 per 1M |
| **Embeddings**  | text-embedding-3-large | OpenAI    | $0.13 per 1M       |

### 4.3 Infraestructura

| Componente   | Tecnología              | Descripción                     |
| ------------ | ----------------------- | ------------------------------- |
| Contenedores | Docker + Compose        | Qdrant, App, Sandbox aislado    |
| Logging      | structlog               | JSON/console, trace correlation |
| Testing      | pytest + pytest-asyncio | 66 tests, fixtures compartidos  |
| Cache        | SQLite + LRU            | Persistente en disco            |

---

## 5. Componentes del Sistema

### 5.1 Ingestion (`src/ingestion/`)

| Archivo      | Función                                            |
| ------------ | -------------------------------------------------- |
| `parser.py`  | Parser Markdown con extracción de header_path      |
| `chunker.py` | Chunking jerárquico (Macro/Meso/Micro) + semántico |
| `indexer.py` | Indexación paralela (10 workers), incremental      |

**Chunking Jerárquico:**

| Nivel | Tamaño      | Contenido              | Uso             |
| ----- | ----------- | ---------------------- | --------------- |
| Macro | 2048 tokens | Sección completa       | Contexto amplio |
| Meso  | 512 tokens  | Párrafos relacionados  | Balance         |
| Micro | 200 tokens  | Definiciones, teoremas | Alta precisión  |

**Chunking Semántico:** Detecta y preserva bloques atómicos:

- Definiciones, Teoremas, Lemas, Corolarios
- Demostraciones, Ejemplos, Algoritmos
- Protocolos, Código, Ecuaciones

### 5.2 Retrieval (`src/retrieval/`)

| Archivo               | Función                                   |
| --------------------- | ----------------------------------------- |
| `vector_retriever.py` | Búsqueda semántica en Qdrant              |
| `bm25_retriever.py`   | Búsqueda léxica exacta                    |
| `graph_retriever.py`  | Traversal del grafo de conocimiento       |
| `fusion.py`           | RRF fusion + Auto-merge + Pesos dinámicos |
| `reranker.py`         | Cross-Encoder re-ranking (4 presets)      |
| `hyde.py`             | HyDE query expansion                      |
| `cache.py`            | Cache de embeddings con LRU               |
| `semantic_cache.py`   | Cache semántico de respuestas             |
| `compressor.py`       | Compresión de contexto (3 niveles)        |

**Presets de Re-ranking:**

| Preset      | top_k | Modelo               | Latencia |
| ----------- | ----- | -------------------- | -------- |
| fast        | 50    | ms-marco-MiniLM-L-4  | ~100ms   |
| balanced    | 100   | ms-marco-MiniLM-L-6  | ~200ms   |
| quality     | 150   | ms-marco-MiniLM-L-12 | ~500ms   |
| max_quality | 200   | cross-encoder/stsb   | ~1s      |

**Niveles de Compresión:**

| Nivel      | Reducción | Caso de uso       |
| ---------- | --------- | ----------------- |
| light      | ~20%      | Preserva detalles |
| medium     | ~40%      | Balance           |
| aggressive | ~60%      | Síntesis amplia   |

### 5.3 Generation (`src/generation/`)

| Archivo                | Función                                   |
| ---------------------- | ----------------------------------------- |
| `prompt_builder.py`    | Construye prompts con contexto comprimido |
| `synthesizer.py`       | Llamada a LLM (Claude/GPT-4.1)            |
| `citation_injector.py` | Inyecta citas `[n]` en respuesta          |

### 5.4 Agents (`src/agents/`)

| Archivo              | Función                                     |
| -------------------- | ------------------------------------------- |
| `router.py`          | Clasifica query y asigna pesos dinámicos    |
| `planner.py`         | Deep Research: descompone queries complejas |
| `critic.py`          | Verifica fidelidad de citas                 |
| `session_manager.py` | Memoria conversacional multi-turno          |

### 5.5 Execution (`src/execution/`)

| Archivo      | Función                           |
| ------------ | --------------------------------- |
| `sandbox.py` | Ejecución segura de código Python |

**Características del Sandbox:**

- Whitelist de 34 módulos (numpy, scipy, sympy, matplotlib, qutip, pennylane, cirq, sklearn, networkx...)
- Validación AST: detecta bucles infinitos, recursión, atributos peligrosos
- Timeout configurable (default 30s)
- Modo Docker para máximo aislamiento

### 5.6 Evaluation (`src/evaluation/`)

| Archivo              | Función                      |
| -------------------- | ---------------------------- |
| `ragas_evaluator.py` | Pipeline de evaluación RAGAS |
| `metrics.py`         | 6 métricas de calidad        |
| `test_suite.py`      | Suite de benchmark           |

**Métricas RAGAS:**

| Métrica            | Descripción                         |
| ------------------ | ----------------------------------- |
| Faithfulness       | ¿Respuesta fiel al contexto?        |
| Answer Relevancy   | ¿Respuesta relevante a la pregunta? |
| Context Precision  | ¿Contexto recuperado es preciso?    |
| Context Recall     | ¿Se recuperó todo lo relevante?     |
| Answer Correctness | ¿Respuesta factualmente correcta?   |
| Harmfulness        | ¿Respuesta potencialmente dañina?   |

### 5.7 Utils (`src/utils/`)

| Archivo             | Función                            |
| ------------------- | ---------------------------------- |
| `logging_config.py` | Logging estructurado con structlog |
| `cost_tracker.py`   | Seguimiento de costes por query    |

### 5.8 Agent API (`src/api/`)

**Nueva API estructurada optimizada para agentes de IA.**

| Archivo              | Función                              |
| -------------------- | ------------------------------------ |
| `agent_interface.py` | 5 modos de operación con output JSON |
| `__init__.py`        | Exports públicos                     |

**Modos de Operación:**

| Modo     | Función                               | Retorna                         |
| -------- | ------------------------------------- | ------------------------------- |
| EXPLORE  | Descubrir contenido disponible        | Árbol de contenido, sugerencias |
| RETRIEVE | Obtener contenido exhaustivo          | Lista completa de chunks        |
| QUERY    | Responder preguntas con citas         | Respuesta + claims + citations  |
| VERIFY   | Verificar afirmaciones contra fuentes | Status + evidencia + confianza  |
| CITE     | Generar citas formateadas             | Citas en APA/IEEE/Chicago/MD    |

**CLI asociado:** `python -m src.cli.librari <comando>`

Ver [CLAUDE.md](../CLAUDE.md) para guía completa de uso.

---

## 6. Guía de Uso

### 6.1 Comandos Básicos

```bash
# Activar entorno
cd "/home/ivan/Computación Cuántica/LibrarAI"
source .venv/bin/activate

# Consulta simple
python -m src.cli.ask_library "¿Qué es el algoritmo de Shor?"

# Consulta con máxima calidad
python -m src.cli.ask_library "Explica BB84" --rerank --hyde --critic

# Deep Research para queries complejas
python -m src.cli.ask_library "Compara BB84 con E91" --deep

# Con compresión para incluir más contexto
python -m src.cli.ask_library "Resumen de protocolos QKD" --compress --top-k 20

# Modo interactivo con memoria
python -m src.cli.ask_library --interactive

# Solo ver fuentes (sin generar respuesta)
python -m src.cli.ask_library "BB84" --sources

# Filtrar por categoría
python -m src.cli.ask_library "Teoría de grupos" --filter categoria:algebra

# Ejecutar código de la respuesta
python -m src.cli.ask_library "Calcula entropía de von Neumann" --exec
```

### 6.2 Parámetros Completos

| Parámetro             | Tipo   | Default  | Descripción                          |
| --------------------- | ------ | -------- | ------------------------------------ |
| `--model`             | choice | claude   | claude, gpt-4.1, gpt-4.1-mini, local |
| `--top-k`             | int    | 10       | Documentos a recuperar               |
| `--deep`              | flag   | false    | Deep Research mode                   |
| `--rerank`            | flag   | false    | Activar re-ranking                   |
| `--rerank-preset`     | choice | balanced | fast, balanced, quality, max_quality |
| `--hyde`              | flag   | false    | HyDE query expansion                 |
| `--compress`          | flag   | false    | Compresión de contexto               |
| `--compress-level`    | choice | medium   | light, medium, aggressive            |
| `--critic`            | flag   | false    | Validación de citas                  |
| `--exec`              | flag   | false    | Permitir ejecución de código         |
| `--filter`            | string | -        | Filtrar metadata (KEY:VALUE)         |
| `--no-cache`          | flag   | false    | Deshabilitar cache embeddings        |
| `--no-semantic-cache` | flag   | false    | Deshabilitar cache semántico         |
| `--json`              | flag   | false    | Salida JSON                          |
| `--verbose`           | flag   | false    | Logging detallado                    |

### 6.3 Evaluación

```bash
# Evaluar query individual
python -m src.cli.evaluate --query "¿Qué es el entrelazamiento?"

# Benchmark completo
python -m src.cli.evaluate --suite default

# Comparar con baseline
python -m src.cli.evaluate --suite default --baseline benchmark_results/baseline.json
```

### 6.4 Indexación

```bash
# Re-indexar todo
python -m src.cli.ingest_library

# Solo documentos nuevos/modificados
python -m src.cli.ingest_library --update

# Ver qué se indexaría
python -m src.cli.ingest_library --dry-run
```

---

## 7. Configuración

### 7.1 Archivo Principal (`config/settings.yaml`)

```yaml
# Embeddings
embedding:
  provider: "openai"
  model: "text-embedding-3-large"
  dimensions: 3072
  batch_size: 100

# Chunking
chunking:
  micro_size: 200
  meso_size: 512
  macro_size: 2048
  overlap: 50

# Retrieval
retrieval:
  vector_top_k: 30
  bm25_top_k: 30
  rrf_k: 60
  fusion_top_k: 10
  final_top_k: 5
  auto_merge_threshold: 0.5
  min_similarity_threshold: 0.65

# Generation
generation:
  provider: "anthropic"
  model: "claude-sonnet-4-5-20250929"
  temperature: 0.3
  max_tokens: 2000

# Routing
routing:
  provider: "openai"
  model: "gpt-4.1-mini"
  temperature: 0.1
```

### 7.2 Variables de Entorno (`.env`)

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
QDRANT_URL=http://localhost:6333  # Si usas Docker
```

### 7.3 Ontología (`config/ontology.yaml`)

18 tipos de entidades organizados en categorías:

- **Computación Cuántica:** Algoritmo, Protocolo, Gate, Hardware
- **Física:** Concepto, Teorema, Autor, Documento
- **Matemáticas:** EstructuraAlgebraica, GrupoEspecifico, EspacioVectorial, Operador
- **Topología:** EspacioTopologico, InvarianteTopologico, ConceptoAnalisis, TeoremaMath
- **Información:** MedidaInformacion, Canal

19 tipos de relaciones:

- Computación: MEJORA, DEPENDE_DE, USA, IMPLEMENTA, CITA, DEFINE, DEMUESTRA, PROPONE, ES_CASO_DE, EQUIVALE_A
- Matemáticas: ACTUA_SOBRE, SUBESPACIO_DE, SUBGRUPO_DE, GENERA, PRESERVA, SE_DESCOMPONE_EN, CARACTERIZA, SATISFACE, REPRESENTA

---

## 8. Despliegue con Docker

### 8.1 Desarrollo (Solo Qdrant)

```bash
# Iniciar Qdrant
docker compose up -d qdrant

# Verificar
curl http://localhost:6333/readiness

# Dashboard: http://localhost:6333/dashboard
```

### 8.2 Producción (Stack Completo)

```bash
# Iniciar todo
docker compose --profile production up -d

# Ver logs
docker compose logs -f librar_ai

# Detener
docker compose down
```

### 8.3 Servicios Disponibles

| Servicio    | Puerto | Descripción               |
| ----------- | ------ | ------------------------- |
| Qdrant REST | 6333   | API vectorial + Dashboard |
| Qdrant gRPC | 6334   | API de alto rendimiento   |
| LibrarAI    | 8000   | App (profile production)  |

### 8.4 Volúmenes

```yaml
volumes:
  qdrant_data: # Datos vectoriales persistentes
  sandbox_code: # Código para sandbox aislado
```

---

## 9. Testing

### 9.1 Ejecutar Tests

```bash
# Todos los tests
pytest tests/ -v

# Por componente
pytest tests/test_compressor.py -v
pytest tests/test_fusion.py -v
pytest tests/test_cache.py -v
pytest tests/test_chunker.py -v

# Tests de integración
pytest tests/test_integration.py -v

# Con cobertura
pytest tests/ --cov=src --cov-report=html

# Excluir tests de integración
pytest tests/ -v -m "not integration"
```

### 9.2 Estructura de Tests

| Archivo             | Tests  | Cobertura               |
| ------------------- | :----: | ----------------------- |
| test_compressor.py  |   21   | Compresión de contexto  |
| test_fusion.py      |   16   | Fusión RRF, re-ranking  |
| test_cache.py       |   16   | Cache embeddings        |
| test_chunker.py     |   11   | Chunking jerárquico     |
| test_integration.py |   10   | Pipeline end-to-end     |
| **Total**           | **74** | (66 passing, 8 skipped) |

---

## 10. Métricas y Observabilidad

### 10.1 Logging Estructurado

```python
from src.utils.logging_config import get_logger, trace_context

logger = get_logger(__name__)

with trace_context(query="¿Qué es BB84?", session_id="abc123"):
    logger.info("processing_query", top_k=5, filters={"domain": "qkd"})
```

**Formato JSON (producción):**

```json
{
  "timestamp": "2026-01-07T14:35:22Z",
  "level": "info",
  "event": "query_processed",
  "trace_id": "abc12345",
  "service": "librar_ai",
  "total_time_ms": 2800,
  "chunks_used": 5,
  "tokens_total": 3500
}
```

### 10.2 Métricas RAG

```python
from src.utils.logging_config import RAGMetrics

metrics = RAGMetrics(trace_id="abc123", query="...")
metrics.embedding_time_ms = 50
metrics.vector_search_time_ms = 120
metrics.generation_time_ms = 2000
metrics.embedding_cache_hit = True
metrics.log()
```

### 10.3 Cost Tracking

Archivo: `logs/cost_tracking.csv`

```csv
timestamp,query,model,tokens_in,tokens_out,cost_usd
2026-01-07T14:35:22,¿Qué es Shor?,claude-sonnet-4-5,3200,480,0.018
```

---

## 11. Estructura del Proyecto

```
LibrarAI/
├── config/
│   ├── settings.yaml          # Configuración principal
│   └── ontology.yaml          # Ontología del dominio
├── data/
│   ├── books/                 # Libros por categoría
│   └── papers/                # Papers por tema
├── docs/
│   ├── CLI_AGENT_MANUAL.md    # Manual para agentes IA
│   └── ADDING_DOCUMENTS.md    # Guía para añadir docs
├── indices/
│   ├── qdrant/                # Base de datos vectorial
│   ├── bm25_index.pkl         # Índice BM25
│   ├── chunks.pkl             # Almacén de chunks
│   └── manifest.json          # Tracking de documentos
├── logs/
│   └── cost_tracking.csv      # Seguimiento de costes
├── outputs/
│   ├── figures/               # Gráficas generadas
│   └── sessions/              # Sesiones persistidas
├── src/
│   ├── agents/                # Router, Planner, Critic
│   ├── cli/                   # ask_library, ingest_library
│   ├── evaluation/            # RAGAS, métricas
│   ├── execution/             # Sandbox de código
│   ├── generation/            # Prompt, Synthesizer
│   ├── ingestion/             # Parser, Chunker, Indexer
│   ├── retrieval/             # Vector, BM25, Graph, Fusion
│   └── utils/                 # Logging, Cost tracking
├── tests/                     # Suite de tests pytest
├── future_work/
│   └── ROADMAP_FINAL.md       # Trabajo futuro
├── docker-compose.yml         # Stack Docker
├── Dockerfile                 # Imagen de producción
├── pytest.ini                 # Configuración tests
├── requirements.txt           # Dependencias Python
└── README.md                  # Guía rápida
```

---

## 12. Estimación de Costes

### 12.1 Setup Inicial (Una vez)

| Concepto                             | Cálculo          |      Coste |
| ------------------------------------ | ---------------- | ---------: | --- |
| Embeddings biblioteca (~375k tokens) | 375k × $0.13/1M  |  **$0.05** |     |
| Extracción grafo                     | ~500k × $0.15/1M |  **$0.08** |     |
| **TOTAL**                            |                  | **~$0.15** |

### 12.2 Coste por Uso

| Tipo Query    |  %  | Tokens In/Out |  Coste |
| ------------- | :-: | ------------- | -----: |
| Simple        | 60% | 3k/500        | $0.016 |
| Compleja      | 30% | 15k/1k        | $0.060 |
| Deep Research | 10% | 25k/1.5k      | $0.098 |

**Promedio: ~$0.04/query**

### 12.3 Ahorro con Caché

| Escenario       | Sin Caché | Con Caché | Ahorro |
| --------------- | --------: | --------: | -----: | --- |
| 100 queries     |     $3.75 |     $1.50 |    60% |     |
| 500 queries/mes |    $18.75 |     $7.50 |    60% |     |

---

## 13. Roadmap y Trabajo Futuro

### 13.1 Completado (TIER 1-4)

✅ Re-ranking con Cross-Encoder
✅ Pipeline RAGAS
✅ Cache de Embeddings
✅ Filtrado por Metadata
✅ Qdrant en Docker
✅ HyDE Query Expansion
✅ Pesos Dinámicos
✅ Ontología Extendida
✅ Memoria Conversacional
✅ Chunking Semántico
✅ Cache Semántico
✅ Indexación Paralela
✅ Compresión de Contexto
✅ Tests Unitarios
✅ Dockerización Completa
✅ Logging Estructurado
✅ Whitelist Sandbox Ampliada
✅ Validación AST

### 13.2 Trabajo Futuro (TIER 5)

|  #  | Mejora                 | Complejidad | Descripción                   |
| :-: | ---------------------- | :---------: | ----------------------------- |
| 21  | Indexación Math-Aware  | ⭐⭐⭐⭐⭐  | Parsear LaTeX semánticamente  |
| 22  | GraphRAG LLM Completo  |  ⭐⭐⭐⭐   | Extracción LLM 100% de chunks |
| 23  | Agente Tool Use        | ⭐⭐⭐⭐⭐  | Arquitectura agentic completa |
| 24  | Fine-tuning Embeddings | ⭐⭐⭐⭐⭐  | Adaptar embeddings al dominio |
| 25  | Neo4j                  |  ⭐⭐⭐⭐   | Migrar grafo a Neo4j          |

Ver detalles en [future_work/ROADMAP_FINAL.md](future_work/ROADMAP_FINAL.md).

---

## 14. Conclusión

LibrarAI es un sistema RAG completo y robusto que combina:

- **Recuperación de alta precisión:** Búsqueda híbrida + re-ranking + HyDE
- **Optimización de costes:** Múltiples capas de caché (70-90% ahorro)
- **Verificación de calidad:** RAGAS + Critic + validación de citas
- **Robustez operacional:** Docker, logging estructurado, tests

El sistema está listo para uso en producción como asistente de investigación para bibliotecas técnicas de Física, Matemáticas y Computación Cuántica.

---

_Última actualización: 7 de enero de 2026_
