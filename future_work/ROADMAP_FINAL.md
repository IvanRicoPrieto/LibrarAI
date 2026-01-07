# 🎯 LibrarAI - Roadmap de Trabajo Futuro (Consolidado)

**Fecha:** 6 de enero de 2026  
**Basado en:** Análisis de Claude Opus 4.5 + Gemini  
**Enfoque:** Sistema para uso por agente GitHub Copilot (sin UI gráfica necesaria)

---

## 📋 Contexto de Uso

Este sistema será consumido por un **agente de GitHub Copilot** desde VS Code para:

- Redactar apuntes en formato `.md`
- Consultar bibliografía durante sesiones de escritura
- Generar código y fórmulas fundamentadas en fuentes

**No se requiere:**

- Web UI (Streamlit/Gradio) → El agente usa CLI o API programática
- Visualización interactiva de grafos → El agente trabaja con texto
- Streaming visual elaborado → El agente procesa respuestas completas

**Sí se requiere:**

- Máxima precisión en retrieval
- Citas verificables y trazables
- Integración programática fácil (API/CLI robusto)
- Bajo coste operativo (caché, eficiencia)
- Capacidad de filtrar por dominio/categoría

---

## 🏆 Líneas de Trabajo Ordenadas por Impacto

### Escala de Complejidad

| Puntuación | Significado | Tiempo estimado |
| :--------: | :---------- | :-------------- |
|     ⭐     | Trivial     | < 1 día         |
|    ⭐⭐    | Baja        | 1-3 días        |
|   ⭐⭐⭐   | Media       | 1-2 semanas     |
|  ⭐⭐⭐⭐  | Alta        | 2-4 semanas     |
| ⭐⭐⭐⭐⭐ | Muy Alta    | > 1 mes         |

---

## 🔴 TIER 1: Crítico para Precisión (Implementar Primero)

|  #  | Línea de Trabajo                    | Impacto en Precisión | Mejora que Ofrece                                                                                                                                                                            | Complejidad | Archivos Afectados                          |
| :-: | :---------------------------------- | :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: | :------------------------------------------ |
|  1  | **Re-ranking con Cross-Encoder**    | 🎯🎯🎯🎯🎯           | +15-25% en precisión post-retrieval. RRF es bueno pero un cross-encoder (`ms-marco-MiniLM` o `bge-reranker`) refina eliminando falsos positivos antes de pasar contexto al LLM.              |   ⭐⭐⭐    | `fusion.py`, nuevo `reranker.py`            |
|  2  | **Pipeline de Evaluación (RAGAS)**  | 🎯🎯🎯🎯🎯           | Base objetiva para medir mejoras. Sin métricas (faithfulness, relevancy, context precision) es imposible saber si los cambios mejoran o empeoran el sistema.                                 |   ⭐⭐⭐    | Nuevo `src/evaluation/`                     |
|  3  | **Caché de Embeddings**             | 🎯🎯🎯🎯             | Reduce costes 70-90% en queries repetidas y elimina latencia de API. Crítico para uso intensivo por agente. LRU cache con hash de query.                                                     |   ⭐⭐⭐    | `vector_retriever.py`, nuevo `cache.py`     |
|  4  | **Filtrado por Categoría/Metadata** | 🎯🎯🎯🎯             | Permite queries dirigidas: `"teoría de grupos" --filter categoria:algebra`. Reduce ruido de dominios no relacionados. La estructura de carpetas ya existe, falta exponer filtros en CLI/API. |   ⭐⭐⭐    | `fusion.py`, `ask_library.py`, `indexer.py` |
|  5  | **Qdrant en Docker**                | 🎯🎯🎯🎯             | El sistema actual tiene 125K chunks en modo local (advertencia >20K). Degradación de rendimiento silenciosa. Docker resuelve con persistencia y mejor rendimiento.                           |    ⭐⭐     | `docker-compose.yml`, `settings.yaml`       |

---

## 🟠 TIER 2: Alto Impacto en Calidad de Respuestas

|  #  | Línea de Trabajo                          | Impacto en Precisión | Mejora que Ofrece                                                                                                                                                                  | Complejidad | Archivos Afectados                           |
| :-: | :---------------------------------------- | :------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: | :------------------------------------------- |
|  6  | **Query Expansion (HyDE)**                | 🎯🎯🎯🎯             | Hypothetical Document Embeddings: genera respuesta hipotética y busca con su embedding. Resuelve desajuste de vocabulario pregunta↔documento. Mejora recall en queries abstractas. |   ⭐⭐⭐    | `vector_retriever.py`                        |
|  7  | **Pesos Dinámicos según Query Type**      | 🎯🎯🎯🎯             | El router ya detecta tipo de query pero los pesos son fijos. Query exacta → más BM25. Query conceptual → más Vector. Query relacional → más Graph.                                 |    ⭐⭐     | `router.py`, `fusion.py`                     |
|  8  | **Ampliación de Ontología (Matemáticas)** | 🎯🎯🎯🎯             | `ontology.yaml` solo tiene entidades de cuántica. Faltan: grupos, espacios vectoriales, topología, análisis funcional. El grafo actual pierde relaciones matemáticas.              |   ⭐⭐⭐    | `config/ontology.yaml`, `graph_retriever.py` |
|  9  | **Memoria Conversacional**                | 🎯🎯🎯               | Permite preguntas de seguimiento: "¿Y si cambio X?", "Expande el punto 3". Crítico para sesiones de redacción de apuntes donde el agente itera.                                    |   ⭐⭐⭐    | `ask_library.py`, nuevo `session_manager.py` |
| 10  | **Chunking Semántico Adaptativo**         | 🎯🎯🎯               | Detectar límites naturales (definiciones, teoremas, demostraciones, ejemplos). Actualmente usa tamaños fijos que cortan contenido semántico.                                       |  ⭐⭐⭐⭐   | `chunker.py`                                 |

---

## 🟡 TIER 3: Optimizaciones para Uso Intensivo

|  #  | Línea de Trabajo                       | Impacto en Precisión | Mejora que Ofrece                                                                                                                                               | Complejidad | Archivos Afectados                  |       Estado        |
| :-: | :------------------------------------- | :------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: | :---------------------------------- | :-----------------: |
| 11  | **API REST con FastAPI**               | 🎯🎯🎯               | Desacopla lógica del CLI. Permite que el agente Copilot consuma LibrarAI via HTTP en lugar de shell. Más limpio, mejor manejo de errores, tipado de respuestas. |   ⭐⭐⭐    | Nuevo `src/api/`                    |  ⏭️ Omitido (CLI)   |
| 12  | **Caché Semántica (GPTCache)**         | 🎯🎯🎯               | Si una query es semánticamente equivalente a una anterior (no idéntica), devuelve respuesta cacheada. Reduce costes LLM dramáticamente.                         |   ⭐⭐⭐    | Nuevo `semantic_cache.py`           |    ✅ Completado    |
| 13  | **Indexación Paralela**                | 🎯🎯                 | Actualmente secuencial. Paralelizar embeddings acelera 3-5x. Importante para reindexaciones tras añadir libros.                                                 |   ⭐⭐⭐    | `indexer.py`                        |    ✅ Completado    |
| 14  | **Compresión de Contexto (LLMLingua)** | 🎯🎯                 | Comprime chunks antes de enviar al LLM. Reduce tokens 50-70%. Permite más contexto en el mismo presupuesto de tokens.                                           |   ⭐⭐⭐    | `prompt_builder.py`                 |    ✅ Completado    |
| 15  | **Embeddings Locales con GPU**         | 🎯🎯                 | Elimina dependencia de API OpenAI para embeddings. BGE-M3 o E5-mistral-7b dan calidad comparable. Reduce costes a cero.                                         |   ⭐⭐⭐    | `indexer.py`, `vector_retriever.py` | ⏭️ Omitido (OpenAI) |

---

## 🟢 TIER 4: Mejoras de Robustez y Mantenibilidad

|  #  | Línea de Trabajo                         | Impacto en Precisión | Mejora que Ofrece                                                                                      | Complejidad | Archivos Afectados                 |    Estado     |
| :-: | :--------------------------------------- | :------------------- | :----------------------------------------------------------------------------------------------------- | :---------: | :--------------------------------- | :-----------: |
| 16  | **Tests Unitarios y de Integración**     | 🎯🎯                 | No hay tests. Impide refactoring seguro. Necesario para evolución sostenible.                          |   ⭐⭐⭐    | Nuevo `tests/`                     | ✅ Completado |
| 17  | **Dockerización Completa**               | 🎯🎯                 | `docker-compose` con RAG + Qdrant. Reproducibilidad total.                                             |    ⭐⭐     | `docker-compose.yml`, `Dockerfile` |               |
| 18  | **Logging Estructurado (OpenTelemetry)** | 🎯                   | Tracing para debugging. Útil cuando el agente reporta respuestas pobres y hay que diagnosticar.        |    ⭐⭐     | Todos los módulos                  |               |
| 19  | **Ampliar Whitelist del Sandbox**        | 🎯                   | Faltan: `networkx`, `scikit-learn`, `pennylane`, `cirq`. Limita cálculos que el agente puede ejecutar. |     ⭐      | `sandbox.py`                       |               |
| 20  | **Validación de Código con AST**         | 🎯                   | Análisis estático del código generado. Detecta bucles infinitos potenciales antes de ejecutar.         |    ⭐⭐     | `sandbox.py`                       |               |

---

## 🔵 TIER 5: Experimental / Largo Plazo

|  #  | Línea de Trabajo                         | Impacto en Precisión | Mejora que Ofrece                                                                                                                                             | Complejidad | Archivos Afectados          |
| :-: | :--------------------------------------- | :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------ | :---------: | :-------------------------- |
| 21  | **Indexación "Math-Aware"**              | 🎯🎯🎯🎯             | Parsear fórmulas LaTeX a representación semántica. Buscar "ecuación de onda" encuentra $\Psi(x,t)$. Muy difícil pero alto impacto para biblioteca matemática. | ⭐⭐⭐⭐⭐  | `parser.py`, `chunker.py`   |
| 22  | **GraphRAG con Extracción LLM Completa** | 🎯🎯🎯               | Actualmente solo 10% de chunks usan LLM. Expandir mejora el grafo significativamente pero es costoso.                                                         |  ⭐⭐⭐⭐   | `graph_retriever.py`        |
| 23  | **Agente con Tool Use**                  | 🎯🎯🎯               | LLM decide cuándo buscar más, ejecutar código, o pedir clarificación. Arquitectura agentic completa.                                                          | ⭐⭐⭐⭐⭐  | `agents/`, `ask_library.py` |
| 24  | **Fine-tuning de Embeddings**            | 🎯🎯🎯🎯             | Entrenar adaptador sobre text-embedding-3-large con pares query-chunk del dominio. +10-20% precision pero requiere dataset de evaluación.                     | ⭐⭐⭐⭐⭐  | Nuevo pipeline de training  |
| 25  | **Migración de NetworkX a Neo4j**        | 🎯🎯                 | NetworkX corre en memoria. Neo4j escala mejor y permite consultas Cypher complejas. Solo necesario si el grafo crece mucho.                                   |  ⭐⭐⭐⭐   | `graph_retriever.py`        |

---

## 📊 Matriz de Priorización (Impacto vs Complejidad)

```
                    COMPLEJIDAD
                    Baja ───────────────────► Alta
                    │
     Alto │   [7] Pesos Dinámicos    [1] Re-ranking
          │   [5] Qdrant Docker      [2] RAGAS
          │                          [3] Caché Embeddings
          │                          [4] Filtrado Metadata
   I      │                          [6] HyDE
   M      │
   P      │
   A      ├───────────────────────────────────────────
   C      │
   T      │   [17] Docker Compose    [9] Memoria Conv.
   O      │   [19] Whitelist         [10] Chunking Sem.
          │                          [11] API REST
     Bajo │   [18] Logging           [21] Math-Aware
          │                          [24] Fine-tuning
          │
```

**Quick Wins (Alto impacto, Baja complejidad):**

- #5 Qdrant Docker
- #7 Pesos Dinámicos

**Inversiones Estratégicas (Alto impacto, Alta complejidad):**

- #1 Re-ranking
- #2 RAGAS
- #6 HyDE

---

## 🚀 Plan de Implementación Sugerido

### Sprint 1: Fundamentos (1-2 semanas)

- [ ] #5 Migrar Qdrant a Docker
- [ ] #3 Implementar caché de embeddings
- [ ] #7 Pesos dinámicos en fusion según query type
- [ ] #19 Ampliar whitelist del sandbox

### Sprint 2: Precisión de Retrieval (2-3 semanas)

- [ ] #1 Re-ranking con cross-encoder
- [ ] #4 Filtrado por categoría/metadata
- [ ] #8 Ampliar ontología para matemáticas

### Sprint 3: Evaluación y Calidad (2-3 semanas)

- [ ] #2 Pipeline RAGAS
- [ ] #6 HyDE para expansión de queries
- [ ] #16 Tests básicos

### Sprint 4: Integración Programática (2-3 semanas)

- [ ] #11 API REST con FastAPI
- [ ] #9 Memoria conversacional
- [ ] #17 Docker Compose completo

### Backlog Futuro

- #10 Chunking semántico
- #12 Caché semántica
- #21 Indexación math-aware
- #23 Arquitectura agentic

---

## 💡 Notas para Uso por Agente Copilot

### Patrón de Uso Recomendado

```python
# El agente puede invocar así:
response = ask_library(
    query="Demuestra el teorema de Noether",
    filters={"categoria": "mecanica_cuantica"},
    top_k=8,
    critic=True
)

# Y usar la respuesta para redactar:
apunte = f"""
## Teorema de Noether

{response.content}

### Fuentes
{format_citations(response.sources)}
"""
```

### Integración con Copilot

Una vez implementada la **API REST (#11)**, el agente puede:

1. **Consultar**: `POST /query` con filtros
2. **Verificar citas**: Respuesta incluye `sources` con `chunk_id` y `header_path`
3. **Ejecutar código**: `POST /execute` para cálculos
4. **Contexto conversacional**: `session_id` para continuidad

### Prioridades desde Perspectiva del Agente

1. **Precisión** → Re-ranking + RAGAS + Filtros
2. **Eficiencia** → Caché + Qdrant Docker
3. **Integración** → API REST
4. **Contexto** → Memoria conversacional

---

_Roadmap consolidado a partir de análisis de Claude Opus 4.5 y Gemini, adaptado para consumo por agente GitHub Copilot._
