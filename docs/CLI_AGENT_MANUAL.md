# 🤖 CLI Agent Manual - LibrarAI

**Manual de uso de la CLI optimizado para agentes de IA (GitHub Copilot, etc.)**

Este documento describe cómo un agente de IA debe interactuar con el sistema LibrarAI a través de la línea de comandos.

---

## 📍 Información del Sistema

| Propiedad                    | Valor                                       |
| ---------------------------- | ------------------------------------------- |
| **Directorio raíz**          | `/home/ivan/Computación Cuántica/LibrarAI/` |
| **Python requerido**         | 3.10+                                       |
| **Entorno virtual**          | `.venv` (OBLIGATORIO activar antes de usar) |
| **Archivo de configuración** | `config/settings.yaml`                      |
| **Archivo de costes**        | `logs/cost_tracking.csv`                    |

---

## ⚡ IMPORTANTE: Activar Entorno Virtual

**SIEMPRE** activar el entorno virtual antes de ejecutar cualquier comando:

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && source .venv/bin/activate
```

O en una sola línea con el comando:

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && source .venv/bin/activate && python -m src.cli.ask_library "PREGUNTA"
```

---

## 🔧 Comandos Disponibles

### 1. Consultar la Biblioteca (`ask_library`)

**Propósito:** Hacer preguntas sobre computación cuántica usando RAG.

#### Sintaxis Básica

```bash
source .venv/bin/activate && python -m src.cli.ask_library "PREGUNTA"
```

#### Parámetros Completos

| Parámetro           | Corto | Tipo   | Default           | Descripción                                               |
| ------------------- | ----- | ------ | ----------------- | --------------------------------------------------------- |
| `query`             | -     | string | -                 | Pregunta a realizar (posicional)                          |
| `--interactive`     | `-i`  | flag   | false             | Modo interactivo                                          |
| `--model`           | `-m`  | choice | claude            | Modelo: `claude`, `gpt-4.1`, `gpt-4.1-mini`, `local`      |
| `--top-k`           | `-k`  | int    | 10                | Documentos a recuperar                                    |
| `--no-sources`      | -     | flag   | false             | Ocultar fuentes                                           |
| `--sources`         | -     | flag   | false             | Solo mostrar fuentes (sin generar respuesta)              |
| `--deep`            | -     | flag   | false             | Deep Research: descompone queries y busca iterativamente  |
| `--stream`          | -     | flag   | false             | Streaming de respuesta                                    |
| `--save`            | `-s`  | flag   | false             | Guardar sesión                                            |
| `--json`            | -     | flag   | false             | Salida JSON                                               |
| `--verbose`         | `-v`  | flag   | false             | Logging detallado                                         |
| `--no-router`       | -     | flag   | false             | Desactivar router                                         |
| `--critic`          | -     | flag   | false             | Activar crítico con validación de citas                   |
| `--exec`            | -     | flag   | false             | Permitir ejecución de código (sandbox seguro)             |
| `--rerank`          | -     | flag   | false             | Aplicar re-ranking con cross-encoder (+15-25% precisión)  |
| `--rerank-preset`   | -     | choice | balanced          | Preset: fast, balanced, quality, max_quality              |
| `--hyde`            | -     | flag   | false             | HyDE: Query expansion para mejorar recall (+10-20%)       |
| `--hyde-domain`     | -     | choice | quantum_computing | Dominio HyDE: quantum_computing, quantum_information, etc |
| `--no-cache`        | -     | flag   | false             | Deshabilitar cache de embeddings                          |
| `--cache-stats`     | -     | flag   | false             | Mostrar estadísticas del cache                            |
| `--filter`          | `-f`  | string | -                 | Filtrar por metadata (KEY:VALUE), repetible               |
| `--list-categories` | -     | flag   | false             | Listar categorías disponibles                             |
| `--costs`           | `-c`  | flag   | false             | Mostrar costes                                            |

#### Ejemplos de Uso para Agentes

**Consulta simple:**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && source .venv/bin/activate && python -m src.cli.ask_library "¿Qué es el algoritmo de Shor?"
```

**Consulta con salida JSON (RECOMENDADO para agentes):**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && source .venv/bin/activate && python -m src.cli.ask_library "¿Cómo funciona BB84?" --json
```

**Consulta con modelo específico:**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && source .venv/bin/activate && python -m src.cli.ask_library "Explica las puertas de Pauli" --model gpt-4.1 --json
```

**Solo ver fuentes (ahorra costes):**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && source .venv/bin/activate && python -m src.cli.ask_library "BB84" --sources
```

**Deep Research para queries complejas:**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && source .venv/bin/activate && python -m src.cli.ask_library "Compara BB84 con E91" --deep --json
```

**Con validación de citas (Critic):**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && source .venv/bin/activate && python -m src.cli.ask_library "¿Qué es un qubit?" --critic
```

**Con HyDE para mejorar recall en queries abstractas:**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && source .venv/bin/activate && python -m src.cli.ask_library "¿Cómo se mantiene la coherencia cuántica?" --hyde --json
```

**HyDE con dominio específico:**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && source .venv/bin/activate && python -m src.cli.ask_library "Seguridad de protocolos de distribución de claves" --hyde --hyde-domain quantum_cryptography --json
```

**Combinar HyDE + Re-ranking (máxima calidad):**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && source .venv/bin/activate && python -m src.cli.ask_library "Deriva la ecuación de Schrödinger" --hyde --rerank --json
```

**Ejecutar código de la respuesta:**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && source .venv/bin/activate && python -m src.cli.ask_library "Calcula entropía de von Neumann" --exec
```

**Consulta con más contexto:**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && source .venv/bin/activate && python -m src.cli.ask_library "Compara los protocolos QKD" --top-k 20 --json
```

**Ver costes acumulados:**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && source .venv/bin/activate && python -m src.cli.ask_library --costs
```

#### Estructura de Respuesta JSON

```json
{
  "query": "¿Qué es el entrelazamiento cuántico?",
  "response": {
    "content": "El entrelazamiento cuántico es...",
    "model": "claude-sonnet-4-5-20250929",
    "tokens_input": 4523,
    "tokens_output": 856,
    "latency_ms": 2341.5,
    "query_type": "conceptual"
  },
  "sources": [
    {
      "doc_title": "Nielsen & Chuang - Quantum Computation",
      "header_path": "Chapter 2 > 2.6 Entanglement",
      "content": "Preview del contenido...",
      "score": 0.005
    }
  ],
  "routing": {
    "strategy": "hybrid",
    "vector_weight": 0.6,
    "bm25_weight": 0.4,
    "graph_weight": 0.0,
    "reasoning": "Query general: estrategia híbrida balanceada"
  }
}
```

#### Routing con Pesos Dinámicos

El router analiza cada query y ajusta automáticamente los pesos de fusión RRF:

| Tipo de Query   | Ejemplo                       | vector | bm25 | graph |
| --------------- | ----------------------------- | ------ | ---- | ----- |
| **Exacta**      | "¿Qué es BB84?"               | 0.3    | 0.6  | 0.1   |
| **Conceptual**  | "Explica el entrelazamiento"  | 0.5    | 0.3  | 0.2   |
| **Relacional**  | "¿Cómo se relaciona X con Y?" | 0.3    | 0.2  | 0.5   |
| **Comparativa** | "Compara BB84 con E91"        | 0.4    | 0.3  | 0.3   |
| **Multi-hop**   | "X y además Y"                | 0.4    | 0.3  | 0.3   |

#### Ontología del Grafo de Conocimiento

El grafo utiliza una ontología ampliada con 18 tipos de entidad y 19 tipos de relación:

**Tipos de Entidad:**

- Computación Cuántica: `Algoritmo`, `Protocolo`, `Gate`, `Hardware`
- Física: `Concepto`, `Teorema`, `Autor`, `Documento`
- Matemáticas: `EstructuraAlgebraica`, `GrupoEspecifico`, `EspacioVectorial`, `Operador`
- Topología: `EspacioTopologico`, `InvarianteTopologico`
- Análisis: `ConceptoAnalisis`, `TeoremaMath`
- Información: `MedidaInformacion`, `Canal`

**Relaciones principales:**

- `DEPENDE_DE`, `USA`, `MEJORA` (algoritmos/protocolos)
- `ACTUA_SOBRE`, `SUBESPACIO_DE`, `SUBGRUPO_DE` (estructuras matemáticas)
- `GENERA`, `PRESERVA`, `SE_DESCOMPONE_EN` (álgebra)
- `CARACTERIZA`, `SATISFACE`, `REPRESENTA` (propiedades)

---

### 2. Indexar Biblioteca (`ingest_library`)

**Propósito:** Procesar y indexar documentos en el sistema RAG.

#### Sintaxis Básica

```bash
python -m src.cli.ingest_library
```

#### Parámetros

| Parámetro   | Corto | Tipo   | Default | Descripción                      |
| ----------- | ----- | ------ | ------- | -------------------------------- |
| `--source`  | `-s`  | choice | all     | Fuente: `books`, `papers`, `all` |
| `--force`   | `-f`  | flag   | false   | Forzar re-indexación             |
| `--dry-run` | `-d`  | flag   | false   | Simular sin ejecutar             |
| `--verbose` | `-v`  | flag   | false   | Logging detallado                |
| `--costs`   | `-c`  | flag   | false   | Mostrar costes                   |

#### Ejemplos para Agentes

**Indexación completa:**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && python -m src.cli.ingest_library
```

**Solo indexar libros:**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && python -m src.cli.ingest_library --source books
```

**Ver qué se indexaría (sin ejecutar):**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && python -m src.cli.ingest_library --dry-run
```

**Re-indexar todo desde cero:**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && python -m src.cli.ingest_library --force
```

**Ver costes de indexación:**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && python -m src.cli.ingest_library --costs
```

---

### 3. Evaluar Calidad RAG (`evaluate`)

**Propósito:** Medir la calidad del sistema RAG con métricas RAGAS (faithfulness, relevancy, precision).

#### Sintaxis Básica

```bash
python -m src.cli.evaluate --query "PREGUNTA"
# o para benchmark completo:
python -m src.cli.evaluate --suite default
```

#### Parámetros

| Parámetro         | Corto | Tipo   | Default           | Descripción                                  |
| ----------------- | ----- | ------ | ----------------- | -------------------------------------------- |
| `--query`         | `-q`  | string | -                 | Query individual a evaluar                   |
| `--ground-truth`  | -     | string | -                 | Respuesta esperada (para recall)             |
| `--suite`         | `-s`  | string | -                 | Suite: `default` o ruta a JSON               |
| `--baseline`      | -     | string | -                 | Ruta a resultados baseline para comparación  |
| `--rerank`        | -     | flag   | true              | Habilitar re-ranking                         |
| `--no-rerank`     | -     | flag   | false             | Deshabilitar re-ranking                      |
| `--rerank-preset` | -     | choice | balanced          | Preset: fast, balanced, quality, max_quality |
| `--eval-model`    | -     | string | gpt-4o-mini       | Modelo para evaluación                       |
| `--output-dir`    | `-o`  | string | benchmark_results | Directorio de salida                         |
| `--verbose`       | `-v`  | flag   | false             | Logging detallado                            |

#### Métricas RAGAS

| Métrica               | Descripción                                            | Rango |
| --------------------- | ------------------------------------------------------ | ----- |
| **Faithfulness**      | ¿La respuesta está basada en el contexto recuperado?   | 0-1   |
| **Answer Relevancy**  | ¿La respuesta aborda la pregunta del usuario?          | 0-1   |
| **Context Precision** | ¿Los chunks recuperados son relevantes para la query?  | 0-1   |
| **Context Recall**    | ¿El contexto contiene info para la respuesta esperada? | 0-1   |

#### Ejemplos para Agentes

**Evaluar query individual:**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && source .venv/bin/activate && python -m src.cli.evaluate --query "¿Qué es el entrelazamiento cuántico?"
```

**Ejecutar benchmark estándar:**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && source .venv/bin/activate && python -m src.cli.evaluate --suite default
```

**Comparar con/sin re-ranking (A/B test):**

```bash
# Con reranking (guardar como baseline)
cd "/home/ivan/Computación Cuántica/LibrarAI" && source .venv/bin/activate && python -m src.cli.evaluate --suite default -o benchmark_results/with_rerank

# Sin reranking (comparar)
cd "/home/ivan/Computación Cuántica/LibrarAI" && source .venv/bin/activate && python -m src.cli.evaluate --suite default --no-rerank --baseline benchmark_results/with_rerank/results_*.json
```

**Benchmark con suite personalizada:**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && source .venv/bin/activate && python -m src.cli.evaluate --suite benchmarks/custom.json
```

#### Estructura de Salida

El comando genera en `benchmark_results/`:

- `report_YYYYMMDD_HHMMSS.md`: Informe legible con métricas agregadas
- `results_YYYYMMDD_HHMMSS.json`: Resultados completos en JSON

---

## 📊 Sistema de Costes

El sistema registra automáticamente todos los costes de API en `logs/cost_tracking.csv`.

### Tipos de Coste

| Tipo    | Descripción             | Operaciones                          |
| ------- | ----------------------- | ------------------------------------ |
| `BUILD` | Construcción del índice | Embeddings de documentos             |
| `QUERY` | Consultas del usuario   | Embeddings de query + Generación LLM |

### Ver Costes

**Costes de consultas:**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && python -m src.cli.ask_library --costs
```

**Costes de indexación:**

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && python -m src.cli.ingest_library --costs
```

### Formato CSV de Costes

```csv
timestamp,usage_type,provider,model,operation,tokens_input,tokens_output,cost_per_1k_input,cost_per_1k_output,total_cost,query
2024-01-15T10:30:00,QUERY,openai,text-embedding-3-large,embedding,256,0,0.00013,0.0,0.000033,¿Qué es BB84?
2024-01-15T10:30:01,QUERY,anthropic,claude-3-5-sonnet-20241022,generation,4500,800,0.003,0.015,0.0255,¿Qué es BB84?
```

---

## 🔄 Workflow Recomendado para Agentes

### Consulta Estándar

```bash
# 1. Cambiar al directorio del proyecto
cd "/home/ivan/Computación Cuántica/LibrarAI"

# 2. Ejecutar consulta con JSON
python -m src.cli.ask_library "PREGUNTA DEL USUARIO" --json
```

### Consulta Compleja (Más Contexto)

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && python -m src.cli.ask_library "PREGUNTA COMPLEJA" --top-k 20 --model claude --json
```

### Indexar Nuevos Documentos

```bash
# 1. Colocar documento en data/books/ o data/papers/
# 2. Re-indexar
cd "/home/ivan/Computación Cuántica/LibrarAI" && python -m src.cli.ingest_library
```

---

## ⚠️ Notas Importantes para Agentes

### 1. Siempre usar rutas absolutas o `cd` al directorio

```bash
# ✅ Correcto
cd "/home/ivan/Computación Cuántica/LibrarAI" && python -m src.cli.ask_library "query"

# ❌ Incorrecto (puede fallar si el CWD no es correcto)
python -m src.cli.ask_library "query"
```

### 2. Escapar comillas en queries

```bash
# ✅ Usar comillas simples si hay comillas dobles
python -m src.cli.ask_library '¿Qué significa "superposición"?'

# ✅ O escapar
python -m src.cli.ask_library "¿Qué significa \"superposición\"?"
```

### 3. Preferir salida JSON

La salida JSON es más fácil de parsear programáticamente:

```bash
python -m src.cli.ask_library "query" --json
```

### 4. Verificar índices antes de consultar

Si hay errores, puede que los índices no existan:

```bash
cd "/home/ivan/Computación Cuántica/LibrarAI" && python -m src.cli.ingest_library --dry-run
```

### 5. Modelos disponibles y sus costes

| Modelo            | Flag                   | Coste Input     | Coste Output    |
| ----------------- | ---------------------- | --------------- | --------------- |
| Claude Sonnet 4.5 | `--model claude`       | $3/1M tokens    | $15/1M tokens   |
| GPT-4.1           | `--model gpt-4.1`      | $2/1M tokens    | $8/1M tokens    |
| GPT-4.1 Mini      | `--model gpt-4.1-mini` | $0.40/1M tokens | $1.60/1M tokens |
| Ollama (local)    | `--model local`        | Gratis          | Gratis          |

---

## 🆕 Nuevas Funcionalidades (Enero 2026)

### Deep Research (`--deep`)

Descompone queries complejas en sub-preguntas, busca iterativamente y sintetiza:

```bash
python -m src.cli.ask_library "Compara BB84 con E91 en seguridad y eficiencia" --deep
```

### Modo Solo Fuentes (`--sources`)

Muestra las fuentes relevantes sin generar respuesta (ahorra costes de API):

```bash
python -m src.cli.ask_library "entrelazamiento cuántico" --sources
```

### Critic con Validación de Citas (`--critic`)

Evalúa la calidad de la respuesta y verifica que cada cita tiene soporte real:

```bash
python -m src.cli.ask_library "qué es un qubit" --critic
```

### Code Sandbox (`--exec`)

Ejecuta código Python de la respuesta en un entorno seguro (permite numpy, scipy, matplotlib):

```bash
python -m src.cli.ask_library "Calcula la entropía de von Neumann para un estado |+⟩" --exec
```

---

## 📖 Ejemplos de Consultas Efectivas

### Conceptuales

```bash
python -m src.cli.ask_library "¿Qué es el entrelazamiento cuántico?" --json
```

### Comparativas

```bash
python -m src.cli.ask_library "Compara BB84 y E91 en términos de seguridad" --top-k 15 --json
```

### Matemáticas

```bash
python -m src.cli.ask_library "¿Cuál es la matriz de la puerta CNOT?" --json
```

### Algorítmicas

```bash
python -m src.cli.ask_library "Explica paso a paso el algoritmo de Grover" --top-k 20 --json
```

### Aplicaciones

```bash
python -m src.cli.ask_library "¿Qué ventajas tiene QKD sobre criptografía clásica?" --json
```

---

## 🔍 Troubleshooting

### Error: "No se encontraron índices"

```bash
# Solución: Ejecutar indexación
cd "/home/ivan/Computación Cuántica/LibrarAI" && python -m src.cli.ingest_library
```

### Error: "API key not found"

```bash
# Verificar que existe .env con las claves
cat "/home/ivan/Computación Cuántica/LibrarAI/.env"
```

### Respuesta vacía o "No encontré información"

```bash
# Verificar con más documentos
python -m src.cli.ask_library "query reformulada" --top-k 30 --json

# O verificar que hay documentos indexados
python -m src.cli.ingest_library --dry-run
```

### Timeout o respuesta lenta

```bash
# Usar modelo más rápido
python -m src.cli.ask_library "query" --model gpt-4o-mini --json
```

---

## 📁 Estructura de Directorios Relevante

```
/home/ivan/Computación Cuántica/LibrarAI/
├── config/
│   └── settings.yaml      # Configuración principal
├── data/
│   ├── books/             # Libros en Markdown
│   └── papers/            # Papers en Markdown
├── indices/               # Índices Qdrant y BM25
├── logs/
│   └── cost_tracking.csv  # Registro de costes
├── outputs/               # Sesiones guardadas
├── src/
│   └── cli/
│       ├── ask_library.py    # CLI de consultas
│       └── ingest_library.py # CLI de indexación
├── .env                   # API keys (no commitear)
└── .env.example           # Plantilla de .env
```

---

**Última actualización:** Enero 2026
