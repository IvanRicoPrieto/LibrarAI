# LibrarAI - Guía para Agentes IA

> **TL;DR**: Usa `python -m src.cli.librari <comando>` para operaciones JSON estructuradas.
> Todos los outputs son JSON parseables directamente.

## Estadísticas de la Biblioteca

```bash
python -m src.cli.librari stats
```

**Output actual**:
- 58,828 chunks indexados (con metadatos de dificultad y términos matemáticos)
- 59 documentos (libros/papers)
- Knowledge graph disponible
- Índices BM25 y vectoriales activos

## API para Agentes (5 Modos)

LibrarAI provee una API JSON optimizada para agentes con 5 modos de operación:

### 1. EXPLORE - Descubrir contenido disponible

Antes de generar apuntes, explora qué contenido existe:

```bash
python -m src.cli.librari explore "algoritmo de Shor"
```

Output JSON:
- `content_tree`: Jerarquía de documentos/secciones relevantes
- `total_documents`: Número de libros que cubren el tema
- `suggested_queries`: Queries sugeridas para profundizar
- `coverage_summary`: Resumen de la cobertura del tema

### 2. RETRIEVE - Obtener contenido exhaustivo

Recupera TODO el contenido relevante (no limitado a top-k):

```bash
python -m src.cli.librari retrieve "transformada de Fourier cuántica" --exhaustive
```

**Filtrar por nivel de dificultad** (adaptar al estudiante):
```bash
# Solo contenido introductorio
python -m src.cli.librari retrieve "qubit" --level introductory

# Contenido de investigación
python -m src.cli.librari retrieve "error correction" --level research

# Combinar niveles
python -m src.cli.librari retrieve "entrelazamiento" --level "introductory,intermediate"
```

**Búsqueda math-aware** (expande términos LaTeX):
```bash
# Encuentra chunks con \otimes, tensor product, Kronecker...
python -m src.cli.librari retrieve "producto tensorial" --math-aware

# Combinar con filtro de nivel
python -m src.cli.librari retrieve "integral" --math-aware --level advanced
```

| Nivel | Descripción |
|-------|-------------|
| `introductory` | Conceptos básicos, definiciones, primeros capítulos |
| `intermediate` | Teoremas, demostraciones simples |
| `advanced` | Matemáticas complejas, proofs rigurosos |
| `research` | Papers, resultados de vanguardia |

Output JSON:
- `chunks`: Lista completa de fragmentos relevantes
- `total_tokens`: Tokens totales del contenido
- `documents_covered`: Libros de donde proviene
- `sections_covered`: Secciones específicas

### 3. QUERY - Responder preguntas con citas

Genera respuestas citadas para preguntas específicas:

```bash
python -m src.cli.librari query "¿Cuál es la complejidad del algoritmo de Shor?" --grounded
```

Output JSON:
- `answer`: Respuesta generada
- `claims`: Lista de afirmaciones con sus citas (`citations: [chunk_ids]`)
- `confidence_score`: Confianza de la respuesta
- `sources_used`: Fuentes utilizadas

### 4. VERIFY - Verificar afirmaciones

Verifica si una afirmación está soportada por las fuentes:

```bash
python -m src.cli.librari verify --claim "El algoritmo de Shor factoriza en tiempo polinómico O(log³n)"
```

Output JSON:
- `status`: `supported`, `contradicted`, `not_found`, `partial`
- `evidence`: Fragmentos que soportan/contradicen
- `confidence`: Nivel de confianza
- `explanation`: Explicación de la verificación

### 5. CITE - Generar citas formateadas

Genera citas formateadas para referencias:

```bash
python -m src.cli.librari cite --chunks "nc_micro_000123,nc_micro_000456" --style apa
```

Output JSON:
- `citations`: Lista de citas formateadas
- `bibliography`: Bibliografía completa si hay múltiples citas

## Flujo de trabajo recomendado para generar apuntes

1. **Explorar el tema**:
   ```bash
   python -m src.cli.librari explore "tema de interés"
   ```
   Esto te da una visión general de qué hay disponible.

2. **Recuperar contenido**:
   ```bash
   python -m src.cli.librari retrieve "tema específico" --exhaustive
   ```
   Obtén todo el contenido relevante para tener contexto completo.

3. **Generar secciones con preguntas**:
   ```bash
   python -m src.cli.librari query "¿Qué es X?" --grounded
   python -m src.cli.librari query "¿Cómo funciona Y?" --grounded
   ```
   Genera respuestas citadas para cada sección de los apuntes.

4. **Verificar afirmaciones clave**:
   ```bash
   python -m src.cli.librari verify --claim "Afirmación importante a verificar"
   ```
   Asegura que las afirmaciones son correctas.

5. **Generar bibliografía**:
   ```bash
   python -m src.cli.librari cite --chunks "id1,id2,id3" --style apa
   ```
   Crea la bibliografía final.

## API Programática (Python)

Para uso más avanzado desde Python:

```python
from pathlib import Path
from src.api import AgentAPI

api = AgentAPI(indices_dir=Path("indices"))

# Explorar
explore = api.explore("entrelazamiento cuántico")
print(explore.to_json())

# Recuperar
retrieve = api.retrieve("Bell states", exhaustive=True)
for chunk in retrieve.chunks:
    print(f"[{chunk.chunk_id}] {chunk.content[:100]}...")

# Query con citas
result = api.query("¿Qué son los estados de Bell?", require_citations=True)
print(result.answer)
for claim in result.claims:
    print(f"  - {claim.claim} [{', '.join(claim.citations)}]")

# Verificar
verify = api.verify("Los estados de Bell violan las desigualdades de Bell")
print(f"Status: {verify.status.value}, Confianza: {verify.confidence}")
```

## Estructura de la biblioteca

La biblioteca contiene 59 documentos con 58,828 fragmentos indexados, incluyendo:
- Nielsen & Chuang - Quantum Computation
- Watrous - Theory of Quantum Information
- Wilde - Quantum Information Theory
- Sakurai - Modern Quantum Mechanics
- Y otros textos de criptografía cuántica, álgebra, topología, finanzas

## CLI Clásico (ask_library)

Para uso interactivo o cuando necesites streaming de respuestas:

```bash
# Pregunta simple con respuesta en streaming
python -m src.cli.ask_library "¿Qué es un qubit?"

# Con técnicas avanzadas
python -m src.cli.ask_library "algoritmo de Grover" --agentic --colbert --multi-query

# Ver opciones
python -m src.cli.ask_library --help
```

**Flags importantes**:
- `--agentic`: RAG iterativo con reformulación automática
- `--colbert`: Reranking con ColBERT
- `--multi-query`: Expansión de query
- `--json`: Output JSON (similar a librari pero con formato diferente)

## Quick Reference para Agentes

| Tarea | Comando |
|-------|---------|
| Ver qué hay sobre un tema | `librari explore "tema"` |
| Obtener todo el contenido | `librari retrieve "tema" --exhaustive` |
| Filtrar por nivel | `librari retrieve "tema" --level introductory` |
| Búsqueda matemática | `librari retrieve "integral" --math-aware` |
| Respuesta citada | `librari query "pregunta" --grounded` |
| Verificar afirmación | `librari verify --claim "afirmación"` |
| Generar cita | `librari cite --chunks "id1,id2" --style apa` |
| Estadísticas | `librari stats` |

## Estructura de chunk_id

Los chunk_ids siguen el formato: `{doc_id}_{level}_{sequence}`

Ejemplos:
- `nielsen_chuang_micro_000123`: Fragmento micro #123 de Nielsen & Chuang
- `watrous_qit_micro_000456`: Fragmento de Watrous - Quantum Information Theory

## Notas importantes

- **Todos los outputs de `librari` son JSON estructurado** para fácil parsing
- **Las citas usan chunk_ids** que pueden trazarse a ubicaciones exactas en los libros
- **Modo exhaustivo** recupera todo el contenido sin límites artificiales
- **Filtrado por nivel** (`--level`) adapta contenido al nivel del estudiante
- **Búsqueda math-aware** (`--math-aware`) expande términos LaTeX automáticamente
- **Verificación** permite validar que los apuntes son fieles a las fuentes
- **Estilos de cita**: apa, ieee, chicago, markdown, inline
- **Directorio de trabajo**: `/home/ivan/Computación Cuántica/LibrarAI/`
- **Directorio de índices**: `indices/`
