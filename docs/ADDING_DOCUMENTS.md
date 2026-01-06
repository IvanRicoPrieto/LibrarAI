# 📚 Añadir Nuevos Documentos al RAG

Esta guía explica cómo añadir nuevos libros o papers al sistema LibrarAI.

---

## 📋 Requisitos Previos

### Formato de Documentos

El sistema acepta documentos en **formato Markdown (.md)**. Si tienes PDFs, primero debes convertirlos.

**Herramientas recomendadas para convertir PDF → Markdown:**

| Herramienta     | Uso                         | Calidad    |
| --------------- | --------------------------- | ---------- |
| **marker-pdf**  | `marker_single archivo.pdf` | ⭐⭐⭐⭐⭐ |
| **pymupdf4llm** | Python script               | ⭐⭐⭐⭐   |
| **pandoc**      | `pandoc -f pdf -t markdown` | ⭐⭐⭐     |
| **pdf2md**      | Web/CLI                     | ⭐⭐⭐     |

**Recomendación:** Usa **marker-pdf** para obtener la mejor calidad de conversión, especialmente con fórmulas matemáticas.

```bash
# Instalar marker-pdf
pip install marker-pdf

# Convertir un PDF
marker_single libro.pdf --output_dir ./output/
```

---

## 📁 Estructura de Directorios

```
LibrarAI/
└── data/
    ├── books/                        # ← Libros organizados por temática
    │   ├── computacion_cuantica/
    │   │   ├── nielsen_chuang/
    │   │   │   └── nielsen_chuang.md
    │   │   └── yanofsky_mannucci/
    │   │       └── quantum_computing.md
    │   ├── mecanica_cuantica/
    │   │   ├── sakurai/
    │   │   │   └── modern_qm.md
    │   │   └── ballentine/
    │   │       └── ballentine.md
    │   ├── estructuras_algebraicas/
    │   ├── topologia/
    │   ├── geometrias_lineales/
    │   ├── espacios_de_hilbert/
    │   ├── teoria_informacion/
    │   └── ...
    │
    └── papers/                       # ← Papers organizados por temática
        ├── qkd/
        │   ├── bb84_original/
        │   │   └── bb84.md
        │   └── e91_protocol/
        │       └── e91.md
        ├── computacion_neuromorfica/
        │   └── loihi/
        │       └── loihi.md
        └── ...
```

### Crear nueva categoría temática

Si tu libro/paper no encaja en las categorías existentes, crea una nueva:

```bash
# Para libros
mkdir -p data/books/nueva_categoria/

# Para papers
mkdir -p data/papers/nueva_categoria/
```

**Categorías sugeridas para matemáticas:**

- `algebra_lineal/`
- `analisis_matematico/`
- `ecuaciones_diferenciales/`
- `probabilidad_estadistica/`
- `teoria_numeros/`

---

## ➕ Añadir un Libro

### Paso 1: Identificar o crear la categoría temática

```bash
cd LibrarAI/data/books/
# Ver categorías existentes
ls -la

# Si necesitas crear una nueva
mkdir algebra_lineal
```

### Paso 2: Crear carpeta para el libro

```bash
cd LibrarAI/data/books/categoria_tematica/
mkdir nombre_libro
```

**Convención de nombres:**

- Usa snake_case (minúsculas con guiones bajos)
- Preferiblemente: `autor_titulo_corto` o `titulo_corto`
- Ejemplos: `nielsen_chuang`, `sakurai_qm`, `rieffel_quantum_computing`

### Paso 2: Colocar el Markdown

```bash
cp /ruta/al/libro_convertido.md LibrarAI/data/books/nombre_libro/
```

**Opcionalmente**, puedes incluir:

- Imágenes en subcarpeta `images/`
- Metadatos en archivo `metadata.yaml`

### Paso 3: (Opcional) Añadir metadatos

Crea `metadata.yaml` en la carpeta del libro:

```yaml
title: "Quantum Computation and Quantum Information"
authors:
  - Michael A. Nielsen
  - Isaac L. Chuang
year: 2010
edition: "10th Anniversary Edition"
isbn: "978-1107002173"
topics:
  - quantum computing
  - quantum information
  - quantum algorithms
  - quantum error correction
```

---

## ➕ Añadir un Paper

### Paso 1: Identificar o crear la categoría temática

```bash
cd LibrarAI/data/papers/
# Ver categorías existentes
ls -la

# Crear nueva categoría si es necesario
mkdir teoria_cuerdas
```

### Paso 2: Crear carpeta para el paper

```bash
cd LibrarAI/data/papers/categoria_tematica/
mkdir nombre_paper
```

**Convención de nombres:**

- `autor_año_tema` o `acronimo_descripcion`
- Ejemplos: `bennett_1984_bb84`, `shor_1994_factoring`, `e91_protocol`

### Paso 2: Colocar el Markdown

```bash
cp /ruta/al/paper_convertido.md LibrarAI/data/papers/nombre_paper/
```

### Paso 3: (Opcional) Añadir metadatos

Crea `metadata.yaml`:

```yaml
title: "Quantum Cryptography: Public Key Distribution and Coin Tossing"
authors:
  - Charles H. Bennett
  - Gilles Brassard
year: 1984
venue: "IEEE International Conference on Computers, Systems and Signal Processing"
doi: null
arxiv: null
topics:
  - quantum cryptography
  - QKD
  - BB84
```

---

## 🔄 Re-indexar la Biblioteca

Después de añadir documentos, **debes re-indexar** para que el RAG los reconozca.

### Opción A: Indexación completa (recomendada para pocos documentos nuevos)

```bash
cd LibrarAI/
python -m src.cli.ingest_library
```

### Opción B: Indexación forzada (reconstruye todo desde cero)

```bash
python -m src.cli.ingest_library --force
```

### Opción C: Solo libros o solo papers

```bash
# Solo libros
python -m src.cli.ingest_library --source books

# Solo papers
python -m src.cli.ingest_library --source papers
```

---

## ✅ Verificar la Indexación

1. **Comprobar que se procesaron los documentos:**

```bash
python -m src.cli.ingest_library --dry-run
```

2. **Hacer una consulta de prueba:**

```bash
python -m src.cli.ask_library "Tema del nuevo documento"
```

3. **Ver las fuentes encontradas:**

En modo interactivo, usa `/sources` después de una consulta:

```bash
python -m src.cli.ask_library -i
❓ Tu pregunta: [tema del documento]
/sources
```

---

## 📝 Mejores Prácticas

### Estructura del Markdown

El sistema funciona mejor con Markdown bien estructurado:

```markdown
# Título del Documento

## Capítulo 1: Introducción

### 1.1 Conceptos básicos

Contenido aquí...

### 1.2 Notación

Usamos la notación de Dirac: $|ψ⟩$

## Capítulo 2: Desarrollo

...
```

**Tips:**

- ✅ Usa headers jerárquicos (H1 → H2 → H3)
- ✅ Mantén las fórmulas matemáticas en LaTeX
- ✅ Incluye referencias cruzadas si las hay
- ❌ Evita headers vacíos
- ❌ Evita saltar niveles (H1 → H3)

### Optimización de Contenido

1. **Elimina contenido no útil:**

   - Índices repetitivos
   - Páginas de copyright
   - Ejercicios sin solución (a menos que sean relevantes)

2. **Mantén ecuaciones importantes:**

   - El sistema preserva bloques de código y LaTeX
   - Las fórmulas se indexan junto con su contexto

3. **Divide documentos muy grandes:**
   - Si un libro tiene >500 páginas, considera dividirlo por capítulos
   - Cada archivo puede estar en la misma carpeta

---

## 🔧 Resolución de Problemas

### El documento no aparece en búsquedas

1. Verifica que está en la carpeta correcta (`data/books/` o `data/papers/`)
2. Ejecuta `--force` para re-indexar completamente
3. Comprueba que el archivo tiene extensión `.md`

### Errores de parsing

1. Verifica que el Markdown es válido
2. Comprueba que no hay caracteres especiales problemáticos
3. Revisa los logs en `logs/`

### Fórmulas no se muestran bien

1. Asegúrate de usar sintaxis LaTeX estándar
2. Los bloques `$$...$$` se preservan mejor que inline `$...$`
3. Evita caracteres Unicode que representen símbolos matemáticos

---

## 📊 Costes de Indexación

Añadir documentos tiene un coste en embeddings. Ver coste estimado:

```bash
python -m src.cli.ingest_library --dry-run
```

Ver costes acumulados:

```bash
python -m src.cli.ingest_library --costs
```

El coste depende de:

- Número de chunks generados (aprox. 1 chunk por cada ~2000 caracteres)
- Modelo de embeddings usado (text-embedding-3-large: ~$0.13/1M tokens)

**Estimación típica:**

- 1 libro (~300 páginas): ~1000 chunks ≈ $0.05-0.10
- 1 paper (~20 páginas): ~50 chunks ≈ $0.002-0.005

---

## 🚀 Workflow Completo de Ejemplo

```bash
# 1. Convertir PDF
marker_single ~/Downloads/nuevo_libro.pdf --output_dir ./temp/

# 2. Crear carpeta
mkdir -p LibrarAI/data/books/nuevo_libro/

# 3. Mover archivo
mv ./temp/nuevo_libro/nuevo_libro.md LibrarAI/data/books/nuevo_libro/

# 4. Re-indexar
cd LibrarAI/
python -m src.cli.ingest_library --source books

# 5. Verificar
python -m src.cli.ask_library "concepto del nuevo libro"
```

---

**¿Preguntas?** Consulta el [README principal](../README.md) o abre un issue.
