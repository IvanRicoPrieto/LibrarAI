# Motor Matemático de LibrarAI

> Sistema de computación matemática verificada integrado en el pipeline RAG.
> Permite al LLM ejecutar código SymPy/NumPy en sandbox, verificar resultados
> a múltiples niveles de rigor, y trazar cada afirmación hasta sus fuentes.

---

## Tabla de Contenidos

1. [Arquitectura General](#arquitectura-general)
2. [Fase 1 — Loop de Computación Bidireccional](#fase-1--loop-de-computación-bidireccional)
3. [Fase 2 — Verificación Multi-Nivel](#fase-2--verificación-multi-nivel)
4. [Fase 3 — Sistema Multi-Agente + Provenance](#fase-3--sistema-multi-agente--provenance)
5. [Fase 4 — Computación Cuántica](#fase-4--computación-cuántica)
6. [Fase 5 — Knowledge Graph Computacional](#fase-5--knowledge-graph-computacional)
7. [Fase 6 — Verificación Formal (Lean 4)](#fase-6--verificación-formal-lean-4)
8. [Integración en el Pipeline RAG](#integración-en-el-pipeline-rag)
9. [Configuración](#configuración)
10. [Tests](#tests)
11. [Patrones de Diseño](#patrones-de-diseño)
12. [Limitaciones y Restricciones del Sandbox](#limitaciones-y-restricciones-del-sandbox)

---

## Arquitectura General

```
┌─────────────────────────────────────────────────────────────────┐
│                    ResponseSynthesizer                          │
│                  (src/generation/synthesizer.py)                │
│                                                                 │
│  query_type == MATHEMATICAL?                                    │
│       │                                                         │
│       ├── multi_agent.enabled ──► MultiAgentOrchestrator (F3)   │
│       │                              ├── PlannerAgent           │
│       │                              ├── CalculatorAgent        │
│       │                              ├── VerifierAgent          │
│       │                              └── SynthesizerAgent       │
│       │                                                         │
│       └── default ──────────► MathComputationOrchestrator (F1)  │
│                                  └── <COMPUTE> loop             │
└────────────────┬────────────────────────────────────────────────┘
                 │
     ┌───────────┼───────────────────────┐
     ▼           ▼                       ▼
┌───────────┐ ┌──────────────┐  ┌───────────────────┐
│MathEngine │ │Verification  │  │  QuantumEngine    │
│  (F1)     │ │Pipeline (F2) │  │     (F4)          │
│           │ │              │  │                   │
│ SymPy     │ │ L0:Dimensio. │  │ Puertas cuánticas │
│ NumPy     │ │ L1:Numérico  │  │ Conmutadores      │
│ Sandbox   │ │ L2:Simbólico │  │ Entropía von Neum.│
└───────────┘ │ L3:Físico    │  │ Fidelidad         │
              │ L5:Formal    │  └───────────────────┘
              └────┬─────────┘
                   │
     ┌─────────────┼──────────────────┐
     ▼             ▼                  ▼
┌──────────┐ ┌───────────┐  ┌──────────────────┐
│ Wolfram  │ │ Formula   │  │ FormalVerifier   │
│ Client   │ │ Graph (F5)│  │    (F6)          │
│  (F2)    │ │           │  │                  │
│ API ext. │ │Fingerprint│  │ LeanInterface    │
└──────────┘ │E-graph    │  │ Autoformalizator │
             │Rewrite    │  │ Repair loop      │
             └───────────┘  └──────────────────┘
```

### Mapa de Archivos

| Archivo                         | Fase | Descripción                             |
| ------------------------------- | ---- | --------------------------------------- |
| `src/math/engine.py`            | 1    | Motor SymPy en sandbox                  |
| `src/math/orchestrator.py`      | 1    | Loop bidireccional LLM ↔ Sandbox (ToRA) |
| `src/math/verification.py`      | 2    | Pipeline de verificación (5 niveles)    |
| `src/math/artifacts.py`         | 2    | Evidencia computacional (MathArtifact)  |
| `src/math/latex_parser.py`      | 2    | Conversión LaTeX → SymPy con confianza  |
| `src/math/wolfram_client.py`    | 2    | Cliente Wolfram Alpha LLM API           |
| `src/math/agents.py`            | 3    | 4 agentes especializados                |
| `src/math/provenance.py`        | 3    | Grafo W3C PROV de trazabilidad          |
| `src/math/quantum.py`           | 4    | Motor de computación cuántica           |
| `src/math/formula_graph.py`     | 5    | Knowledge graph de fórmulas             |
| `src/math/formula_retriever.py` | 5    | Puente FormulaGraph ↔ retrieval         |
| `src/math/formal_verifier.py`   | 6    | Integración con Lean 4                  |

---

## Fase 1 — Loop de Computación Bidireccional

### Concepto

Implementa el patrón **ToRA** (Tool-integrated Reasoning Agent): el LLM genera
texto con bloques `<COMPUTE>` que contienen código Python/SymPy. El orchestrator
extrae estos bloques, los ejecuta en sandbox, e inyecta los resultados como
`<RESULT>` para que el LLM continúe razonando.

```
LLM: "Para verificar, calculemos..."
     <COMPUTE>
     import sympy as sp
     x = sp.Symbol('x')
     print(sp.integrate(sp.sin(x), x))
     </COMPUTE>

Sandbox ejecuta → stdout: "-cos(x)"

LLM recibe:
     <RESULT>
     -cos(x)
     Execution time: 42ms
     </RESULT>

LLM: "El resultado confirma que ∫sin(x)dx = -cos(x) + C"
```

### `MathEngine` (`engine.py`)

Motor principal de ejecución matemática. Todas las operaciones se ejecutan
en un `CodeSandbox` aislado con timeout.

```python
engine = MathEngine(timeout=30)

# Operaciones disponibles
r = engine.solve("x**2 - 4", "x")              # → [-2, 2]
r = engine.differentiate("x**3", "x")           # → 3*x**2
r = engine.integrate("sin(x)", "x")             # → -cos(x)
r = engine.integrate("x**2", "x", limits=(0,1)) # → 1/3
r = engine.simplify("sin(x)**2 + cos(x)**2")    # → 1
r = engine.verify_equation("sin(2*x)", "2*sin(x)*cos(x)")  # → True
r = engine.matrix_operation("eigenvalues", "[[1,0],[0,-1]]")
r = engine.execute_raw("import sympy as sp\nprint(sp.factorial(10))")
```

**Retorno:** `MathResult` con `success`, `output_expr`, `output_latex`,
`numeric_value`, `execution_time_ms`, `code_executed`, `stdout`, `error`.

### `MathComputationOrchestrator` (`orchestrator.py`)

Coordina el loop iterativo LLM ↔ Sandbox.

```python
orch = MathComputationOrchestrator(engine=engine, max_iterations=5)

response, steps, artifacts = orch.run(
    query="Calcula la integral de sin²(x)",
    retrieval_context="...",
    system_prompt="...",
    user_template="...",
)
```

**Flujo interno:**

1. LLM genera respuesta con `<COMPUTE>` tags
2. `_extract_compute_blocks()` extrae el código (soporta con/sin triple backtick)
3. `engine.execute_raw()` ejecuta en sandbox
4. Resultado inyectado como `<RESULT>` en la conversación
5. Si hay más `<COMPUTE>` → repetir (máx. `max_iterations`)
6. `_clean_response()` limpia la respuesta final para el usuario

**Fallbacks:** Si SymPy falla, intenta Wolfram Alpha (si habilitado).

---

## Fase 2 — Verificación Multi-Nivel

### `VerificationPipeline` (`verification.py`)

Pipeline secuencial de 5 niveles de rigor creciente. Se detiene al primer
nivel que da un resultado definitivo.

| Nivel | Nombre      | Motor  | Descripción                                           |
| ----- | ----------- | ------ | ----------------------------------------------------- |
| 0     | Dimensional | Pint   | Consistencia de unidades (m/s vs kg)                  |
| 1     | Numérico    | NumPy  | Sampling aleatorio (100 puntos, tol. 1e-8)            |
| 2     | Simbólico   | SymPy  | Transforms específicos (expand, factor, trigsimp...)  |
| 3     | Físico      | SymPy  | Invariantes (unitariedad, hermiticidad, traza, norma) |
| 5     | Formal      | Lean 4 | Prueba formal verificada por compilador               |

```python
pipeline = VerificationPipeline(max_level=3)

# Verificar identidad trigonométrica
artifact = pipeline.verify("sin(2*x)", "2*sin(x)*cos(x)")
assert artifact.verification_passed  # True
assert artifact.verification_level == VerificationLevel.SYMBOLIC

# Verificar invariante físico
artifact = pipeline.verify(
    "[[0,1],[1,0]]", "[[0,1],[1,0]]",
    min_level=3, max_level=3,
    context={"invariant_type": "unitary"}
)

# Verificación formal (requiere Lean 4)
artifact = pipeline.verify(
    "sin(x)**2 + cos(x)**2", "1",
    context={
        "formal_statement": "Para todo x, sin²(x) + cos²(x) = 1",
        "llm_fn": my_llm_function,
    }
)
```

**Nivel 2 — Transforms simbólicos (orden de ejecución):**
`expand` → `factor` → `cancel` → `trigsimp` → `together` → `powsimp` → `radsimp` → `logcombine`

> **Nota:** NO se usa `sp.simplify()` genérico porque su comportamiento
> no es determinista entre versiones de SymPy.

**Nivel 3 — Invariantes físicos soportados:**

- `unitary`: U†U = I
- `hermitian`: H = H†
- `trace_one`: Tr(ρ) = 1
- `positive_semidefinite`: autovalores ≥ 0
- `normalized`: ⟨ψ|ψ⟩ = 1

### `MathArtifact` (`artifacts.py`)

Objeto de evidencia computacional que acompaña cada verificación.

```python
artifact = MathArtifact(
    input_sympy="sin(2*x)",
    engine="sympy",
    operation="verification",
    result="True",
    verification_level=VerificationLevel.SYMBOLIC,
    verification_passed=True,
    verification_details={"method": "trigsimp"},
    source_chunks=["nielsen_001"],
)

# Serialización
d = artifact.to_dict()           # Dict completo
block = artifact.to_evidence_block()  # Markdown para respuestas
json_str = json.dumps(d)         # JSON serializable

# Desde MathResult
artifact = MathArtifact.from_math_result(math_result, source_chunks=["..."])
```

**Campos principales:**

- **Input:** `input_latex`, `input_sympy`, `parse_confidence`
- **Computación:** `engine`, `operation`, `code`, `result`, `result_latex`
- **Verificación:** `verification_level`, `verification_passed`, `verification_details`
- **Provenance:** `content_hash` (SHA256), `provenance_chain`, `timestamp`

### `LaTeXParser` (`latex_parser.py`)

Conversión LaTeX → SymPy con puntuación de confianza.

```python
parser = LaTeXParser(use_llm_normalization=False)

sympy_str, confidence = parser.parse(r"\frac{x^2 + 1}{x - 1}")
# → ("(x**2 + 1)/(x - 1)", 0.85)
```

**Estrategia de parsing (en orden):**

1. Parsing determinista con `latex2sympy2`
2. Normalización LLM para LaTeX malformado (opcional)
3. Fallback con regex manual (`\frac{a}{b}` → `(a)/(b)`, etc.)
4. Confianza por round-trip: SymPy → LaTeX → similitud Jaccard con original

### `WolframClient` (`wolfram_client.py`)

Fallback externo cuando SymPy no puede resolver un problema.

```python
client = WolframClient()  # Lee WOLFRAM_MCP_API_KEY de .env

r = client.query("integrate sin(x)^2 dx")
r = client.solve("x^3 - 6x + 4 = 0")
r = client.eigenvalues("{{1,2},{3,4}}")
r = client.step_by_step("derivative of ln(sin(x))")
```

> Se ejecuta FUERA del sandbox (necesita acceso a red).
> Endpoint: `https://api.wolframalpha.com/v1/llm-api`

---

## Fase 3 — Sistema Multi-Agente + Provenance

### Arquitectura Multi-Agente (`agents.py`)

Cuatro agentes especializados coordinados por un orchestrator:

```
┌───────────────────────────────────────┐
│        MultiAgentOrchestrator         │
│                                       │
│  1. PlannerAgent.plan()               │
│     └── DerivationPlan (pasos 1..N)   │
│                                       │
│  2. Para cada paso:                   │
│     ├── CalculatorAgent.compute_step()│
│     └── VerifierAgent.verify_step()   │
│                                       │
│  3. SynthesizerAgent.synthesize()     │
│     └── Respuesta final con LaTeX     │
└───────────────────────────────────────┘
```

#### PlannerAgent

Descompone un problema en 3-8 pasos verificables.

```python
planner = PlannerAgent()
plan = planner.plan(
    query="Demuestra que d/dx[sin(x)] = cos(x)",
    context="Definición de derivada como límite...",
    llm_fn=my_llm,
)
# DerivationPlan(
#   goal="...",
#   strategy="directa",  # directa | inducción | contradicción
#   steps=[DerivationStep(index=0, ...), ...],
#   assumptions=["x real"],
#   required_tools=["sympy"],
# )
```

#### CalculatorAgent

Ejecuta cada paso en el sandbox.

```python
calculator = CalculatorAgent(engine)
result = calculator.compute_step(step, prev_results=[], llm_fn=my_llm)
```

- Si `step.expression` es SymPy válido → ejecución directa
- Si no → LLM genera código → ejecución en sandbox

#### VerifierAgent

Verifica cada paso con VerificationPipeline (niveles 1-2).

```python
verifier = VerifierAgent(engine)
artifact = verifier.verify_step(step, math_result=result)
```

#### SynthesizerAgent

Genera la respuesta final con citas y evidencia.

```python
synthesizer = SynthesizerAgent()
response = synthesizer.synthesize(query, plan, results, llm_fn=my_llm)
```

#### MultiAgentOrchestrator

```python
orch = MultiAgentOrchestrator(max_steps=10)

result = orch.run(
    query="Verifica que sin(2x) = 2sin(x)cos(x)",
    retrieval_context="La identidad del ángulo doble...",
    llm_fn=my_llm,
)

# result = {
#     "response": "...",          # Respuesta final
#     "plan": DerivationPlan,     # Plan de derivación
#     "artifacts": [...],         # Lista de MathArtifact
#     "provenance": {...},        # Grafo W3C PROV
#     "steps_verified": 3,
#     "steps_total": 3,
#     "elapsed_ms": 1234.5,
# }
```

### Provenance W3C PROV (`provenance.py`)

Grafo de trazabilidad que conecta cada afirmación con sus fuentes.

```python
graph = ProvenanceGraph()

# Registrar agentes
llm_id = graph.add_agent(AgentType.LLM, "Claude", "opus-4.6")
sympy_id = graph.add_agent(AgentType.SYMPY, "SymPy", "1.12")

# Registrar entidades y actividades
source_id = graph.add_entity(EntityType.SOURCE_CHUNK, content="...")
compute_id = graph.add_activity(
    ActivityType.COMPUTATION,
    description="Verificar identidad",
    used=[source_id],
    agent_id=sympy_id,
)
result_id = graph.add_entity(EntityType.COMPUTATION_RESULT, content="True")
graph.record_generation(compute_id, result_id)
graph.record_derivation(result_id, source_id)

# Consultar lineage
lineage = graph.get_lineage(result_id)
# → Cadena completa: resultado ← actividad ← fuente
```

**Tipos de entidad:** `SOURCE_CHUNK`, `PARSED_EQUATION`, `COMPUTATION_RESULT`,
`MATH_ARTIFACT`, `LLM_RESPONSE`, `DERIVATION_STEP`, `FINAL_RESPONSE`

**Tipos de actividad:** `RETRIEVAL`, `LATEX_PARSING`, `COMPUTATION`,
`VERIFICATION`, `LLM_REASONING`, `SYNTHESIS`, `PLANNING`

**Tipos de agente:** `LLM`, `SYMPY`, `NUMPY`, `WOLFRAM`, `QUTIP`, `PINT`,
`VERIFIER`, `PLANNER`

---

## Fase 4 — Computación Cuántica

### `QuantumEngine` (`quantum.py`)

Motor especializado para operaciones de mecánica cuántica.

```python
qe = QuantumEngine()

# Puertas cuánticas estándar (10 puertas + QFT)
r = qe.quantum_gate("H")        # Hadamard
r = qe.quantum_gate("X")        # Pauli X
r = qe.quantum_gate("CNOT")     # Controlled-NOT
r = qe.quantum_gate("QFT", {"n": 3})  # QFT de 3 qubits
r = qe.quantum_gate("Rx", {"theta": "pi/4"})  # Rotación

# Álgebra de operadores
r = qe.commutator("[[0,1],[1,0]]", "[[0,-1j],[1j,0]]")
# [X, Y] = 2iZ
r = qe.anticommutator("[[0,1],[1,0]]", "[[0,-1j],[1j,0]]")
# {X, Y} = 0

# Producto tensorial y traza parcial
r = qe.tensor_product("[[0,1],[1,0]]", "[[1,0],[0,-1]]")  # X ⊗ Z
r = qe.partial_trace("...", trace_out=1, dims=[2, 2])

# Evolución y medición
r = qe.apply_gate("[[1,1],[1,-1]]", "[[1],[0]]")  # H|0⟩
r = qe.measure_probabilities("[[1],[1]]")  # P(|0⟩), P(|1⟩)

# Información cuántica
r = qe.von_neumann_entropy("[[sp.Rational(1,2),0],[0,sp.Rational(1,2)]]")
# S(I/2) = 1 bit

r = qe.fidelity(rho_expr, sigma_expr)
# F(ρ, σ) = (Tr√(√ρ σ √ρ))²
```

**Verificación de invariantes cuánticos:**

```python
invariants = qe.verify_quantum_invariants(
    "[[0,1],[1,0]]",         # Pauli X
    ["unitary", "hermitian"]  # Propiedades a verificar
)
# → {"unitary": MathArtifact(passed=True), "hermitian": MathArtifact(passed=True)}
```

**Puertas soportadas:**

| Puerta     | Qubits | Parámetros             |
| ---------- | ------ | ---------------------- |
| H          | 1      | —                      |
| X, Y, Z    | 1      | —                      |
| S, T       | 1      | —                      |
| Rx, Ry, Rz | 1      | `theta`                |
| CNOT, CZ   | 2      | —                      |
| SWAP       | 2      | —                      |
| QFT        | N      | `n` (número de qubits) |

> **Nota sobre eigenvalores:** La entropía de von Neumann y la fidelidad
> usan `eigenvals().items()` con multiplicidad para resultados correctos.
> `S(I/2) = 1` (no 0.5).

---

## Fase 5 — Knowledge Graph Computacional

### `FormulaGraph` (`formula_graph.py`)

Grafo de conocimiento de fórmulas matemáticas con búsqueda por equivalencia
simbólica.

#### Fingerprinting Simbólico

Cada fórmula recibe un fingerprint invariante bajo renombramiento de variables:

```python
fpe = FormulaFingerprintEngine()

fp1 = fpe.fingerprint("sin(x)**2 + cos(x)**2")
fp2 = fpe.fingerprint("sin(y)**2 + cos(y)**2")
# fp1.hash == fp2.hash  (misma estructura, diferentes variables)

# Detección de equivalencia
equiv, method = fpe.are_equivalent("sin(2*x)", "2*sin(x)*cos(x)")
# → (True, "trigsimp")
```

**Propiedades del fingerprint:**

- `ops_signature`: Operaciones ordenadas extraídas vía `srepr()`
- `n_free_vars`: Número de variables libres
- `depth`: Profundidad del árbol de expresión
- `has_trig`, `has_exp`, `has_matrix`: Flags de estructura
- `polynomial_degree`: Grado polinomial (si aplica)
- `hash`: SHA256 de las propiedades combinadas

#### Grafo de Fórmulas

```python
graph = FormulaGraph()

# Añadir fórmulas (deduplicación automática por fingerprint)
id1 = graph.add_formula(
    "sin(x)**2 + cos(x)**2",
    latex=r"\sin^2(x) + \cos^2(x)",
    source_chunks=["nielsen_001"],
    description="Identidad pitagórica",
    domain="trigonometry",
)

# Relaciones entre fórmulas
graph.add_relation(id1, id2, RelationType.EQUIVALENT)
graph.add_relation(id3, id1, RelationType.DERIVES_FROM)

# Reglas de reescritura (e-graph)
graph.add_rewrite_rule(
    lhs="sin(2*x)",
    rhs="2*sin(x)*cos(x)",
    name="double_angle_sin",
    bidirectional=True,
)

# Búsquedas
results = graph.find_by_fingerprint("sin(a)**2 + cos(a)**2")
results = graph.find_by_structure(has_trig=True, domain="trigonometry")
path = graph.find_derivation_path(source_id, target_id)

# Aplicar reglas de reescritura
steps = graph.apply_rewrite_rules("sin(2*x)")
# → [("2*sin(x)*cos(x)", RewriteRule(name="double_angle_sin"))]
```

**Tipos de relación:** `DERIVES_FROM`, `EQUIVALENT`, `SPECIAL_CASE`,
`GENERALIZES`, `APPROXIMATES`, `DEFINES`, `COMPONENT_OF`, `CONTRADICTS`

### `FormulaRetriever` (`formula_retriever.py`)

Puente entre el FormulaGraph y el pipeline de retrieval.

```python
retriever = FormulaRetriever()

# Indexar chunks del corpus
n_indexed = retriever.index_chunks(chunks)

# Buscar por equivalencia simbólica
results = retriever.search_by_formula("E = m*c**2", top_k=10)

# Enriquecer resultados de retrieval con metadata de fórmulas
enriched = retriever.enrich_results(retrieval_results)

# Buscar fórmulas relacionadas por estructura
related = retriever.find_related_formulas("sin(x)**2 + cos(x)**2")
```

---

## Fase 6 — Verificación Formal (Lean 4)

### Concepto

Pipeline aspiracional de verificación formal:

1. El LLM traduce una proposición a Lean 4 (autoformalization)
2. El compilador Lean 4 verifica la prueba
3. Si falla, el LLM corrige (repair loop, hasta 3 intentos)
4. Si Lean no está disponible → degradación a verificación simbólica

### `FormalVerifier` (`formal_verifier.py`)

```python
verifier = FormalVerifier(lean_timeout=60, max_repair_attempts=3)

# Verificar proposición
proof = verifier.verify(
    statement="Para todo x, sin²(x) + cos²(x) = 1",
    context="Identidad trigonométrica fundamental",
    llm_fn=my_llm,
)

# proof.verified → True/False
# proof.lean_code → Código Lean 4 generado
# proof.attempts → Número de intentos
# proof.artifact → MathArtifact con resultado
```

### Componentes

#### `LeanInterface`

Wrapper del compilador Lean 4 vía subprocess.

```python
lean = LeanInterface(timeout=60)
lean.available  # True si Lean está instalado

result = lean.check("theorem trivial : True := trivial")
# LeanResult(success=True, errors=[], execution_time_ms=...)
```

Busca Lean en `~/.elan/bin/lean` o en `PATH`.

#### `Autoformalizator`

Traduce lenguaje natural + LaTeX a Lean 4.

```python
formalizer = Autoformalizator(max_attempts=3)

# Con LLM
code = formalizer.formalize("∀x, x² ≥ 0", context="...", llm_fn=my_llm)

# Sin LLM (template básico)
code = formalizer.formalize("∀x, x² ≥ 0")
# → "theorem auto_theorem : True := trivial  -- TODO: ..."

# Reparar errores
fixed = formalizer.repair(code, errors=["unknown identifier 'Real'"], llm_fn=my_llm)
```

### Degradación Graceful

Si Lean 4 no está instalado en el sistema:

- `FormalVerifier.available` → `False`
- `verify()` retorna un `FormalProof` con `verified=False`
- El `MathArtifact` asociado tiene `verification_level=NONE`
- El pipeline de verificación (`_verify_formal`) retorna `conclusive=False`
- La verificación se detiene en el nivel más alto disponible (típicamente SYMBOLIC o PHYSICAL)

---

## Integración en el Pipeline RAG

### `ResponseSynthesizer` (`src/generation/synthesizer.py`)

El motor matemático se integra en el método `_generate_with_computation()`,
que despacha entre dos modos según la configuración:

#### Modo Básico (Fase 1)

```yaml
math_computation:
  enabled: true
  multi_agent:
    enabled: false # ← modo básico
```

Usa `MathComputationOrchestrator` con el loop `<COMPUTE>` ↔ `<RESULT>`.

#### Modo Multi-Agente (Fase 3)

```yaml
math_computation:
  enabled: true
  multi_agent:
    enabled: true # ← modo multi-agente
```

Usa `MultiAgentOrchestrator` con los 4 agentes, provenance, y
verificación paso a paso.

### Metadata en la Respuesta

```python
response.metadata = {
    "math_computation": True,
    "math_mode": "basic_loop" | "multi_agent",
    "computation_steps": 3,
    "computation_iterations": 2,
    "math_artifacts": [...],          # Lista de MathArtifact.to_dict()
    "computation_trace": [...],       # ComputationSteps del loop
    "provenance": {...},              # Grafo PROV (solo multi-agente)
    "plan": {...},                    # DerivationPlan (solo multi-agente)
}
```

### Detección de Consultas Matemáticas

El `QueryTypeClassifier` detecta queries de tipo `MATHEMATICAL` mediante:

- Presencia de LaTeX (`$...$`, `\frac`, `\int`, etc.)
- Verbos matemáticos conjugados: "demuestra", "deriva", "calcula", "integra", "resuelve"
- Palabras clave: "eigenvalue", "hamiltoniano", "conmutador", etc.

---

## Configuración

Todas las opciones en `config/settings.yaml` bajo `math_computation`:

```yaml
math_computation:
  enabled: true # Activar motor matemático
  max_iterations: 5 # Máx. iteraciones del loop <COMPUTE>
  timeout_per_step_seconds: 30 # Timeout por ejecución en sandbox

  verification:
    enabled: false # Activar verificación automática
    default_level: 1 # Nivel mínimo por defecto
    max_level: 3 # Nivel máximo por defecto

  wolfram:
    enabled: false # Activar fallback a Wolfram Alpha
    timeout_seconds: 10

  latex_parser:
    use_llm_normalization: true # Usar LLM para normalizar LaTeX
    confidence_threshold: 0.7 # Umbral mínimo de confianza

  multi_agent:
    enabled: false # Activar modo multi-agente (Fase 3)
    max_steps: 10 # Máx. pasos de derivación
    provenance: true # Registrar grafo PROV

  quantum:
    enabled: true # Activar motor cuántico (Fase 4)
    use_qutip: false # Usar QuTiP (si instalado)
    use_pennylane: false # Usar PennyLane (si instalado)

  formula_graph:
    enabled: false # Activar knowledge graph (Fase 5)
    max_rewrite_steps: 10 # Máx. pasos de reescritura

  formal_verification:
    enabled: false # Activar verificación formal (Fase 6)
    lean_timeout_seconds: 60 # Timeout de compilación Lean
    max_repair_attempts: 3 # Intentos de reparación con LLM
```

### Variables de Entorno (`.env`)

```
WOLFRAM_MCP_API_KEY="<tu-api-key>"   # API key para Wolfram Alpha (obtener en developer.wolframalpha.com)
```

---

## Tests

### Suite de Tests E2E

| Archivo                        | Fases   | Tests  | Estado         |
| ------------------------------ | ------- | ------ | -------------- |
| `tests/test_e2e_phases_1_2.py` | 1-2     | 11     | 11/11 PASS     |
| `tests/test_e2e_phases_3_6.py` | 3-6     | 19     | 19/19 PASS     |
| **Total**                      | **1-6** | **30** | **30/30 PASS** |

### Ejecutar Tests

```bash
cd LibrarAI

# Fases 1-2
python3 tests/test_e2e_phases_1_2.py

# Fases 3-6
python3 tests/test_e2e_phases_3_6.py

# Ambos
python3 tests/test_e2e_phases_1_2.py && python3 tests/test_e2e_phases_3_6.py
```

### Cobertura por Fase

**Fase 1** (3 tests):

- MathEngine: 10 operaciones (solve, diff, integrate, simplify, verify, matrix, raw)
- Orchestrator: extracción de `<COMPUTE>`, limpieza de respuesta
- Loop E2E completo (extract → sandbox → result)

**Fase 2** (5 tests):

- VerificationPipeline: 6 verificaciones (trig, falsa, algebraica, unitary, hermitian, integrity)
- LaTeXParser: 5 formatos (fracción, raíz, trig, polinomio, área)
- WolframClient: query real a la API
- MathArtifact: serialización completa (dict, JSON, evidence block)
- Pipeline completo: LaTeX → parse → verify → artifact → serialize

**Fase 3** (5 tests):

- ProvenanceGraph: entidades, actividades, agentes, lineage
- DerivationStep/Plan: estructura y serialización
- CalculatorAgent: ejecución en sandbox
- VerifierAgent: producción de MathArtifacts
- MultiAgentOrchestrator: pipeline completo sin LLM

**Fase 4** (4 tests):

- 10 puertas cuánticas (H, X, Y, Z, S, T, CNOT, CZ, SWAP, QFT)
- Operaciones: conmutador, anticonmutador, tensor, apply_gate, measure
- Verificación: unitariedad, hermiticidad
- Entropía de von Neumann: S(|0⟩⟨0|)=0, S(I/2)=1

**Fase 5** (4 tests):

- Fingerprinting invariante bajo renombramiento de variables
- Equivalencia simbólica (trigsimp, expand)
- FormulaGraph: nodos, aristas, reglas, deduplicación, búsqueda
- Reglas de reescritura: aplicación sin loops infinitos

**Fase 6** (4 tests):

- LeanInterface: detección de disponibilidad
- Autoformalizator: templates sin LLM
- FormalVerifier: pipeline con degradación graceful
- LeanResult: estructura de datos

**Integración** (2 tests):

- Quantum + Verification: invariantes cuánticos con VerificationPipeline
- Pipeline completo Fases 3-6: multi-agente + quantum + formula graph + formal

---

## Patrones de Diseño

### 1. Ejecución en Sandbox

Toda ejecución de código se realiza en un `CodeSandbox` aislado con:

- Timeout configurable
- Sin acceso a red
- Builtins peligrosos bloqueados (`type()`, `__class__`)
- Resultados extraídos vía delimitadores `__MATH_RESULT__`

### 2. Inicialización Lazy

Los motores pesados (Wolfram, Quantum, FormalVerifier) se cargan bajo demanda
usando `@property` con inicialización diferida:

```python
@property
def quantum(self):
    if self._quantum is None:
        from .quantum import QuantumEngine
        self._quantum = QuantumEngine(math_engine=self.engine)
    return self._quantum
```

### 3. Verificación Dual

Las ecuaciones se verifican tanto simbólica como numéricamente para maximizar
la confianza (sampling + transforms específicos).

### 4. Degradación Graceful

Si un componente externo no está disponible (Lean 4, Wolfram, QuTiP),
el sistema se degrada al siguiente mejor nivel de verificación sin fallar.

### 5. Provenance Tracing

Cada afirmación computacional se puede trazar hasta su fuente original
mediante el grafo W3C PROV.

### 6. Estrategia SMART

"Reason first, verify with tools second" — el LLM razona primero y usa
herramientas computacionales para verificar, evitando Tool-Induced Myopia.

### 7. Dual-Source Grounding

Las respuestas se sustentan en dos fuentes independientes:

- Citas textuales `[n]` del corpus
- Evidencia computacional (MathArtifacts)

---

## Limitaciones y Restricciones del Sandbox

| Restricción                  | Detalle                                 | Workaround                               |
| ---------------------------- | --------------------------------------- | ---------------------------------------- |
| `type()` bloqueado           | En la lista `DANGEROUS_BUILTINS`        | Usar `isinstance()` o `srepr()` + regex  |
| `__class__` bloqueado        | En `DANGEROUS_ATTRS`                    | Usar `.__class__.__name__` vía `srepr()` |
| Sin acceso a red             | Sandbox aislado                         | Wolfram se ejecuta fuera del sandbox     |
| matplotlib puede crashear    | Incompatibilidad NumPy 2.x              | Envuelto en try/except                   |
| Warnings de recursión        | `tree_depth()` recursiva                | Son warnings, no errores                 |
| `pip` requiere flag          | Ubuntu system packages                  | `--break-system-packages`                |
| `python` no existe           | Solo `python3` en Ubuntu                | Siempre usar `python3`                   |
| f-strings en código generado | Variables evaluadas en scope incorrecto | Usar `{{var}}` para doble escape         |
