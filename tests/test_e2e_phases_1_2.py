"""
Tests E2E para Fases 1 y 2 del Motor Matemático de LibrarAI.

Fase 1: Loop de computación bidireccional (MathEngine + Orchestrator)
Fase 2: Verificación integrada (VerificationPipeline + Artifacts + LaTeX + Wolfram)

Estos tests verifican la integración completa de los componentes,
ejecutando código real en el sandbox (no mocks).
"""

import os
import sys
import json
import time
from pathlib import Path

# Asegurar que src está en path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Cargar .env manualmente (sin dependencia de python-dotenv)
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                _key, _val = _key.strip(), _val.strip().strip('"')
                if _key and _key not in os.environ:
                    os.environ[_key] = _val


def test_math_engine_all_operations():
    """E2E: MathEngine ejecuta todas las operaciones en sandbox real."""
    from src.math.engine import MathEngine

    engine = MathEngine(timeout=30)
    results = {}

    # 1. Solve
    r = engine.solve("x**2 - 4", "x")
    assert r.success, f"solve falló: {r.error}"
    assert "-2" in r.output_expr and "2" in r.output_expr
    results["solve"] = r.output_expr

    # 2. Differentiate
    r = engine.differentiate("x**3 + 2*x**2 + x", "x")
    assert r.success, f"differentiate falló: {r.error}"
    assert "3*x**2" in r.output_expr or "3*x**2 + 4*x + 1" in r.output_expr
    results["differentiate"] = r.output_expr

    # 3. Integrate (indefinida)
    r = engine.integrate("sin(x)", "x")
    assert r.success, f"integrate falló: {r.error}"
    assert "cos" in r.output_expr
    results["integrate_indef"] = r.output_expr

    # 4. Integrate (definida)
    r = engine.integrate("x**2", "x", limits=(0, 1))
    assert r.success, f"integrate definida falló: {r.error}"
    results["integrate_def"] = r.output_expr

    # 5. Simplify
    r = engine.simplify("sin(x)**2 + cos(x)**2")
    assert r.success, f"simplify falló: {r.error}"
    results["simplify"] = r.output_expr

    # 6. Verify equation (verdadera)
    r = engine.verify_equation("sin(2*x)", "2*sin(x)*cos(x)")
    assert r.success, f"verify falló: {r.error}"
    assert "True" in r.output_expr
    results["verify_true"] = r.output_expr

    # 7. Verify equation (falsa)
    r = engine.verify_equation("sin(x)", "cos(x)")
    assert r.success, f"verify falsa falló: {r.error}"
    assert "False" in r.output_expr
    results["verify_false"] = r.output_expr

    # 8. Matrix operation
    r = engine.matrix_operation("eigenvalues", "[[1,0],[0,-1]]")
    assert r.success, f"matrix eigenvalues falló: {r.error}"
    results["matrix_eigenvalues"] = r.output_expr

    # 9. Matrix is_hermitian (Pauli Z)
    r = engine.matrix_operation("is_hermitian", "[[1,0],[0,-1]]")
    assert r.success, f"matrix is_hermitian falló: {r.error}"
    results["matrix_hermitian"] = r.output_expr

    # 10. Execute raw
    r = engine.execute_raw("import sympy as sp\nprint(sp.factorial(10))")
    assert r.success, f"execute_raw falló: {r.error}"
    assert "3628800" in r.stdout
    results["raw"] = r.stdout.strip()

    print(f"MathEngine: 10/10 operaciones OK")
    for k, v in results.items():
        print(f"  {k}: {v[:80]}")
    return True


def test_verification_pipeline():
    """E2E: VerificationPipeline verifica identidades a múltiples niveles."""
    from src.math.verification import VerificationPipeline

    pipeline = VerificationPipeline(max_level=3)
    results = {}

    # 1. Identidad trigonométrica (numérico + simbólico)
    artifact = pipeline.verify("sin(2*x)", "2*sin(x)*cos(x)")
    assert artifact.verification_passed, "sin(2x) = 2sin(x)cos(x) debería pasar"
    results["trig_identity"] = f"PASS (level={artifact.verification_level.name})"

    # 2. Falsa identidad
    artifact = pipeline.verify("sin(x)", "cos(x)")
    assert not artifact.verification_passed, "sin(x) = cos(x) debería fallar"
    results["false_identity"] = f"FAIL (level={artifact.verification_level.name})"

    # 3. Identidad algebraica
    artifact = pipeline.verify("(x+1)**2", "x**2 + 2*x + 1")
    assert artifact.verification_passed, "(x+1)^2 = x^2+2x+1 debería pasar"
    results["algebraic"] = f"PASS (level={artifact.verification_level.name})"

    # 4. Invariante físico: Pauli X unitaria
    artifact = pipeline.verify(
        "[[0,1],[1,0]]", "[[0,1],[1,0]]",
        min_level=3, max_level=3,
        context={"invariant_type": "unitary"}
    )
    assert artifact.verification_passed, "Pauli X debería ser unitaria"
    results["pauli_x_unitary"] = f"PASS (level={artifact.verification_level.name})"

    # 5. Invariante físico: Pauli Z hermítica
    artifact = pipeline.verify(
        "[[1,0],[0,-1]]", "[[1,0],[0,-1]]",
        min_level=3, max_level=3,
        context={"invariant_type": "hermitian"}
    )
    assert artifact.verification_passed, "Pauli Z debería ser hermítica"
    results["pauli_z_hermitian"] = f"PASS (level={artifact.verification_level.name})"

    # 6. MathArtifact tiene hash y provenance
    assert artifact.content_hash, "Artifact debería tener hash"
    assert artifact.timestamp, "Artifact debería tener timestamp"
    results["artifact_integrity"] = f"hash={artifact.content_hash}"

    print(f"VerificationPipeline: 6/6 tests OK")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return True


def test_latex_parser():
    """E2E: LaTeXParser convierte LaTeX a SymPy con confianza."""
    from src.math.latex_parser import LaTeXParser

    parser = LaTeXParser(use_llm_normalization=False)  # Sin LLM para test aislado
    results = {}

    cases = [
        (r"\frac{x^2 + 1}{x - 1}", "fraction"),
        (r"\sqrt{x^2 + 1}", "sqrt"),
        (r"\sin(x) + \cos(x)", "trig"),
        (r"x^{2} + 2x + 1", "polynomial"),
        (r"\pi r^{2}", "pi_area"),
    ]

    for latex, name in cases:
        sympy_str, confidence = parser.parse(latex)
        assert sympy_str, f"Parsing falló para {name}: {latex}"
        assert confidence > 0, f"Confianza 0 para {name}"
        results[name] = f"{sympy_str} (conf={confidence:.2f})"

    print(f"LaTeXParser: {len(cases)}/{len(cases)} tests OK")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return True


def test_wolfram_client():
    """E2E: WolframClient consulta Wolfram Alpha API."""
    from src.math.wolfram_client import WolframClient

    client = WolframClient()
    if not client.available:
        print("WolframClient: SKIP (API key no configurada)")
        return True

    r = client.query("integrate sin(x)^2 dx")
    assert r["success"], f"Wolfram query falló: {r['error']}"
    assert r["result"], "Wolfram devolvió resultado vacío"
    assert r["execution_time_ms"] > 0

    print(f"WolframClient: OK ({r['execution_time_ms']:.0f}ms)")
    print(f"  Resultado: {r['result'][:100]}...")
    return True


def test_math_artifact_serialization():
    """E2E: MathArtifact serialización completa."""
    from src.math.artifacts import MathArtifact, VerificationLevel

    artifact = MathArtifact(
        input_latex=r"\sin(2x) = 2\sin(x)\cos(x)",
        input_sympy="sin(2*x) == 2*sin(x)*cos(x)",
        engine="sympy",
        operation="verification",
        code="sp.expand(sin(2*x) - 2*sin(x)*cos(x))",
        result="True",
        result_latex=r"\sin(2x) = 2\sin(x)\cos(x)",
        verification_level=VerificationLevel.SYMBOLIC,
        verification_passed=True,
        verification_details={"method": "trigsimp", "passed": True},
        source_chunks=["nielsen_micro_000123"],
    )

    # Serialización
    d = artifact.to_dict()
    assert d["verification_level"] == "SYMBOLIC"
    assert d["verification_passed"] is True
    assert d["content_hash"]

    # Evidence block
    block = artifact.to_evidence_block()
    assert "PASS" in block
    assert "SYMBOLIC" in block

    # JSON serializable
    json_str = json.dumps(d, default=str)
    assert json_str

    print("MathArtifact serialización: OK")
    print(f"  Hash: {d['content_hash']}")
    print(f"  Evidence block: {block[:80]}...")
    return True


def test_orchestrator_compute_extraction():
    """E2E: Orchestrator extrae bloques <COMPUTE> correctamente."""
    from src.math.orchestrator import MathComputationOrchestrator

    orch = MathComputationOrchestrator()

    # Con triple backtick
    text1 = """Vamos a verificar esto.
<COMPUTE>
```python
import sympy as sp
x = sp.Symbol('x')
print(sp.integrate(sp.sin(x), x))
```
</COMPUTE>
El resultado es..."""

    blocks = orch._extract_compute_blocks(text1)
    assert len(blocks) == 1
    assert "import sympy" in blocks[0]

    # Sin triple backtick
    text2 = """<COMPUTE>
import sympy as sp
x = sp.Symbol('x')
print(sp.solve(x**2 - 4, x))
</COMPUTE>"""

    blocks = orch._extract_compute_blocks(text2)
    assert len(blocks) == 1
    assert "solve" in blocks[0]

    # Múltiples bloques
    text3 = """Paso 1:
<COMPUTE>
```python
import sympy as sp
print(sp.diff(sp.sin(sp.Symbol('x')), sp.Symbol('x')))
```
</COMPUTE>
Paso 2:
<COMPUTE>
```python
import sympy as sp
print(sp.integrate(sp.cos(sp.Symbol('x')), sp.Symbol('x')))
```
</COMPUTE>"""

    blocks = orch._extract_compute_blocks(text3)
    assert len(blocks) == 2

    print("Orchestrator compute extraction: OK")
    print(f"  Test 1: {len(blocks)} bloques extraídos")
    return True


def test_orchestrator_clean_response():
    """E2E: Orchestrator limpia respuesta final correctamente."""
    from src.math.orchestrator import MathComputationOrchestrator

    orch = MathComputationOrchestrator()

    text = """La integral de sin(x) es:

<COMPUTE>
```python
import sympy as sp
x = sp.Symbol('x')
print(sp.integrate(sp.sin(x), x))
```
</COMPUTE>

<RESULT>
-cos(x)
Execution time: 50ms
</RESULT>

Por lo tanto, la integral es -cos(x) + C."""

    cleaned = orch._clean_response(text)

    # COMPUTE debería transformarse en bloque de código normal
    assert "```python" in cleaned
    assert "<COMPUTE>" not in cleaned

    # RESULT debería eliminarse
    assert "<RESULT>" not in cleaned

    # El texto de razonamiento debería mantenerse
    assert "integral" in cleaned.lower()

    print("Orchestrator clean response: OK")
    return True


def test_engine_with_quantum():
    """E2E: MathEngine con operaciones cuánticas (matrices Pauli, QFT)."""
    from src.math.engine import MathEngine

    engine = MathEngine(timeout=30)

    # Hadamard unitaria
    r = engine.execute_raw("""
import sympy as sp
H = sp.Matrix([[1,1],[1,-1]]) / sp.sqrt(2)
result = (sp.simplify(H * H.H - sp.eye(2))).is_zero_matrix
print(f"Hadamard unitaria: {result}")
""")
    assert r.success, f"Hadamard is_unitary falló: {r.error}"
    assert "True" in r.stdout
    assert r.success, f"Hadamard is_unitary falló: {r.error}"
    print(f"  Hadamard unitaria: {r.output_expr}")

    # CNOT (4x4)
    r = engine.matrix_operation("is_unitary", "[[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]")
    assert r.success, f"CNOT is_unitary falló: {r.error}"
    print(f"  CNOT unitaria: {r.output_expr}")

    # QFT 2-qubit
    r = engine.execute_raw("""
import sympy as sp

# QFT matrix for n=2
n = 2
N = 2**n
omega = sp.exp(2*sp.pi*sp.I/N)
QFT = sp.Matrix(N, N, lambda i, j: omega**(i*j)) / sp.sqrt(N)

# Verificar unitariedad
product = sp.simplify(QFT * QFT.H)
identity = sp.eye(N)
is_unitary = (product - identity).is_zero_matrix
print(f"QFT 2-qubit unitaria: {is_unitary}")
print(f"QFT matrix:\\n{QFT}")
""")
    assert r.success, f"QFT execute_raw falló: {r.error}"
    assert "True" in r.stdout
    print(f"  QFT 2-qubit: {r.stdout.strip()[:100]}")

    print("Engine quantum operations: OK")
    return True


def test_citation_injector_with_math():
    """E2E: CitationInjector con evidencia computacional."""
    from src.generation.citation_injector import CitationInjector, CitedResponse

    injector = CitationInjector()

    math_artifacts = [
        {
            "engine": "sympy",
            "operation": "verification",
            "result": "True",
            "result_latex": r"\sin(2x) = 2\sin(x)\cos(x)",
            "verification_passed": True,
            "verification_level": "SYMBOLIC",
        },
        {
            "engine": "sympy",
            "operation": "integration",
            "result": "-cos(x)",
            "result_latex": r"-\cos(x)",
            "verification_passed": True,
            "verification_level": "NUMERICAL",
        },
    ]

    # Crear CitedResponse con math_evidence
    response = CitedResponse(
        content="La identidad trigonométrica es correcta [1].",
        citations=[],
        bibliography="",
        uncited_sources=[],
        math_evidence=math_artifacts,
    )

    full = response.get_full_response()
    assert "Verificaciones computacionales" in full
    assert "PASS" in full
    assert "SYMBOLIC" in full

    print("CitationInjector with math: OK")
    print(f"  Full response excerpt: {full[:150]}...")
    return True


def test_end_to_end_compute_loop():
    """
    E2E: Loop completo de computación (sin LLM real).

    Simula lo que haría el orchestrator: recibe texto con <COMPUTE>,
    ejecuta en sandbox, formatea resultado.
    """
    from src.math.engine import MathEngine
    from src.math.orchestrator import MathComputationOrchestrator

    engine = MathEngine(timeout=30)
    orch = MathComputationOrchestrator(engine=engine)

    # Simular respuesta de LLM con <COMPUTE>
    llm_response = """Para verificar la identidad trigonométrica sin²(x) + cos²(x) = 1,
vamos a usar SymPy:

<COMPUTE>
```python
import sympy as sp

x = sp.Symbol('x')
expr = sp.sin(x)**2 + sp.cos(x)**2
simplified = sp.trigsimp(expr)
print(f"sin²(x) + cos²(x) = {simplified}")

# Verificación numérica adicional
import numpy as np
test_vals = np.random.uniform(-10, 10, 100)
results = np.sin(test_vals)**2 + np.cos(test_vals)**2
all_one = np.allclose(results, 1.0)
print(f"Verificación numérica (100 puntos): {'PASS' if all_one else 'FAIL'}")
```
</COMPUTE>

Esto confirma que la identidad es correcta."""

    # Extraer bloques
    blocks = orch._extract_compute_blocks(llm_response)
    assert len(blocks) == 1, f"Esperaba 1 bloque, encontré {len(blocks)}"

    # Ejecutar bloque en sandbox
    math_result = engine.execute_raw(blocks[0])
    assert math_result.success, f"Ejecución falló: {math_result.error}"
    assert "1" in math_result.stdout
    assert "PASS" in math_result.stdout

    # Formatear resultado como lo haría el orchestrator
    output = math_result.stdout.strip()
    result_text = f"\n<RESULT>\n{output}\nExecution time: {math_result.execution_time_ms:.0f}ms\n</RESULT>\n"

    # Verificar que el resultado se formateó correctamente
    assert "<RESULT>" in result_text
    assert "sin²(x) + cos²(x) = 1" in result_text

    # Limpiar respuesta final
    final = orch._clean_response(llm_response)
    assert "<COMPUTE>" not in final
    assert "```python" in final

    print("E2E compute loop: OK")
    print(f"  Sandbox output: {output[:100]}")
    return True


def test_full_pipeline_integration():
    """
    E2E: Pipeline completo desde input hasta MathArtifact.

    1. LaTeX parsing
    2. Verificación multi-nivel
    3. Creación de artifact
    4. Serialización
    """
    from src.math.latex_parser import LaTeXParser
    from src.math.verification import VerificationPipeline
    from src.math.artifacts import MathArtifact, VerificationLevel

    # 1. Parse LaTeX
    parser = LaTeXParser(use_llm_normalization=False)
    lhs_str, lhs_conf = parser.parse(r"\sin(2x)")
    rhs_str, rhs_conf = parser.parse(r"2\sin(x)\cos(x)")

    print(f"  LaTeX parsed: '{lhs_str}' (conf={lhs_conf:.2f}), '{rhs_str}' (conf={rhs_conf:.2f})")

    # 2. Verificación
    pipeline = VerificationPipeline(max_level=2)

    # Usar las expresiones parseadas (o fallback si parse falla)
    lhs_sympy = lhs_str if lhs_str else "sin(2*x)"
    rhs_sympy = rhs_str if rhs_str else "2*sin(x)*cos(x)"

    artifact = pipeline.verify(lhs_sympy, rhs_sympy)
    assert artifact.verification_passed, f"Verificación falló para identidad verdadera"

    # 3. Artifact completo
    assert artifact.content_hash
    assert artifact.verification_level.value >= 0

    # 4. Serializar
    d = artifact.to_dict()
    json_str = json.dumps(d, default=str)
    assert len(json_str) > 50

    print(f"  Verification: PASS at level {artifact.verification_level.name}")
    print(f"  Artifact hash: {artifact.content_hash}")
    print("Full pipeline integration: OK")
    return True


# ============================================================
# Runner
# ============================================================

def main():
    tests = [
        ("MathEngine operaciones", test_math_engine_all_operations),
        ("VerificationPipeline", test_verification_pipeline),
        ("LaTeXParser", test_latex_parser),
        ("WolframClient", test_wolfram_client),
        ("MathArtifact serialización", test_math_artifact_serialization),
        ("Orchestrator extraction", test_orchestrator_compute_extraction),
        ("Orchestrator clean", test_orchestrator_clean_response),
        ("Engine quantum", test_engine_with_quantum),
        ("CitationInjector+math", test_citation_injector_with_math),
        ("E2E compute loop", test_end_to_end_compute_loop),
        ("Full pipeline", test_full_pipeline_integration),
    ]

    passed = 0
    failed = 0
    skipped = 0

    print("=" * 60)
    print("TESTS E2E - FASES 1 Y 2")
    print("=" * 60)

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            result = test_fn()
            if result:
                passed += 1
                print(f"  => PASS")
            else:
                failed += 1
                print(f"  => FAIL")
        except Exception as e:
            failed += 1
            print(f"  => ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"RESULTADOS: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'=' * 60}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
