"""
Tests E2E para Fases 3-6 del Motor Matemático de LibrarAI.

Fase 3: Multi-agente + Provenance W3C PROV
Fase 4: Computación cuántica especializada
Fase 5: Knowledge Graph Computacional + Fingerprinting
Fase 6: Verificación formal (Lean 4)
"""

import os
import sys
import json
from pathlib import Path

import pytest

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


# ============================================================
# FASE 3: Multi-agente + Provenance
# ============================================================

@pytest.mark.e2e
def test_provenance_graph():
    """E2E: ProvenanceGraph registra trazabilidad W3C PROV."""
    from src.math.provenance import (
        ProvenanceGraph, EntityType, ActivityType, AgentType
    )

    graph = ProvenanceGraph()

    # Registrar agentes
    llm_id = graph.add_agent(AgentType.LLM, "Claude", "opus-4.6")
    sympy_id = graph.add_agent(AgentType.SYMPY, "SymPy", "1.12")

    # Registrar entidad fuente
    source_id = graph.add_entity(
        EntityType.SOURCE_CHUNK,
        content="La identidad de Euler: e^{iπ} + 1 = 0",
        metadata={"chunk_id": "euler_001"},
    )

    # Actividad de computación
    compute_id = graph.add_activity(
        ActivityType.COMPUTATION,
        description="Verificar identidad de Euler",
        used=[source_id],
        agent_id=sympy_id,
    )

    # Resultado
    result_id = graph.add_entity(
        EntityType.COMPUTATION_RESULT,
        content="exp(I*pi) + 1 = 0: True",
    )
    graph.record_generation(compute_id, result_id)
    graph.record_derivation(result_id, source_id)
    graph.end_activity(compute_id)

    # Actividad de síntesis
    synth_id = graph.add_activity(
        ActivityType.SYNTHESIS,
        description="Generar respuesta final",
        used=[result_id],
        agent_id=llm_id,
    )
    final_id = graph.add_entity(EntityType.FINAL_RESPONSE, content="La identidad es correcta.")
    graph.record_generation(synth_id, final_id)

    # Verificar grafo
    data = graph.to_dict()
    assert len(data["entities"]) == 3
    assert len(data["activities"]) == 2
    assert len(data["agents"]) == 2
    assert len(data["derivations"]) == 1

    # Verificar lineage
    lineage = graph.get_lineage(final_id)
    assert len(lineage) > 0

    print(f"ProvenanceGraph: OK")
    print(f"  {graph.summary()}")
    print(f"  Lineage depth for final response: {len(lineage)} nodes")


@pytest.mark.e2e
def test_derivation_step_structure():
    """E2E: DerivationStep y DerivationPlan se serializan correctamente."""
    from src.math.agents import DerivationStep, DerivationPlan, StepStatus

    plan = DerivationPlan(
        goal="Demostrar que sin²(x) + cos²(x) = 1",
        strategy="directa",
        steps=[
            DerivationStep(
                index=0,
                description="Definir sin y cos como componentes del círculo unidad",
                expression="sin(x)**2 + cos(x)**2",
                justification="Teorema de Pitágoras",
                status=StepStatus.VERIFIED,
            ),
            DerivationStep(
                index=1,
                description="Simplificar usando identidad pitagórica",
                expression="1",
                justification="Identidad fundamental",
                status=StepStatus.VERIFIED,
            ),
        ],
        assumptions=["x es real"],
        required_tools=["sympy"],
    )

    d = plan.to_dict()
    assert d["goal"] == "Demostrar que sin²(x) + cos²(x) = 1"
    assert len(d["steps"]) == 2
    assert d["steps"][0]["status"] == "verified"

    print("DerivationStep/Plan: OK")


@pytest.mark.e2e
def test_calculator_agent():
    """E2E: CalculatorAgent ejecuta pasos en sandbox real."""
    from src.math.agents import CalculatorAgent, DerivationStep
    from src.math.engine import MathEngine

    engine = MathEngine(timeout=30)
    calculator = CalculatorAgent(engine)

    # Paso con expresión SymPy
    step = DerivationStep(
        index=0,
        description="Calcular la derivada de x^3",
        expression="3*x**2",
    )

    result = calculator.compute_step(step, [])
    assert result.success, f"CalculatorAgent falló: {result.error}"
    print(f"  Paso con expresión: {result.output_expr}")

    # Paso sin expresión (solo descripción)
    step2 = DerivationStep(
        index=1,
        description="Verificar que 2+2=4",
    )

    result2 = calculator.compute_step(step2, [])
    # Sin expresión ni LLM, debería fallar gracefully
    assert not result2.success  # Expected: no expression, no LLM

    print("CalculatorAgent: OK")


@pytest.mark.e2e
def test_verifier_agent():
    """E2E: VerifierAgent produce MathArtifacts."""
    from src.math.agents import VerifierAgent, DerivationStep
    from src.math.engine import MathEngine, MathResult

    engine = MathEngine(timeout=30)
    verifier = VerifierAgent(engine)

    step = DerivationStep(
        index=0,
        description="Verificar identidad",
        expression="sin(x)**2 + cos(x)**2 - 1",
    )

    artifact = verifier.verify_step(step)
    assert artifact is not None
    assert artifact.verification_level is not None
    print(f"  Artifact: passed={artifact.verification_passed}, level={artifact.verification_level.name}")

    # Verificar con MathResult
    math_result = MathResult(
        success=True,
        operation="test",
        input_expr="x^2",
        output_expr="x**2",
    )
    artifact2 = verifier.verify_step(
        DerivationStep(index=1, description="test"),
        math_result=math_result,
    )
    assert artifact2 is not None

    print("VerifierAgent: OK")


@pytest.mark.e2e
def test_multi_agent_orchestrator_no_llm():
    """E2E: MultiAgentOrchestrator sin LLM (modo estructurado)."""
    from src.math.agents import MultiAgentOrchestrator

    orch = MultiAgentOrchestrator(max_steps=3)

    result = orch.run(
        query="Verifica que sin(2x) = 2sin(x)cos(x)",
        retrieval_context="La identidad del ángulo doble establece que sin(2x) = 2sin(x)cos(x).",
    )

    assert "response" in result
    assert "plan" in result
    assert "artifacts" in result
    assert "provenance" in result
    assert result["steps_total"] > 0

    prov = result["provenance"]
    assert len(prov["entities"]) > 0
    assert len(prov["activities"]) > 0
    assert len(prov["agents"]) > 0

    print("MultiAgentOrchestrator (no LLM): OK")
    print(f"  Steps: {result['steps_verified']}/{result['steps_total']} verified")
    print(f"  Artifacts: {len(result['artifacts'])}")
    print(f"  Provenance: {len(prov['entities'])} entities, {len(prov['activities'])} activities")


# ============================================================
# FASE 4: Computación Cuántica
# ============================================================

@pytest.mark.e2e
def test_quantum_gates():
    """E2E: QuantumEngine genera y verifica puertas cuánticas."""
    from src.math.quantum import QuantumEngine

    qe = QuantumEngine()
    results = {}

    # Pauli gates
    for gate in ["H", "X", "Y", "Z", "S", "T"]:
        r = qe.quantum_gate(gate)
        assert r.success, f"Gate {gate} falló: {r.error}"
        results[gate] = "OK"

    # Multi-qubit gates
    for gate in ["CNOT", "CZ", "SWAP"]:
        r = qe.quantum_gate(gate)
        assert r.success, f"Gate {gate} falló: {r.error}"
        results[gate] = "OK"

    # QFT 2-qubit
    r = qe.quantum_gate("QFT", {"n": 2})
    assert r.success, f"QFT falló: {r.error}"
    results["QFT(2)"] = "OK"

    print(f"Quantum gates: {len(results)}/{len(results)} OK")
    for k, v in results.items():
        print(f"  {k}: {v}")


@pytest.mark.e2e
def test_quantum_operations():
    """E2E: QuantumEngine operaciones cuánticas avanzadas."""
    from src.math.quantum import QuantumEngine

    qe = QuantumEngine()

    # Conmutador de Pauli: [X, Y] = 2iZ
    r = qe.commutator("[[0,1],[1,0]]", "[[0,-1j],[1j,0]]")
    assert r.success, f"Commutator falló: {r.error}"
    print(f"  [X, Y] = {r.output_expr[:60]}")

    # Anticonmutador de Pauli: {X, Y} = 0
    r = qe.anticommutator("[[0,1],[1,0]]", "[[0,-1j],[1j,0]]")
    assert r.success, f"Anticommutator falló: {r.error}"
    print(f"  {{X, Y}} = {r.output_expr[:60]}")

    # Producto tensorial: X ⊗ Z
    r = qe.tensor_product("[[0,1],[1,0]]", "[[1,0],[0,-1]]")
    assert r.success, f"Tensor product falló: {r.error}"
    print(f"  X ⊗ Z: {r.output_expr[:60]}")

    # Aplicar Hadamard a |0⟩
    r = qe.apply_gate("[[1,1],[1,-1]]", "[[1],[0]]")
    assert r.success, f"Apply gate falló: {r.error}"
    print(f"  H|0⟩ = {r.output_expr[:60]}")

    # Probabilidades de medición de |+⟩
    r = qe.measure_probabilities("[[1],[1]]")  # No normalizado
    assert r.success, f"Measure probabilities falló: {r.error}"
    print(f"  P(|+⟩): {r.output_expr[:60]}")

    print("Quantum operations: OK")


@pytest.mark.e2e
def test_quantum_verification():
    """E2E: Verificación de unitariedad y hermiticidad."""
    from src.math.quantum import QuantumEngine

    qe = QuantumEngine()

    # Hadamard es unitaria
    r = qe.verify_unitary("sp.Matrix([[1,1],[1,-1]]) / sp.sqrt(2)")
    assert r.success, f"Verify unitary falló: {r.error}"
    assert "True" in r.output_expr
    print(f"  Hadamard unitaria: {r.output_expr}")

    # Pauli X es hermítica
    r = qe.verify_hermitian("[[0,1],[1,0]]")
    assert r.success, f"Verify hermitian falló: {r.error}"
    assert "True" in r.output_expr
    print(f"  Pauli X hermítica: {r.output_expr}")

    print("Quantum verification: OK")


@pytest.mark.e2e
def test_von_neumann_entropy():
    """E2E: Entropía de von Neumann para estados cuánticos."""
    from src.math.quantum import QuantumEngine

    qe = QuantumEngine()

    # Estado puro (entropía = 0)
    r = qe.von_neumann_entropy("[[1,0],[0,0]]")
    assert r.success, f"Von Neumann entropy falló: {r.error}"
    print(f"  S(|0⟩⟨0|) = {r.output_expr}")

    # Estado maximamente mezclado (entropía = 1 bit)
    r = qe.von_neumann_entropy("[[sp.Rational(1,2),0],[0,sp.Rational(1,2)]]")
    assert r.success, f"Von Neumann entropy mezclado falló: {r.error}"
    assert r.output_expr == "1", f"S(I/2) debería ser 1, obtuvo {r.output_expr}"
    print(f"  S(I/2) = {r.output_expr}")

    print("Von Neumann entropy: OK")


# ============================================================
# FASE 5: Knowledge Graph Computacional
# ============================================================

@pytest.mark.e2e
def test_formula_fingerprinting():
    """E2E: Fingerprinting simbólico de fórmulas."""
    from src.math.formula_graph import FormulaFingerprintEngine

    fpe = FormulaFingerprintEngine()

    # Misma fórmula con diferentes variables
    fp1 = fpe.fingerprint("sin(x)**2 + cos(x)**2")
    fp2 = fpe.fingerprint("sin(y)**2 + cos(y)**2")
    fp3 = fpe.fingerprint("x**2 + x + 1")

    assert fp1.hash, "Fingerprint 1 debería tener hash"
    assert fp2.hash, "Fingerprint 2 debería tener hash"

    # Misma estructura → mismo fingerprint
    assert fp1.has_trig == fp2.has_trig
    assert fp1.n_free_vars == fp2.n_free_vars

    # Estructura diferente → diferente fingerprint
    assert fp1.has_trig != fp3.has_trig

    print(f"Formula fingerprinting: OK")
    print(f"  sin(x)^2+cos(x)^2: hash={fp1.hash}, trig={fp1.has_trig}")
    print(f"  sin(y)^2+cos(y)^2: hash={fp2.hash}, trig={fp2.has_trig}")
    print(f"  x^2+x+1: hash={fp3.hash}, trig={fp3.has_trig}")


@pytest.mark.e2e
def test_formula_equivalence():
    """E2E: Detección de equivalencia simbólica."""
    from src.math.formula_graph import FormulaFingerprintEngine

    fpe = FormulaFingerprintEngine()

    # Equivalentes
    equiv, method = fpe.are_equivalent("sin(2*x)", "2*sin(x)*cos(x)")
    assert equiv, "sin(2x) y 2sin(x)cos(x) deberían ser equivalentes"
    print(f"  sin(2x) == 2sin(x)cos(x): {equiv} (method: {method})")

    equiv, method = fpe.are_equivalent("(x+1)**2", "x**2 + 2*x + 1")
    assert equiv, "(x+1)^2 y x^2+2x+1 deberían ser equivalentes"
    print(f"  (x+1)^2 == x^2+2x+1: {equiv} (method: {method})")

    # No equivalentes
    equiv, method = fpe.are_equivalent("sin(x)", "cos(x)")
    assert not equiv, "sin(x) y cos(x) no deberían ser equivalentes"
    print(f"  sin(x) == cos(x): {equiv} (method: {method})")

    print("Formula equivalence: OK")


@pytest.mark.e2e
def test_formula_graph():
    """E2E: Knowledge Graph de fórmulas completo."""
    from src.math.formula_graph import FormulaGraph, RelationType

    graph = FormulaGraph()

    # Añadir fórmulas
    id1 = graph.add_formula(
        "sin(x)**2 + cos(x)**2",
        latex=r"\sin^2(x) + \cos^2(x)",
        source_chunks=["nielsen_001"],
        description="Identidad pitagórica",
        domain="trigonometry",
    )

    id2 = graph.add_formula(
        "1",
        description="Uno",
        domain="constants",
    )

    id3 = graph.add_formula(
        "sin(2*x)",
        latex=r"\sin(2x)",
        source_chunks=["calc_042"],
        description="Seno del ángulo doble",
        domain="trigonometry",
    )

    id4 = graph.add_formula(
        "2*sin(x)*cos(x)",
        latex=r"2\sin(x)\cos(x)",
        source_chunks=["calc_043"],
        description="Ángulo doble expandido",
        domain="trigonometry",
    )

    # Añadir relaciones
    graph.add_relation(id1, id2, RelationType.EQUIVALENT, description="Identidad pitagórica")
    graph.add_relation(id3, id4, RelationType.EQUIVALENT, description="Ángulo doble")
    graph.add_relation(id3, id1, RelationType.DERIVES_FROM, description="Usa identidad pitagórica")

    # Añadir regla de reescritura
    rule = graph.add_rewrite_rule(
        lhs="sin(2*x)",
        rhs="2*sin(x)*cos(x)",
        name="double_angle_sin",
        source_chunk="calc_042",
    )
    assert rule.verified, "Regla de ángulo doble debería verificarse"

    # Buscar por dominio
    trig = graph.find_by_structure(has_trig=True)
    assert len(trig) >= 2, f"Debería haber al menos 2 fórmulas trigonométricas, encontré {len(trig)}"

    # Buscar camino de derivación
    path = graph.find_derivation_path(id3, id2)
    # Puede o no encontrar camino dependiendo de las aristas

    # Detectar duplicado
    id_dup = graph.add_formula("sin(2*x)")
    assert id_dup == id3, f"Debería detectar duplicado: {id_dup} != {id3}"

    data = graph.to_dict()
    assert data["stats"]["n_nodes"] >= 4
    assert data["stats"]["n_edges"] >= 2
    assert data["stats"]["n_verified_rules"] >= 1

    print(f"FormulaGraph: OK")
    print(f"  {graph.summary()}")


@pytest.mark.e2e
def test_rewrite_rules():
    """E2E: Reglas de reescritura del e-graph."""
    from src.math.formula_graph import FormulaGraph

    graph = FormulaGraph()

    # Añadir reglas trigonométricas
    graph.add_rewrite_rule("sin(2*x)", "2*sin(x)*cos(x)", "double_angle_sin")
    graph.add_rewrite_rule("cos(2*x)", "cos(x)**2 - sin(x)**2", "double_angle_cos")
    graph.add_rewrite_rule("sin(x)**2 + cos(x)**2", "1", "pythagorean")

    verified = sum(1 for r in graph.rewrite_rules if r.verified)
    print(f"  {verified}/{len(graph.rewrite_rules)} reglas verificadas")

    # Aplicar reglas a una expresión
    steps = graph.apply_rewrite_rules("sin(2*x)")
    print(f"  Rewriting sin(2x): {len(steps)} pasos")
    for expr, rule in steps:
        print(f"    → {expr} (via {rule.name})")

    print("Rewrite rules: OK")


# ============================================================
# FASE 6: Verificación Formal (Lean 4)
# ============================================================

@pytest.mark.e2e
def test_lean_interface():
    """E2E: LeanInterface detecta disponibilidad de Lean 4."""
    from src.math.formal_verifier import LeanInterface

    lean = LeanInterface()
    available = lean.available

    if available:
        # Compilar código trivial
        result = lean.check("theorem trivial_thm : True := trivial")
        print(f"  Lean 4 disponible: {available}")
        print(f"  Compilación trivial: {'OK' if result.success else 'FAIL'}")
    else:
        print(f"  Lean 4 no disponible (esperado en este sistema)")

    print("LeanInterface: OK")


@pytest.mark.e2e
def test_autoformalizator_templates():
    """E2E: Autoformalizator genera templates sin LLM."""
    from src.math.formal_verifier import Autoformalizator

    formalizer = Autoformalizator()

    # Sin LLM, debería generar template básico
    code = formalizer.formalize("Para todo x real, x^2 >= 0")
    assert "theorem" in code or "True" in code
    assert "sorry" not in code.lower() or "todo" in code.lower()

    print("Autoformalizator templates: OK")
    print(f"  Generated: {code[:80]}...")


@pytest.mark.e2e
def test_formal_verifier():
    """E2E: FormalVerifier pipeline completo (sin Lean = graceful degradation)."""
    from src.math.formal_verifier import FormalVerifier

    verifier = FormalVerifier()

    proof = verifier.verify("Para todo x, sin(x)^2 + cos(x)^2 = 1")

    assert proof.artifact is not None
    if verifier.available:
        print(f"  Lean 4 disponible: verificación formal intentada")
        print(f"  Resultado: {'VERIFIED' if proof.verified else 'FAILED'}")
    else:
        # Degradación graceful
        assert proof.artifact.verification_level.name in ("NONE", "SYMBOLIC")
        print(f"  Lean 4 no disponible: degradación a {proof.artifact.verification_level.name}")

    d = proof.artifact.to_dict()
    json_str = json.dumps(d, default=str)
    assert json_str

    print("FormalVerifier: OK")


@pytest.mark.e2e
def test_lean_result_structure():
    """E2E: LeanResult se crea correctamente."""
    from src.math.formal_verifier import LeanResult

    result = LeanResult(
        success=False,
        source="theorem test : True := sorry",
        stderr="error: declaration uses 'sorry'",
        errors=["declaration uses 'sorry'"],
    )

    assert not result.success
    assert len(result.errors) == 1

    print("LeanResult structure: OK")


# ============================================================
# INTEGRACIÓN: Fases 3-6 juntas
# ============================================================

@pytest.mark.e2e
def test_quantum_with_verification():
    """E2E: Computación cuántica + verificación multi-nivel."""
    from src.math.quantum import QuantumEngine
    from src.math.verification import VerificationPipeline

    qe = QuantumEngine()
    pipeline = VerificationPipeline(engine=qe.engine, max_level=3)

    # Verificar que Pauli X es unitaria Y hermítica
    invariants = qe.verify_quantum_invariants(
        "[[0,1],[1,0]]",
        ["unitary", "hermitian"],
    )

    for name, artifact in invariants.items():
        assert artifact.verification_passed, f"Pauli X debería pasar {name}"
        print(f"  Pauli X {name}: {artifact.verification_level.name} PASS")

    print("Quantum + Verification: OK")


@pytest.mark.e2e
def test_full_pipeline_phases_3_6():
    """E2E: Pipeline completo Fases 3-6 integrado."""
    from src.math.agents import MultiAgentOrchestrator
    from src.math.quantum import QuantumEngine
    from src.math.formula_graph import FormulaGraph
    from src.math.formal_verifier import FormalVerifier
    from src.math.provenance import ProvenanceGraph

    # 1. Multi-agente sin LLM
    orch = MultiAgentOrchestrator(max_steps=2)
    result = orch.run(
        query="Verifica que la puerta Hadamard es unitaria",
        retrieval_context="La puerta Hadamard H = (1/√2)[[1,1],[1,-1]] es una puerta cuántica fundamental."
    )
    assert result["response"]
    assert result["provenance"]["agents"]

    # 2. Motor cuántico
    qe = QuantumEngine()
    h_result = qe.quantum_gate("H")
    assert h_result.success

    # 3. Formula graph
    graph = FormulaGraph()
    fid = graph.add_formula("sin(x)**2 + cos(x)**2", domain="trig")
    assert fid

    # 4. Formal verifier (graceful degradation)
    fv = FormalVerifier()
    proof = fv.verify("sin(x)^2 + cos(x)^2 = 1")
    assert proof.artifact is not None

    print("Full pipeline Phases 3-6: OK")
    print(f"  Multi-agent: {result['steps_verified']}/{result['steps_total']} steps verified")
    print(f"  Quantum: Hadamard gate generated")
    print(f"  Formula graph: {graph.summary()}")
    print(f"  Formal: {proof.artifact.verification_level.name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
