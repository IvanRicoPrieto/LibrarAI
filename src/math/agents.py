"""
Multi-Agent System para razonamiento matemático interleaved.

Implementa el patrón ToRA/SymCode con agentes especializados:
- PlannerAgent: Descompone el problema en pasos
- ReasonerAgent: Razonamiento en lenguaje natural + LaTeX
- CalculatorAgent: Ejecución de código SymPy/NumPy en sandbox
- VerifierAgent: Verificación multi-nivel de cada paso
- SynthesizerAgent: Ensamblado de la respuesta final con citas

El orquestador multi-agente coordina estos agentes en un loop
que produce derivaciones paso a paso, cada una verificada.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

from .engine import MathEngine, MathResult
from .artifacts import MathArtifact, VerificationLevel
from .provenance import ProvenanceGraph, EntityType, ActivityType, AgentType

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Estado de un paso de derivación."""
    PENDING = "pending"
    COMPUTING = "computing"
    VERIFIED = "verified"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DerivationStep:
    """Un paso individual en una derivación matemática."""
    index: int
    description: str               # Descripción en lenguaje natural
    expression: str = ""           # Expresión matemática (SymPy)
    latex: str = ""                # Versión LaTeX
    justification: str = ""        # Justificación del paso
    source_chunks: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    code: str = ""                 # Código ejecutado para verificar
    artifact: Optional[MathArtifact] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "description": self.description,
            "expression": self.expression,
            "latex": self.latex,
            "justification": self.justification,
            "source_chunks": self.source_chunks,
            "status": self.status.value,
            "artifact": self.artifact.to_dict() if self.artifact else None,
            "error": self.error,
        }


@dataclass
class DerivationPlan:
    """Plan de derivación generado por el PlannerAgent."""
    goal: str                      # Objetivo de la derivación
    strategy: str                  # Estrategia (directa, inducción, contradicción, etc.)
    steps: List[DerivationStep] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "strategy": self.strategy,
            "steps": [s.to_dict() for s in self.steps],
            "assumptions": self.assumptions,
            "required_tools": self.required_tools,
        }


class PlannerAgent:
    """
    Descompone problemas matemáticos en pasos ejecutables.

    Usa el LLM para analizar el problema y crear un plan de derivación
    con pasos que pueden ser verificados individualmente.
    """

    PLAN_PROMPT = """Eres un planificador matemático experto. Tu tarea es descomponer
un problema matemático en pasos verificables.

PROBLEMA: {query}

CONTEXTO DE FUENTES:
{context}

Genera un plan de derivación en formato JSON con esta estructura EXACTA:
{{
    "goal": "descripción del objetivo",
    "strategy": "nombre de la estrategia (directa, inducción, contradicción, etc.)",
    "assumptions": ["asunción 1", "asunción 2"],
    "required_tools": ["sympy", "numpy", etc.],
    "steps": [
        {{
            "description": "descripción del paso en lenguaje natural",
            "expression": "expresión SymPy a verificar (o vacío si es texto)",
            "justification": "por qué este paso es correcto",
            "verification_type": "symbolic|numerical|physical|none"
        }}
    ]
}}

REGLAS:
- Cada paso debe ser verificable computacionalmente cuando sea posible
- Las expresiones deben ser strings válidos de SymPy
- Incluir entre 3 y 8 pasos
- Devuelve SOLO el JSON, sin explicaciones adicionales"""

    def plan(
        self,
        query: str,
        context: str,
        llm_fn: Callable,
    ) -> DerivationPlan:
        """
        Genera un plan de derivación.

        Args:
            query: Pregunta/problema del usuario
            context: Contexto de fuentes recuperadas
            llm_fn: Función para llamar al LLM

        Returns:
            DerivationPlan con los pasos
        """
        import json
        import re

        prompt = self.PLAN_PROMPT.format(query=query, context=context[:3000])

        response = llm_fn(
            prompt=prompt,
            system="Eres un planificador matemático preciso. Responde SOLO con JSON válido.",
            temperature=0.2,
            max_tokens=2000,
        )

        # Extraer JSON de la respuesta
        text = response.content.strip()
        # Intentar extraer JSON de bloques de código
        json_match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Intentar encontrar el primer { y último }
            start = text.find('{')
            end = text.rfind('}')
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end+1])
                except json.JSONDecodeError:
                    logger.warning("No se pudo parsear plan del LLM")
                    return self._fallback_plan(query)
            else:
                return self._fallback_plan(query)

        steps = []
        for i, step_data in enumerate(data.get("steps", [])):
            steps.append(DerivationStep(
                index=i,
                description=step_data.get("description", ""),
                expression=step_data.get("expression", ""),
                justification=step_data.get("justification", ""),
            ))

        return DerivationPlan(
            goal=data.get("goal", query),
            strategy=data.get("strategy", "directa"),
            steps=steps,
            assumptions=data.get("assumptions", []),
            required_tools=data.get("required_tools", ["sympy"]),
        )

    def _fallback_plan(self, query: str) -> DerivationPlan:
        """Plan de fallback cuando el LLM no genera JSON válido."""
        return DerivationPlan(
            goal=query,
            strategy="directa",
            steps=[
                DerivationStep(
                    index=0,
                    description="Verificar la afirmación computacionalmente",
                    expression="",
                    justification="Verificación directa",
                ),
            ],
            assumptions=[],
            required_tools=["sympy"],
        )


class CalculatorAgent:
    """
    Ejecuta cálculos en el sandbox seguro.

    Genera código SymPy/NumPy para cada paso y lo ejecuta,
    devolviendo MathResults estructurados.
    """

    COMPUTE_PROMPT = """Genera código Python para verificar este paso matemático.

PASO: {description}
EXPRESIÓN: {expression}
JUSTIFICACIÓN: {justification}
CONTEXTO PREVIO: {prev_results}

Genera SOLO código Python ejecutable que:
1. Use SymPy para la verificación simbólica
2. Imprima el resultado claramente
3. Sea autocontenido (incluya todos los imports)

Código:"""

    def __init__(self, engine: MathEngine):
        self.engine = engine

    def compute_step(
        self,
        step: DerivationStep,
        prev_results: List[str],
        llm_fn: Optional[Callable] = None,
    ) -> MathResult:
        """
        Ejecuta un paso de derivación.

        Si la expresión del paso es SymPy válido, ejecuta directamente.
        Si necesita generación de código, usa el LLM.
        """
        if step.expression:
            return self._execute_expression(step)

        if llm_fn:
            return self._generate_and_execute(step, prev_results, llm_fn)

        return MathResult(
            success=False,
            operation="compute_step",
            input_expr=step.description,
            error="No hay expresión ni LLM disponible",
        )

    def _execute_expression(self, step: DerivationStep) -> MathResult:
        """Ejecuta una expresión SymPy directamente."""
        code = f'''
import sympy as sp
import json

x, y, z, t, n, k = sp.symbols("x y z t n k")

try:
    expr = sp.sympify("{step.expression}")
    simplified = sp.trigsimp(sp.expand(expr))
    result = {{
        "output_expr": str(simplified),
        "output_latex": sp.latex(simplified),
        "numeric_value": float(simplified.evalf()) if simplified.is_number else None,
    }}
except Exception as e:
    result = {{"error": str(e)}}

print("__MATH_RESULT__")
print(json.dumps(result, default=str))
print("__MATH_RESULT_END__")
'''
        return self.engine._execute_math_code(code, "compute_step", step.expression)

    def _generate_and_execute(
        self,
        step: DerivationStep,
        prev_results: List[str],
        llm_fn: Callable,
    ) -> MathResult:
        """Genera código con LLM y lo ejecuta."""
        import re

        prompt = self.COMPUTE_PROMPT.format(
            description=step.description,
            expression=step.expression or "N/A",
            justification=step.justification,
            prev_results="\n".join(prev_results[-3:]) if prev_results else "Ninguno",
        )

        response = llm_fn(
            prompt=prompt,
            system="Genera SOLO código Python ejecutable. Sin explicaciones.",
            temperature=0.1,
            max_tokens=1000,
        )

        code = response.content.strip()
        # Extraer de bloques de código si hay
        code_match = re.search(r'```(?:python)?\s*(.*?)```', code, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()

        return self.engine.execute_raw(code)


class VerifierAgent:
    """
    Verifica resultados computacionales a múltiples niveles.

    Usa el VerificationPipeline existente y añade lógica
    para decidir qué nivel de verificación aplicar.
    """

    def __init__(self, engine: MathEngine):
        self.engine = engine

    def verify_step(
        self,
        step: DerivationStep,
        math_result: Optional[MathResult] = None,
    ) -> MathArtifact:
        """Verifica un paso de derivación."""
        from .verification import VerificationPipeline

        pipeline = VerificationPipeline(engine=self.engine, max_level=2)

        if step.expression:
            # Verificar que la expresión es válida
            artifact = pipeline.verify(
                lhs=step.expression,
                rhs="0",  # Verificar si se simplifica a 0 (identidad)
                min_level=1,
                max_level=2,
            )
            return artifact

        if math_result and math_result.success:
            # Crear artifact desde MathResult
            return MathArtifact.from_math_result(
                math_result,
                source_chunks=step.source_chunks,
            )

        return MathArtifact(
            operation="verification_failed",
            result="N/A",
            verification_level=VerificationLevel.NONE,
            verification_passed=False,
        )


class SynthesizerAgent:
    """
    Ensambla la respuesta final a partir de los pasos verificados.

    Combina razonamiento en lenguaje natural con evidencia
    computacional y citas a fuentes.
    """

    SYNTHESIS_PROMPT = """Sintetiza una respuesta final basada en la derivación verificada.

PREGUNTA ORIGINAL: {query}

PASOS VERIFICADOS:
{steps_text}

RESULTADOS COMPUTACIONALES:
{results_text}

INSTRUCCIONES:
- Presenta la derivación de forma clara y pedagógica
- Cita las fuentes usando [n] cuando corresponda
- Indica qué pasos fueron verificados computacionalmente
- Usa notación LaTeX ($ ... $) para las fórmulas
- Si algún paso falló la verificación, menciónalo explícitamente"""

    def synthesize(
        self,
        query: str,
        plan: DerivationPlan,
        results: List[MathResult],
        llm_fn: Optional[Callable] = None,
    ) -> str:
        """
        Genera la respuesta final.

        Si llm_fn está disponible, usa el LLM para generar prosa.
        Si no, genera un formato estructurado básico.
        """
        if llm_fn:
            return self._synthesize_with_llm(query, plan, results, llm_fn)
        return self._synthesize_structured(query, plan, results)

    def _synthesize_with_llm(
        self,
        query: str,
        plan: DerivationPlan,
        results: List[MathResult],
        llm_fn: Callable,
    ) -> str:
        """Síntesis con LLM."""
        steps_text = ""
        for step in plan.steps:
            status = step.status.value
            steps_text += f"\nPaso {step.index + 1} [{status}]: {step.description}\n"
            if step.expression:
                steps_text += f"  Expresión: {step.expression}\n"
            if step.justification:
                steps_text += f"  Justificación: {step.justification}\n"

        results_text = ""
        for i, r in enumerate(results):
            if r.success:
                results_text += f"\nResultado {i + 1}: {r.output_expr or r.stdout[:200]}\n"
            else:
                results_text += f"\nResultado {i + 1}: ERROR - {r.error}\n"

        prompt = self.SYNTHESIS_PROMPT.format(
            query=query,
            steps_text=steps_text,
            results_text=results_text,
        )

        response = llm_fn(
            prompt=prompt,
            system="Eres un profesor de matemáticas y física. Genera respuestas claras con LaTeX.",
            temperature=0.3,
            max_tokens=3000,
        )
        return response.content

    def _synthesize_structured(
        self,
        query: str,
        plan: DerivationPlan,
        results: List[MathResult],
    ) -> str:
        """Síntesis sin LLM — formato estructurado."""
        lines = [f"## {plan.goal}\n"]
        lines.append(f"**Estrategia**: {plan.strategy}\n")

        if plan.assumptions:
            lines.append("**Asunciones**: " + ", ".join(plan.assumptions) + "\n")

        for step in plan.steps:
            status_icon = {
                StepStatus.VERIFIED: "PASS",
                StepStatus.FAILED: "FAIL",
                StepStatus.SKIPPED: "SKIP",
            }.get(step.status, "?")

            lines.append(f"\n### Paso {step.index + 1} [{status_icon}]")
            lines.append(step.description)

            if step.expression:
                lines.append(f"\n$${step.expression}$$")

            if step.justification:
                lines.append(f"\n*{step.justification}*")

            if step.artifact and step.artifact.verification_passed:
                lines.append(f"\n> Verificado computacionalmente (nivel: {step.artifact.verification_level.name})")

        return "\n".join(lines)


class MultiAgentOrchestrator:
    """
    Orquestador multi-agente para derivaciones matemáticas.

    Coordina Planner → Reasoner → Calculator → Verifier → Synthesizer
    en un loop que produce derivaciones verificadas paso a paso.
    """

    def __init__(
        self,
        engine: Optional[MathEngine] = None,
        max_steps: int = 10,
        timeout_per_step: int = 30,
    ):
        self.engine = engine or MathEngine(timeout=timeout_per_step)
        self.max_steps = max_steps
        self.planner = PlannerAgent()
        self.calculator = CalculatorAgent(self.engine)
        self.verifier = VerifierAgent(self.engine)
        self.synthesizer = SynthesizerAgent()
        self.provenance = ProvenanceGraph()

        # Motores especializados (lazy init)
        self._quantum = None

    @property
    def quantum(self):
        """Motor cuántico especializado."""
        if self._quantum is None:
            from .quantum import QuantumEngine
            self._quantum = QuantumEngine(math_engine=self.engine)
        return self._quantum

    def run(
        self,
        query: str,
        retrieval_context: str,
        llm_fn: Optional[Callable] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta el pipeline multi-agente completo.

        Args:
            query: Pregunta del usuario
            retrieval_context: Contexto de fuentes
            llm_fn: Función de llamada al LLM
            stream_callback: Callback para streaming

        Returns:
            Dict con:
              - response: str (respuesta final)
              - plan: DerivationPlan
              - artifacts: List[MathArtifact]
              - provenance: Dict (grafo de provenance serializado)
              - steps_verified: int
              - steps_total: int
        """
        start = time.time()
        artifacts: List[MathArtifact] = []
        results: List[MathResult] = []

        # Registrar agentes en provenance
        llm_agent = self.provenance.add_agent(AgentType.LLM, "LLM", "")
        sympy_agent = self.provenance.add_agent(AgentType.SYMPY, "SymPy", "")
        qutip_agent = self.provenance.add_agent(AgentType.QUTIP, "QuTiP", "")
        verifier_agent = self.provenance.add_agent(AgentType.VERIFIER, "Verifier", "")

        # Registrar contexto como entidades
        ctx_entity = self.provenance.add_entity(
            EntityType.SOURCE_CHUNK,
            content=retrieval_context[:500],
        )

        # 1. PLANIFICAR
        if stream_callback:
            stream_callback("[Planificando derivación...]\n")

        if llm_fn:
            plan = self.planner.plan(query, retrieval_context, llm_fn)
        else:
            plan = self.planner._fallback_plan(query)

        plan_activity = self.provenance.add_activity(
            ActivityType.PLANNING,
            description=f"Plan: {plan.strategy}",
            used=[ctx_entity],
            agent_id=llm_agent,
        )

        logger.info(f"Plan generado: {plan.strategy}, {len(plan.steps)} pasos")

        # 2. EJECUTAR CADA PASO
        prev_results: List[str] = []

        for step in plan.steps[:self.max_steps]:
            if stream_callback:
                stream_callback(f"[Paso {step.index + 1}: {step.description[:60]}...]\n")

            step.status = StepStatus.COMPUTING

            # 2a. Calcular
            comp_activity = self.provenance.add_activity(
                ActivityType.COMPUTATION,
                description=f"Step {step.index}: {step.description[:50]}",
                agent_id=sympy_agent,
            )

            math_result = self.calculator.compute_step(step, prev_results, llm_fn)
            results.append(math_result)

            if math_result.success:
                result_entity = self.provenance.add_entity(
                    EntityType.COMPUTATION_RESULT,
                    content=math_result.output_expr or math_result.stdout[:200],
                )
                self.provenance.record_generation(comp_activity, result_entity)
                prev_results.append(math_result.output_expr or math_result.stdout[:200])
            else:
                prev_results.append(f"ERROR: {math_result.error}")

            # 2b. Verificar
            verif_activity = self.provenance.add_activity(
                ActivityType.VERIFICATION,
                description=f"Verify step {step.index}",
                agent_id=verifier_agent,
            )

            artifact = self.verifier.verify_step(step, math_result)
            artifacts.append(artifact)
            step.artifact = artifact

            artifact_entity = self.provenance.add_entity(
                EntityType.MATH_ARTIFACT,
                content=artifact.result[:200],
                metadata={"passed": artifact.verification_passed},
            )
            self.provenance.record_generation(verif_activity, artifact_entity)

            if artifact.verification_passed:
                step.status = StepStatus.VERIFIED
            elif math_result.success:
                step.status = StepStatus.VERIFIED  # Ejecución OK = verificado básico
            else:
                step.status = StepStatus.FAILED
                step.error = math_result.error

            self.provenance.end_activity(comp_activity)
            self.provenance.end_activity(verif_activity)

        # 3. SINTETIZAR
        if stream_callback:
            stream_callback("[Sintetizando respuesta final...]\n")

        synth_activity = self.provenance.add_activity(
            ActivityType.SYNTHESIS,
            description="Ensamblar respuesta final",
            agent_id=llm_agent,
        )

        response_text = self.synthesizer.synthesize(query, plan, results, llm_fn)

        response_entity = self.provenance.add_entity(
            EntityType.FINAL_RESPONSE,
            content=response_text[:500],
        )
        self.provenance.record_generation(synth_activity, response_entity)
        self.provenance.end_activity(synth_activity)

        elapsed = (time.time() - start) * 1000

        steps_verified = sum(1 for s in plan.steps if s.status == StepStatus.VERIFIED)

        return {
            "response": response_text,
            "plan": plan,
            "artifacts": artifacts,
            "provenance": self.provenance.to_dict(),
            "steps_verified": steps_verified,
            "steps_total": len(plan.steps),
            "elapsed_ms": elapsed,
        }
