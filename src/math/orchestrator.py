"""
MathComputationOrchestrator — Loop bidireccional LLM ↔ Sandbox.

Implementa el patrón ToRA (Tool-integrated Reasoning):
1. El LLM razona en lenguaje natural
2. Cuando necesita computar, emite bloques <COMPUTE>
3. El orquestador ejecuta el código en el sandbox
4. El resultado se devuelve al LLM como <RESULT>
5. El LLM continúa razonando con el resultado
6. Repite hasta que no haya más <COMPUTE> o se alcance max_iterations
"""

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable, Dict, Any

from .engine import MathEngine, MathResult
from .artifacts import MathArtifact, VerificationLevel

logger = logging.getLogger(__name__)


@dataclass
class ComputationStep:
    """Un paso en el loop de computación."""
    iteration: int
    role: str               # "llm_response", "computation", "result_injection"
    content: str            # Texto del LLM o resultado del sandbox
    code: Optional[str] = None
    math_result: Optional[MathResult] = None
    execution_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "role": self.role,
            "content": self.content[:500],
            "code": self.code,
            "execution_time_ms": self.execution_time_ms,
        }


class MathComputationOrchestrator:
    """
    Orquesta el loop bidireccional LLM ↔ Sandbox para queries matemáticas.

    El LLM genera respuestas que pueden contener bloques <COMPUTE> con
    código Python. El orquestador extrae estos bloques, los ejecuta en
    el sandbox seguro, y alimenta los resultados de vuelta al LLM.
    """

    # Pattern para extraer bloques <COMPUTE>
    COMPUTE_PATTERN = re.compile(
        r'<COMPUTE>\s*```(?:python)?\s*(.*?)\s*```\s*</COMPUTE>',
        re.DOTALL
    )

    # Pattern alternativo sin triple backtick (por si el LLM omite el bloque de código)
    COMPUTE_PATTERN_ALT = re.compile(
        r'<COMPUTE>\s*(.*?)\s*</COMPUTE>',
        re.DOTALL
    )

    def __init__(
        self,
        engine: Optional[MathEngine] = None,
        max_iterations: int = 5,
        timeout_per_step: int = 30,
        wolfram_enabled: bool = False,
        verification_enabled: bool = False,
    ):
        self.engine = engine or MathEngine(timeout=timeout_per_step)
        self.max_iterations = max_iterations
        self.timeout_per_step = timeout_per_step
        self.wolfram_enabled = wolfram_enabled
        self.verification_enabled = verification_enabled

        # Lazy init de componentes opcionales
        self._wolfram = None
        self._verifier = None
        self._quantum = None

        # Leer configuración de settings.yaml
        self._load_config()

    def _load_config(self):
        """Carga configuración desde settings.yaml."""
        try:
            from pathlib import Path
            import yaml
            settings_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
            if settings_path.exists():
                with open(settings_path) as f:
                    settings = yaml.safe_load(f)
                math_cfg = settings.get("math_computation", {})
                self.max_iterations = math_cfg.get("max_iterations", self.max_iterations)
                self.timeout_per_step = math_cfg.get("timeout_per_step_seconds", self.timeout_per_step)
                self.wolfram_enabled = math_cfg.get("wolfram", {}).get("enabled", self.wolfram_enabled)
                self.verification_enabled = math_cfg.get("verification", {}).get("enabled", self.verification_enabled)
        except Exception as e:
            logger.debug(f"No se pudo leer config: {e}")

    @property
    def wolfram(self):
        """Lazy init del cliente Wolfram."""
        if self._wolfram is None and self.wolfram_enabled:
            from .wolfram_client import WolframClient
            self._wolfram = WolframClient()
        return self._wolfram

    @property
    def verifier(self):
        """Lazy init del pipeline de verificación."""
        if self._verifier is None and self.verification_enabled:
            from .verification import VerificationPipeline
            self._verifier = VerificationPipeline(engine=self.engine)
        return self._verifier

    @property
    def quantum(self):
        """Lazy init del motor cuántico."""
        if self._quantum is None:
            from .quantum import QuantumEngine
            self._quantum = QuantumEngine(math_engine=self.engine)
        return self._quantum

    def run(
        self,
        query: str,
        retrieval_context: str,
        system_prompt: str,
        user_template: str,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, List[ComputationStep], List[MathArtifact]]:
        """
        Ejecuta el loop de computación bidireccional.

        Args:
            query: Pregunta del usuario
            retrieval_context: Contexto de fuentes recuperadas (formateado)
            system_prompt: System prompt del template MATHEMATICAL
            user_template: User prompt formateado con contexto
            stream_callback: Callback para streaming (opcional)

        Returns:
            (texto_final_respuesta, pasos_computación, artefactos_matemáticos)
        """
        from src.llm_provider import complete as llm_complete

        steps: List[ComputationStep] = []
        artifacts: List[MathArtifact] = []
        conversation_parts: List[str] = []

        # Acumular tokens para el GeneratedResponse
        total_input_tokens = 0
        total_output_tokens = 0
        model_name = ""

        for iteration in range(self.max_iterations):
            logger.info(f"Iteración {iteration + 1}/{self.max_iterations} del loop de computación")

            # Construir prompt para esta iteración
            if iteration == 0:
                prompt_text = user_template
            else:
                # Continuación: incluir historial de computación
                prompt_text = self._build_continuation(user_template, conversation_parts)

            # Llamar al LLM
            start = time.time()
            llm_response = llm_complete(
                prompt=prompt_text,
                system=system_prompt,
                temperature=0.3,
                max_tokens=4096,
                stream=stream_callback is not None,
                stream_callback=stream_callback,
            )
            llm_time = (time.time() - start) * 1000

            total_input_tokens += llm_response.tokens_input
            total_output_tokens += llm_response.tokens_output
            model_name = llm_response.model

            response_text = llm_response.content

            # Registrar paso del LLM
            steps.append(ComputationStep(
                iteration=iteration,
                role="llm_response",
                content=response_text,
                execution_time_ms=llm_time,
            ))

            # Buscar bloques <COMPUTE>
            compute_blocks = self._extract_compute_blocks(response_text)

            if not compute_blocks:
                # No hay más computación necesaria — respuesta final
                logger.info(f"Loop completado en iteración {iteration + 1}: sin bloques COMPUTE")
                final_text = self._clean_response(response_text)
                return final_text, steps, artifacts

            # Ejecutar cada bloque de código
            results_text = ""
            for block_idx, code in enumerate(compute_blocks):
                logger.info(f"  Ejecutando bloque {block_idx + 1}/{len(compute_blocks)}")

                math_result = self.engine.execute_raw(code)

                # Fallback a Wolfram si SymPy falla y Wolfram está habilitado
                wolfram_used = False
                if not math_result.success and self.wolfram and self.wolfram.available:
                    logger.info("  SymPy falló, intentando Wolfram como fallback...")
                    wolfram_result = self._try_wolfram_fallback(code)
                    if wolfram_result:
                        math_result = wolfram_result
                        wolfram_used = True

                steps.append(ComputationStep(
                    iteration=iteration,
                    role="computation",
                    content=math_result.stdout if math_result.success else (math_result.error or ""),
                    code=code,
                    math_result=math_result,
                    execution_time_ms=math_result.execution_time_ms,
                ))

                # Crear MathArtifact
                artifact = MathArtifact(
                    engine="wolfram" if wolfram_used else "sympy",
                    operation="raw_execution",
                    code=code,
                    result=math_result.output_expr or math_result.stdout,
                    verification_level=VerificationLevel.NONE,
                    verification_passed=math_result.success,
                )
                artifacts.append(artifact)

                # Formatear resultado para devolver al LLM
                if math_result.success:
                    output = math_result.stdout.strip() if math_result.stdout else math_result.output_expr
                    source_tag = " [via Wolfram]" if wolfram_used else ""
                    results_text += f"\n<RESULT>\n{output}\nExecution time: {math_result.execution_time_ms:.0f}ms{source_tag}\n</RESULT>\n"
                else:
                    error_msg = math_result.error or "Error desconocido"
                    results_text += f"\n<RESULT>\nERROR: {error_msg}\n</RESULT>\n"

            # Registrar paso de inyección de resultado
            steps.append(ComputationStep(
                iteration=iteration,
                role="result_injection",
                content=results_text,
            ))

            # Añadir al historial para la siguiente iteración
            conversation_parts.append(response_text)
            conversation_parts.append(results_text)

        # Max iteraciones alcanzadas
        logger.warning(f"Loop alcanzó máximo de {self.max_iterations} iteraciones")
        final_text = self._clean_response(response_text)
        return final_text, steps, artifacts

    def get_token_counts(self, steps: List[ComputationStep]) -> Dict[str, int]:
        """Extrae conteos de tokens acumulados de los pasos."""
        # Los tokens se trackean en el método run() directamente
        return {}

    def _extract_compute_blocks(self, text: str) -> List[str]:
        """Extrae código Python de bloques <COMPUTE>."""
        # Intentar pattern con triple backtick primero
        blocks = self.COMPUTE_PATTERN.findall(text)
        if blocks:
            return [b.strip() for b in blocks if b.strip()]

        # Fallback: pattern sin backtick
        blocks = self.COMPUTE_PATTERN_ALT.findall(text)
        if blocks:
            # Filtrar solo los que parecen código Python
            code_blocks = []
            for b in blocks:
                b = b.strip()
                if b and ("import" in b or "print" in b or "=" in b or "def " in b):
                    code_blocks.append(b)
            return code_blocks

        return []

    def _build_continuation(self, original_prompt: str, conversation_parts: List[str]) -> str:
        """
        Construye prompt de continuación con historial de computación.

        Acumula el contexto original + todos los pares respuesta/resultado previos
        para que el LLM tenga el historial completo.
        """
        parts = [original_prompt, "\n\n--- HISTORIAL DE COMPUTACIÓN ---\n"]

        for i, part in enumerate(conversation_parts):
            if i % 2 == 0:
                parts.append(f"\n[Tu respuesta previa]:\n{part}")
            else:
                parts.append(f"\n[Resultados de ejecución]:\n{part}")

        parts.append("\n\n--- CONTINÚA TU RESPUESTA ---")
        parts.append("\nIncorpora los resultados de la ejecución en tu respuesta. "
                     "Si necesitas más cálculos, usa <COMPUTE> de nuevo. "
                     "Si ya tienes todo lo necesario, da tu respuesta final sin <COMPUTE>.")

        return "\n".join(parts)

    def _clean_response(self, text: str) -> str:
        """
        Limpia la respuesta final eliminando tags <COMPUTE> y <RESULT> residuales.

        Reemplaza los bloques <COMPUTE> por el código formateado como bloque Python
        (para que se muestre al usuario) y elimina los <RESULT> inline.
        """
        # Reemplazar <COMPUTE> blocks por bloques de código normales
        def replace_compute(match):
            code = match.group(1).strip()
            return f"```python\n{code}\n```"

        text = self.COMPUTE_PATTERN.sub(replace_compute, text)
        text = self.COMPUTE_PATTERN_ALT.sub(replace_compute, text)

        # Eliminar <RESULT> blocks (los resultados ya están incorporados en el razonamiento)
        text = re.sub(r'<RESULT>.*?</RESULT>', '', text, flags=re.DOTALL)

        # Limpiar líneas vacías excesivas
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _try_wolfram_fallback(self, code: str) -> Optional[MathResult]:
        """
        Intenta extraer la operación matemática del código y resolverla con Wolfram.

        Parsea el código Python para encontrar la expresión principal y enviarla
        a Wolfram Alpha. Solo funciona para operaciones simples.
        """
        import time

        try:
            # Extraer la expresión principal del código (heurística simple)
            lines = code.strip().split("\n")
            expr_line = None
            for line in reversed(lines):
                line = line.strip()
                if line.startswith("print") or line.startswith("#") or line.startswith("import"):
                    continue
                if "=" in line and not line.startswith("result"):
                    # Línea tipo "resultado = sp.integrate(...)"
                    expr_line = line
                    break

            if not expr_line:
                return None

            # Intentar extraer la expresión Wolfram-friendly
            # Buscar patrones como sp.integrate(expr, x), sp.solve(expr, x), etc.
            import re as _re
            match = _re.search(r'sp\.(integrate|solve|diff|simplify)\((.+)\)', expr_line)
            if not match:
                return None

            operation = match.group(1)
            args = match.group(2)

            start = time.time()
            wolfram_result = self.wolfram.query(f"{operation} {args}")
            elapsed = (time.time() - start) * 1000

            if wolfram_result["success"]:
                return MathResult(
                    success=True,
                    operation=f"wolfram_{operation}",
                    input_expr=args,
                    output_expr=wolfram_result["result"][:500],
                    stdout=wolfram_result["result"],
                    execution_time_ms=elapsed,
                    code_executed=code,
                )
        except Exception as e:
            logger.debug(f"Wolfram fallback falló: {e}")

        return None
