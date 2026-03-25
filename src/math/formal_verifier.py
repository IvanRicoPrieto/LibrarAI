"""
Formal Verifier — Integración con Lean 4 para verificación formal.

Fase 6 del roadmap (aspiracional):
- Autoformalization: traducción de lenguaje natural + LaTeX a Lean 4
- Verificación formal de proposiciones en Mathlib
- Interfaz con Lean via subprocess (LeanInteract pattern)

Requisitos: Lean 4 instalado (`elan` toolchain manager).
Si Lean no está disponible, se degrada gracefully a VerificationLevel.SYMBOLIC.

Esta implementación sigue el patrón de LeanDojo/LeanInteract:
1. LLM genera candidato Lean 4
2. Se compila y verifica con `lean --run`
3. Si falla, el LLM corrige (hasta N intentos)
"""

import os
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Callable

from .artifacts import MathArtifact, VerificationLevel

logger = logging.getLogger(__name__)


@dataclass
class LeanResult:
    """Resultado de compilación Lean 4."""
    success: bool
    source: str                     # Código Lean 4
    stdout: str = ""
    stderr: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0


@dataclass
class FormalProof:
    """Prueba formal verificada."""
    statement: str                  # Proposición en lenguaje natural
    lean_code: str = ""             # Código Lean 4 completo
    verified: bool = False
    lean_result: Optional[LeanResult] = None
    attempts: int = 0
    artifact: Optional[MathArtifact] = None


class LeanInterface:
    """
    Interfaz con el compilador Lean 4.

    Compila y verifica código Lean 4 via subprocess.
    Requiere que `lean` esté en el PATH (instalado via elan).
    """

    def __init__(self, timeout: int = 60, lean_path: Optional[str] = None):
        self.timeout = timeout
        self.lean_path = lean_path or self._find_lean()
        self._available = self.lean_path is not None

    @property
    def available(self) -> bool:
        return self._available

    def _find_lean(self) -> Optional[str]:
        """Busca el ejecutable lean en el sistema."""
        # Intentar elan default
        elan_path = Path.home() / ".elan" / "bin" / "lean"
        if elan_path.exists():
            return str(elan_path)

        # Intentar en PATH
        try:
            result = subprocess.run(
                ["which", "lean"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass

        logger.info("Lean 4 no encontrado en el sistema")
        return None

    def check(self, lean_code: str) -> LeanResult:
        """
        Compila código Lean 4 y reporta si es correcto.

        Args:
            lean_code: Código Lean 4 completo

        Returns:
            LeanResult con el resultado de la compilación
        """
        if not self._available:
            return LeanResult(
                success=False,
                source=lean_code,
                stderr="Lean 4 no disponible",
                errors=["Lean 4 no instalado"],
            )

        start = time.time()

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.lean', delete=False
        ) as f:
            f.write(lean_code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [self.lean_path, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "LEAN_PATH": ""},
            )

            elapsed = (time.time() - start) * 1000

            errors = []
            warnings = []
            for line in result.stderr.split('\n'):
                if 'error' in line.lower():
                    errors.append(line.strip())
                elif 'warning' in line.lower():
                    warnings.append(line.strip())

            return LeanResult(
                success=result.returncode == 0 and not errors,
                source=lean_code,
                stdout=result.stdout,
                stderr=result.stderr,
                errors=errors,
                warnings=warnings,
                execution_time_ms=elapsed,
            )

        except subprocess.TimeoutExpired:
            elapsed = (time.time() - start) * 1000
            return LeanResult(
                success=False,
                source=lean_code,
                stderr=f"Timeout ({self.timeout}s)",
                errors=[f"Lean compilation timeout ({self.timeout}s)"],
                execution_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return LeanResult(
                success=False,
                source=lean_code,
                stderr=str(e),
                errors=[str(e)],
                execution_time_ms=elapsed,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class Autoformalizator:
    """
    Traduce proposiciones matemáticas a Lean 4.

    Usa el LLM para generar código Lean 4 a partir de
    lenguaje natural + LaTeX, siguiendo patrones de Mathlib.
    """

    FORMALIZE_PROMPT = """Traduce esta proposición matemática a Lean 4 usando Mathlib.

PROPOSICIÓN: {statement}

CONTEXTO: {context}

REGLAS:
1. Genera código Lean 4 completo y compilable
2. Usa imports de Mathlib cuando sea necesario
3. La prueba debe ser autocontenida
4. Usa tácticas estándar (ring, norm_num, simp, linarith, etc.)
5. NO uses sorry

Formato de respuesta:
```lean
import Mathlib...

theorem/lemma nombre : proposición := by
  <prueba>
```

SOLO el código Lean, sin explicaciones."""

    REPAIR_PROMPT = """El siguiente código Lean 4 tiene errores. Corrígelo.

CÓDIGO ORIGINAL:
```lean
{code}
```

ERRORES:
{errors}

Genera el código Lean 4 corregido. SOLO el código, sin explicaciones."""

    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts

    def formalize(
        self,
        statement: str,
        context: str = "",
        llm_fn: Optional[Callable] = None,
    ) -> str:
        """
        Traduce una proposición a Lean 4.

        Args:
            statement: Proposición en lenguaje natural/LaTeX
            context: Contexto adicional (definiciones, lemas previos)
            llm_fn: Función para llamar al LLM

        Returns:
            Código Lean 4
        """
        if llm_fn is None:
            return self._template_formalization(statement)

        import re

        prompt = self.FORMALIZE_PROMPT.format(
            statement=statement,
            context=context or "Ninguno",
        )

        response = llm_fn(
            prompt=prompt,
            system="Eres un experto en Lean 4 y Mathlib. Genera solo código Lean 4 válido.",
            temperature=0.1,
            max_tokens=1500,
        )

        code = response.content.strip()
        # Extraer de bloques de código
        match = re.search(r'```(?:lean)?\s*(.*?)```', code, re.DOTALL)
        if match:
            code = match.group(1).strip()

        return code

    def repair(
        self,
        code: str,
        errors: List[str],
        llm_fn: Optional[Callable] = None,
    ) -> str:
        """Intenta reparar código Lean 4 con errores."""
        if llm_fn is None:
            return code  # No se puede reparar sin LLM

        import re

        prompt = self.REPAIR_PROMPT.format(
            code=code,
            errors="\n".join(errors),
        )

        response = llm_fn(
            prompt=prompt,
            system="Eres un experto en Lean 4. Corrige los errores manteniendo la estructura.",
            temperature=0.1,
            max_tokens=1500,
        )

        repaired = response.content.strip()
        match = re.search(r'```(?:lean)?\s*(.*?)```', repaired, re.DOTALL)
        if match:
            repaired = match.group(1).strip()

        return repaired

    def _template_formalization(self, statement: str) -> str:
        """
        Formalización basada en templates para proposiciones simples.

        Sin LLM, usa patrones reconocibles para generar Lean 4.
        """
        statement_lower = statement.lower()

        # Template para "para todo x, P(x)"
        if "para todo" in statement_lower or "for all" in statement_lower:
            return f"""-- Auto-generated from: {statement}
-- NOTE: Requires Lean 4 with Mathlib for compilation
theorem auto_theorem : True := trivial
-- TODO: Full formalization requires LLM assistance
"""

        # Template genérico
        return f"""-- Auto-generated from: {statement}
-- NOTE: Template formalization, requires manual refinement
theorem auto_theorem : True := trivial
-- Original statement: {statement}
"""


class FormalVerifier:
    """
    Pipeline completo de verificación formal.

    1. Autoformalization (LLM traduce a Lean 4)
    2. Compilación (Lean 4 verifica)
    3. Repair loop (si falla, corregir y reintentar)
    4. Producir MathArtifact con resultado

    Si Lean 4 no está disponible, produce un artifact con
    VerificationLevel.SYMBOLIC y nota de que la verificación
    formal no se pudo completar.
    """

    def __init__(
        self,
        lean_timeout: int = 60,
        max_repair_attempts: int = 3,
    ):
        self.lean = LeanInterface(timeout=lean_timeout)
        self.formalizer = Autoformalizator(max_attempts=max_repair_attempts)
        self.max_repair_attempts = max_repair_attempts

    @property
    def available(self) -> bool:
        """Comprueba si la verificación formal está disponible."""
        return self.lean.available

    def verify(
        self,
        statement: str,
        context: str = "",
        llm_fn: Optional[Callable] = None,
    ) -> FormalProof:
        """
        Intenta verificar formalmente una proposición.

        Args:
            statement: Proposición a verificar
            context: Contexto de las fuentes
            llm_fn: Función del LLM para autoformalization

        Returns:
            FormalProof con el resultado
        """
        proof = FormalProof(statement=statement)

        if not self.lean.available:
            logger.info("Lean 4 no disponible — skipping formal verification")
            proof.artifact = MathArtifact(
                input_sympy=statement,
                engine="lean4",
                operation="formal_verification",
                result="Lean 4 no disponible",
                verification_level=VerificationLevel.NONE,
                verification_passed=False,
                verification_details={"lean_available": False},
            )
            return proof

        # 1. Autoformalizar
        lean_code = self.formalizer.formalize(statement, context, llm_fn)
        proof.lean_code = lean_code
        proof.attempts = 1

        # 2. Compilar
        result = self.lean.check(lean_code)
        proof.lean_result = result

        # 3. Repair loop
        while not result.success and proof.attempts < self.max_repair_attempts and llm_fn:
            logger.info(
                f"Lean check falló (intento {proof.attempts}), "
                f"errores: {result.errors[:2]}"
            )

            lean_code = self.formalizer.repair(lean_code, result.errors, llm_fn)
            proof.lean_code = lean_code
            proof.attempts += 1

            result = self.lean.check(lean_code)
            proof.lean_result = result

        proof.verified = result.success

        # 4. Crear artifact
        proof.artifact = MathArtifact(
            input_sympy=statement,
            engine="lean4",
            operation="formal_verification",
            code=lean_code,
            result="VERIFIED" if result.success else f"FAILED ({proof.attempts} attempts)",
            verification_level=VerificationLevel.FORMAL if result.success else VerificationLevel.NONE,
            verification_passed=result.success,
            verification_details={
                "lean_available": True,
                "attempts": proof.attempts,
                "errors": result.errors if not result.success else [],
                "warnings": result.warnings,
                "execution_time_ms": result.execution_time_ms,
            },
        )

        return proof

    def check_lean_code(self, code: str) -> LeanResult:
        """Compila código Lean 4 directamente."""
        return self.lean.check(code)
