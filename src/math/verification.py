"""
VerificationPipeline — Verificación multi-nivel de afirmaciones matemáticas.

Pipeline secuencial de 5 niveles, de menor a mayor rigor:
- Nivel 0 (Dimensional): Verifica consistencia dimensional con Pint
- Nivel 1 (Numérico): Sampling aleatorio con NumPy/mpmath
- Nivel 2 (Simbólico): Transforms específicos de SymPy (NO simplify genérico)
- Nivel 3 (Físico): Invariantes físicos (unitariedad, traza, conservación)
- Nivel 5 (Formal): Verificación formal con Lean 4 (requiere Lean instalado)

Cada nivel produce un dict con los detalles de la verificación.
Se puede ejecutar un solo nivel o un rango.
"""

import logging
from typing import Dict, Any, Optional, List

from .engine import MathEngine, MathResult
from .artifacts import MathArtifact, VerificationLevel

logger = logging.getLogger(__name__)


class VerificationPipeline:
    """
    Pipeline de verificación multi-nivel para expresiones matemáticas.

    Ejecuta verificaciones progresivamente más rigurosas. Se detiene
    al primer nivel que da un resultado definitivo (pass o fail claro),
    o sube al siguiente nivel si el resultado es inconcluso.
    """

    def __init__(self, engine: Optional[MathEngine] = None, max_level: int = 5):
        self.engine = engine or MathEngine()
        self.max_level = min(max_level, 5)
        self._formal_verifier = None

    def verify(
        self,
        lhs: str,
        rhs: str,
        variables: Optional[List[str]] = None,
        min_level: int = 0,
        max_level: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MathArtifact:
        """
        Verifica si lhs == rhs usando el pipeline multi-nivel.

        Args:
            lhs: Lado izquierdo (expresión SymPy como string)
            rhs: Lado derecho
            variables: Variables libres (auto-detectadas si None)
            min_level: Nivel mínimo de verificación
            max_level: Nivel máximo
            context: Contexto adicional (unidades esperadas, tipo de invariante, etc.)

        Returns:
            MathArtifact con resultados de verificación
        """
        if max_level is None:
            max_level = self.max_level

        context = context or {}
        all_details: Dict[str, Any] = {}
        passed = False
        highest_level = VerificationLevel.NONE

        # Nivel 0: Dimensional (si hay información de unidades)
        if min_level <= 0 <= max_level and context.get("units"):
            result = self._verify_dimensional(
                lhs, rhs, context.get("expected_units", "")
            )
            all_details["dimensional"] = result
            if result.get("conclusive"):
                passed = result["passed"]
                highest_level = VerificationLevel.DIMENSIONAL
                if not passed:
                    return self._build_artifact(lhs, rhs, passed, highest_level, all_details)

        # Nivel 1: Numérico
        if min_level <= 1 <= max_level:
            result = self._verify_numerical(lhs, rhs, variables)
            all_details["numerical"] = result
            if result.get("conclusive"):
                passed = result["passed"]
                highest_level = VerificationLevel.NUMERICAL
                if not passed:
                    return self._build_artifact(lhs, rhs, passed, highest_level, all_details)

        # Nivel 2: Simbólico
        if min_level <= 2 <= max_level:
            result = self._verify_symbolic(lhs, rhs, variables)
            all_details["symbolic"] = result
            if result.get("conclusive"):
                passed = result["passed"]
                highest_level = VerificationLevel.SYMBOLIC

        # Nivel 3: Invariantes físicos
        if min_level <= 3 <= max_level and context.get("invariant_type"):
            result = self._verify_physical(
                lhs, context["invariant_type"]
            )
            all_details["physical"] = result
            if result.get("conclusive"):
                passed = result["passed"]
                highest_level = VerificationLevel.PHYSICAL

        # Nivel 5: Verificación formal (Lean 4)
        if min_level <= 5 <= max_level and passed and context.get("formal_statement"):
            result = self._verify_formal(
                context["formal_statement"],
                context.get("formal_context", ""),
                context.get("llm_fn"),
            )
            all_details["formal"] = result
            if result.get("conclusive"):
                passed = result["passed"]
                highest_level = VerificationLevel.FORMAL

        return self._build_artifact(lhs, rhs, passed, highest_level, all_details)

    def _verify_dimensional(
        self,
        lhs: str,
        rhs: str,
        expected_units: str,
    ) -> Dict[str, Any]:
        """Nivel 0: Verificación dimensional con Pint."""
        code = f'''
import pint
import json

ureg = pint.UnitRegistry()

try:
    lhs_qty = ureg.parse_expression("{lhs}")
    rhs_qty = ureg.parse_expression("{rhs}")

    dims_match = lhs_qty.dimensionality == rhs_qty.dimensionality
    result = {{
        "passed": dims_match,
        "conclusive": True,
        "lhs_dims": str(lhs_qty.dimensionality),
        "rhs_dims": str(rhs_qty.dimensionality),
    }}
except Exception as e:
    result = {{"passed": False, "conclusive": False, "error": str(e)}}

print("__MATH_RESULT__")
print(json.dumps(result))
print("__MATH_RESULT_END__")
'''
        math_result = self.engine._execute_math_code(code, "dimensional_check", f"{lhs} vs {rhs}")
        data = self.engine._extract_math_result(math_result.stdout) if math_result.success else None
        return data or {"passed": False, "conclusive": False, "error": math_result.error}

    def _verify_numerical(
        self,
        lhs: str,
        rhs: str,
        variables: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Nivel 1: Verificación numérica por sampling aleatorio."""
        vars_code = ""
        if variables:
            for v in variables:
                vars_code += f'    {v} = sp.Symbol("{v}")\n'
        else:
            vars_code = '    pass  # auto-detect free symbols\n'

        code = f'''
import sympy as sp
import numpy as np
import json

try:
{vars_code}
    left = sp.sympify("{lhs}")
    right = sp.sympify("{rhs}")
    diff = left - right

    free_vars = list(diff.free_symbols)

    if not free_vars:
        # Expresión constante
        val = complex(diff.evalf())
        passed = abs(val) < 1e-10
        result = {{
            "passed": passed,
            "conclusive": True,
            "method": "constant_evaluation",
            "residual": abs(val),
            "test_points": 1,
        }}
    else:
        # Sampling con 100 puntos
        max_residual = 0.0
        failures = 0
        n_tests = 100

        for _ in range(n_tests):
            point = {{v: np.random.uniform(-10, 10) for v in free_vars}}
            try:
                val = complex(diff.subs(point))
                residual = abs(val)
                max_residual = max(max_residual, residual)
                if residual > 1e-8:
                    failures += 1
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                continue

        passed = failures == 0
        result = {{
            "passed": passed,
            "conclusive": True if failures > 3 else (True if passed else False),
            "method": "random_sampling",
            "test_points": n_tests,
            "failures": failures,
            "max_residual": float(max_residual),
        }}
except Exception as e:
    result = {{"passed": False, "conclusive": False, "error": str(e)}}

print("__MATH_RESULT__")
print(json.dumps(result, default=str))
print("__MATH_RESULT_END__")
'''
        math_result = self.engine._execute_math_code(code, "numerical_verify", f"{lhs} == {rhs}")
        data = self.engine._extract_math_result(math_result.stdout) if math_result.success else None
        return data or {"passed": False, "conclusive": False, "error": math_result.error}

    def _verify_symbolic(
        self,
        lhs: str,
        rhs: str,
        variables: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Nivel 2: Verificación simbólica con transforms específicos.

        Usa expand, factor, trigsimp, cancel, together, powsimp.
        NO usa simplify() genérico (no determinista entre versiones).
        """
        code = f'''
import sympy as sp
import json

try:
    left = sp.sympify("{lhs}")
    right = sp.sympify("{rhs}")
    diff = left - right

    # Probar transforms específicos (NO simplify genérico)
    transforms = [
        ("expand", sp.expand),
        ("factor", sp.factor),
        ("cancel", sp.cancel),
        ("trigsimp", sp.trigsimp),
        ("together", sp.together),
        ("powsimp", sp.powsimp),
        ("radsimp", sp.radsimp),
        ("logcombine", sp.logcombine),
    ]

    passed = False
    method_used = "none"
    for name, transform in transforms:
        try:
            if transform(diff) == 0:
                passed = True
                method_used = name
                break
        except Exception:
            continue

    # Si ningún transform funciona, intentar como último recurso
    if not passed:
        try:
            if diff.equals(0):
                passed = True
                method_used = "equals"
        except Exception:
            pass

    result = {{
        "passed": passed,
        "conclusive": True,
        "method": method_used,
        "transforms_tried": [t[0] for t in transforms],
    }}
except Exception as e:
    result = {{"passed": False, "conclusive": False, "error": str(e)}}

print("__MATH_RESULT__")
print(json.dumps(result, default=str))
print("__MATH_RESULT_END__")
'''
        math_result = self.engine._execute_math_code(code, "symbolic_verify", f"{lhs} == {rhs}")
        data = self.engine._extract_math_result(math_result.stdout) if math_result.success else None
        return data or {"passed": False, "conclusive": False, "error": math_result.error}

    def _verify_physical(
        self,
        expr: str,
        invariant_type: str,
    ) -> Dict[str, Any]:
        """
        Nivel 3: Verificación de invariantes físicos.

        Tipos soportados:
        - "unitary": U†U = I (matrices unitarias)
        - "hermitian": H = H† (operadores hermíticos)
        - "trace_one": Tr(ρ) = 1 (matrices de densidad)
        - "positive_semidefinite": eigenvalores ≥ 0
        - "normalized": ⟨ψ|ψ⟩ = 1 (estados cuánticos)
        """
        invariant_checks = {
            "unitary": f'''
    M = sp.Matrix({expr})
    product = sp.simplify(M * M.H - sp.eye(M.rows))
    passed = product.is_zero_matrix if product.is_zero_matrix is not None else False
    details = "U†U = I check"
''',
            "hermitian": f'''
    M = sp.Matrix({expr})
    diff_mat = sp.simplify(M - M.H)
    passed = diff_mat.is_zero_matrix if diff_mat.is_zero_matrix is not None else False
    details = "H = H† check"
''',
            "trace_one": f'''
    M = sp.Matrix({expr})
    tr = sp.trace(M)
    passed = sp.simplify(tr - 1) == 0
    details = f"Tr(ρ) = {{tr}} (expected 1)"
''',
            "positive_semidefinite": f'''
    M = sp.Matrix({expr})
    eigenvals = list(M.eigenvals().keys())
    passed = all(sp.re(ev) >= 0 for ev in eigenvals)
    details = f"Eigenvalues: {{eigenvals}}"
''',
            "normalized": f'''
    v = sp.Matrix({expr})
    norm_sq = sp.simplify(v.H * v)
    passed = norm_sq == sp.Matrix([[1]])
    details = f"⟨ψ|ψ⟩ = {{norm_sq[0,0]}}"
''',
        }

        if invariant_type not in invariant_checks:
            return {
                "passed": False,
                "conclusive": False,
                "error": f"Invariante no soportado: {invariant_type}",
            }

        code = f'''
import sympy as sp
import json

try:
{invariant_checks[invariant_type]}
    result = {{
        "passed": bool(passed),
        "conclusive": True,
        "invariant": "{invariant_type}",
        "details": str(details),
    }}
except Exception as e:
    result = {{"passed": False, "conclusive": False, "error": str(e)}}

print("__MATH_RESULT__")
print(json.dumps(result, default=str))
print("__MATH_RESULT_END__")
'''
        math_result = self.engine._execute_math_code(code, f"physical_{invariant_type}", expr)
        data = self.engine._extract_math_result(math_result.stdout) if math_result.success else None
        return data or {"passed": False, "conclusive": False, "error": math_result.error}

    def _verify_formal(
        self,
        statement: str,
        context: str = "",
        llm_fn=None,
    ) -> Dict[str, Any]:
        """
        Nivel 5: Verificación formal con Lean 4.

        Usa FormalVerifier para autoformalize + compile + repair.
        Si Lean 4 no está disponible, retorna inconcluso.
        """
        try:
            if self._formal_verifier is None:
                from .formal_verifier import FormalVerifier
                self._formal_verifier = FormalVerifier()

            proof = self._formal_verifier.verify(
                statement=statement,
                context=context,
                llm_fn=llm_fn,
            )

            if not self._formal_verifier.available:
                return {
                    "passed": False,
                    "conclusive": False,
                    "lean_available": False,
                    "error": "Lean 4 no disponible en el sistema",
                }

            return {
                "passed": proof.verified,
                "conclusive": True,
                "lean_available": True,
                "lean_code": proof.lean_code,
                "attempts": proof.attempts,
                "errors": proof.lean_result.errors if proof.lean_result else [],
                "execution_time_ms": proof.lean_result.execution_time_ms if proof.lean_result else 0,
            }
        except Exception as e:
            logger.warning(f"Verificación formal falló: {e}")
            return {"passed": False, "conclusive": False, "error": str(e)}

    def _build_artifact(
        self,
        lhs: str,
        rhs: str,
        passed: bool,
        level: VerificationLevel,
        details: Dict[str, Any],
    ) -> MathArtifact:
        """Construye un MathArtifact con los resultados de la verificación."""
        return MathArtifact(
            input_sympy=f"{lhs} == {rhs}",
            engine="sympy",
            operation="verification",
            result=str(passed),
            verification_level=level,
            verification_passed=passed,
            verification_details=details,
        )
