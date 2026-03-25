"""
MathEngine — Operaciones matemáticas estructuradas sobre el sandbox.

Cada método genera código Python/SymPy, lo ejecuta en el sandbox seguro,
y devuelve un MathResult con el resultado parseado y estructurado.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class MathResult:
    """Resultado estructurado de una operación matemática."""
    success: bool
    operation: str
    input_expr: str
    output_expr: str = ""
    output_latex: str = ""
    numeric_value: Optional[Any] = None
    assumptions: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    code_executed: str = ""
    stdout: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "input_expr": self.input_expr,
            "output_expr": self.output_expr,
            "output_latex": self.output_latex,
            "numeric_value": self.numeric_value,
            "assumptions": self.assumptions,
            "execution_time_ms": self.execution_time_ms,
            "error": self.error,
        }


class MathEngine:
    """
    Motor de operaciones matemáticas que usa el sandbox existente.

    Genera código SymPy/NumPy, lo ejecuta en CodeSandbox, y parsea
    el resultado estructurado. Cada operación produce un MathResult
    con la expresión resultante, su forma LaTeX y valor numérico.
    """

    # Delimitadores para resultado estructurado (mismo patrón que __SANDBOX_FIGURES__)
    RESULT_START = "__MATH_RESULT__"
    RESULT_END = "__MATH_RESULT_END__"

    def __init__(self, sandbox=None, timeout: int = 30):
        if sandbox is None:
            from ..execution.sandbox import CodeSandbox
            sandbox = CodeSandbox(timeout_seconds=timeout)
        self.sandbox = sandbox

    def solve(
        self,
        equation: str,
        variable: str = "x",
        domain: str = "complex",
    ) -> MathResult:
        """Resuelve una ecuación simbólicamente."""
        assumptions = self._domain_assumptions(domain)
        code = f'''
import sympy as sp
import json

{variable} = sp.Symbol("{variable}"{', ' + assumptions if assumptions else ''})
try:
    expr = sp.sympify("{equation}")
    solutions = sp.solve(expr, {variable})
    result = {{
        "output_expr": str(solutions),
        "output_latex": sp.latex(solutions) if solutions else "",
        "numeric_value": [float(s.evalf()) for s in solutions if s.is_number] if isinstance(solutions, list) else None,
        "assumptions": ["{domain}"],
    }}
except Exception as e:
    result = {{"error": str(e)}}

print("{self.RESULT_START}")
print(json.dumps(result, default=str))
print("{self.RESULT_END}")
'''
        return self._execute_math_code(code, "solve", equation)

    def differentiate(
        self,
        expr: str,
        variable: str = "x",
        order: int = 1,
    ) -> MathResult:
        """Calcula la derivada de una expresión."""
        code = f'''
import sympy as sp
import json

{variable} = sp.Symbol("{variable}")
try:
    expression = sp.sympify("{expr}")
    derivative = sp.diff(expression, {variable}, {order})
    result = {{
        "output_expr": str(derivative),
        "output_latex": sp.latex(derivative),
        "numeric_value": float(derivative.evalf()) if derivative.is_number else None,
        "assumptions": ["order={order}"],
    }}
except Exception as e:
    result = {{"error": str(e)}}

print("{self.RESULT_START}")
print(json.dumps(result, default=str))
print("{self.RESULT_END}")
'''
        return self._execute_math_code(code, "differentiate", expr)

    def integrate(
        self,
        expr: str,
        variable: str = "x",
        limits: Optional[Tuple] = None,
    ) -> MathResult:
        """Calcula integral definida o indefinida."""
        if limits:
            integral_call = f"sp.integrate(expression, ({variable}, {limits[0]}, {limits[1]}))"
        else:
            integral_call = f"sp.integrate(expression, {variable})"

        code = f'''
import sympy as sp
import json

{variable} = sp.Symbol("{variable}")
try:
    expression = sp.sympify("{expr}")
    integral = {integral_call}
    result = {{
        "output_expr": str(integral),
        "output_latex": sp.latex(integral),
        "numeric_value": float(integral.evalf()) if integral.is_number else None,
        "assumptions": ["limits={limits}"] if {limits is not None} else ["indefinite"],
    }}
except Exception as e:
    result = {{"error": str(e)}}

print("{self.RESULT_START}")
print(json.dumps(result, default=str))
print("{self.RESULT_END}")
'''
        return self._execute_math_code(code, "integrate", expr)

    def simplify(
        self,
        expr: str,
        method: str = "auto",
    ) -> MathResult:
        """
        Simplifica una expresión usando transforms específicos.

        Métodos: "auto" (prueba todos), "expand", "factor", "trigsimp",
        "cancel", "together", "powsimp", "radsimp", "logcombine"
        """
        if method == "auto":
            simplify_code = '''
    # Intentar transforms específicos en orden
    transforms = [
        ("expand", sp.expand),
        ("factor", sp.factor),
        ("cancel", sp.cancel),
        ("trigsimp", sp.trigsimp),
        ("together", sp.together),
        ("powsimp", sp.powsimp),
    ]
    simplified = expression
    method_used = "none"
    for name, transform in transforms:
        try:
            candidate = transform(expression)
            if sp.count_ops(candidate) < sp.count_ops(simplified):
                simplified = candidate
                method_used = name
        except Exception:
            continue
'''
        else:
            simplify_code = f'''
    simplified = sp.{method}(expression)
    method_used = "{method}"
'''

        code = f'''
import sympy as sp
import json

try:
    expression = sp.sympify("{expr}")
{simplify_code}
    result = {{
        "output_expr": str(simplified),
        "output_latex": sp.latex(simplified),
        "numeric_value": float(simplified.evalf()) if simplified.is_number else None,
        "assumptions": ["method=" + method_used],
    }}
except Exception as e:
    result = {{"error": str(e)}}

print("{self.RESULT_START}")
print(json.dumps(result, default=str))
print("{self.RESULT_END}")
'''
        return self._execute_math_code(code, "simplify", expr)

    def verify_equation(
        self,
        lhs: str,
        rhs: str,
        variables: Optional[List[str]] = None,
    ) -> MathResult:
        """
        Verifica si dos expresiones son equivalentes.

        Usa doble verificación: simbólica (transforms) + numérica (sampling).
        """
        vars_decl = ""
        if variables:
            for v in variables:
                vars_decl += f'{v} = sp.Symbol("{v}")\n'
        else:
            vars_decl = 'x, y, z = sp.symbols("x y z")\n'

        code = f'''
import sympy as sp
import numpy as np
import json

{vars_decl}
try:
    left = sp.sympify("{lhs}")
    right = sp.sympify("{rhs}")
    diff = sp.expand(left - right)

    # Verificación simbólica: intentar transforms específicos
    symbolic_equal = False
    method_used = "none"
    for name, transform in [("expand", sp.expand), ("cancel", sp.cancel),
                             ("trigsimp", sp.trigsimp), ("factor", sp.factor)]:
        try:
            if transform(diff) == 0:
                symbolic_equal = True
                method_used = name
                break
        except Exception:
            continue

    # Verificación numérica: sampling aleatorio
    free_vars = list(diff.free_symbols)
    numeric_equal = True
    if free_vars:
        for _ in range(50):
            point = {{v: np.random.uniform(-10, 10) for v in free_vars}}
            try:
                val = complex(diff.subs(point))
                if abs(val) > 1e-8:
                    numeric_equal = False
                    break
            except (TypeError, ValueError, ZeroDivisionError):
                continue

    verified = symbolic_equal or numeric_equal

    left_latex = sp.latex(left)
    right_latex = sp.latex(right)
    result = {{
        "output_expr": str(verified),
        "output_latex": f"{{left_latex}} = {{right_latex}}" if verified else f"{{left_latex}} \\\\neq {{right_latex}}",
        "numeric_value": 1.0 if verified else 0.0,
        "assumptions": [f"symbolic_check={{symbolic_equal}} ({{method_used}})", f"numeric_check={{numeric_equal}}"],
        "verified": verified,
    }}
except Exception as e:
    result = {{"error": str(e), "verified": False}}

print("{self.RESULT_START}")
print(json.dumps(result, default=str))
print("{self.RESULT_END}")
'''
        return self._execute_math_code(code, "verify_equation", f"{lhs} == {rhs}")

    def matrix_operation(
        self,
        operation: str,
        matrix_expr: str,
    ) -> MathResult:
        """
        Operaciones matriciales: eigenvalues, eigenvectors, determinant,
        inverse, trace, rank, is_unitary, is_hermitian.
        """
        op_map = {
            "eigenvalues": "sp.Matrix({matrix}).eigenvals()",
            "eigenvectors": "sp.Matrix({matrix}).eigenvects()",
            "determinant": "sp.Matrix({matrix}).det()",
            "inverse": "sp.Matrix({matrix}).inv()",
            "trace": "sp.trace(sp.Matrix({matrix}))",
            "rank": "sp.Matrix({matrix}).rank()",
            "is_unitary": "(sp.Matrix({matrix}) * sp.Matrix({matrix}).H - sp.eye(sp.Matrix({matrix}).rows)).is_zero_matrix",
            "is_hermitian": "(sp.Matrix({matrix}) - sp.Matrix({matrix}).H).is_zero_matrix",
        }

        if operation not in op_map:
            return MathResult(
                success=False, operation=f"matrix_{operation}",
                input_expr=matrix_expr, error=f"Operación no soportada: {operation}"
            )

        op_code = op_map[operation].format(matrix=matrix_expr)

        code = f'''
import sympy as sp
import json

try:
    result_val = {op_code}
    result = {{
        "output_expr": str(result_val),
        "output_latex": sp.latex(result_val) if hasattr(result_val, '__class__') and hasattr(sp, 'latex') else str(result_val),
        "numeric_value": None,
        "assumptions": ["operation={operation}"],
    }}
except Exception as e:
    result = {{"error": str(e)}}

print("{self.RESULT_START}")
print(json.dumps(result, default=str))
print("{self.RESULT_END}")
'''
        return self._execute_math_code(code, f"matrix_{operation}", matrix_expr)

    def execute_raw(self, code: str) -> MathResult:
        """
        Ejecuta código Python arbitrario en el sandbox.

        Usado por el orquestador cuando el LLM genera código libre
        dentro de tags <COMPUTE>.
        """
        start = time.time()
        sandbox_result = self.sandbox.execute(code, capture_figures=True)
        elapsed = (time.time() - start) * 1000

        return MathResult(
            success=sandbox_result.success,
            operation="raw_execution",
            input_expr="<code>",
            output_expr=sandbox_result.stdout.strip() if sandbox_result.success else "",
            stdout=sandbox_result.stdout,
            execution_time_ms=elapsed,
            code_executed=code,
            error=sandbox_result.error_message if not sandbox_result.success else None,
        )

    def _execute_math_code(self, code: str, operation: str, input_expr: str) -> MathResult:
        """Ejecuta código en sandbox y extrae resultado estructurado."""
        start = time.time()
        sandbox_result = self.sandbox.execute(code, capture_figures=True)
        elapsed = (time.time() - start) * 1000

        if not sandbox_result.success:
            logger.warning(f"MathEngine.{operation} falló: {sandbox_result.error_message}")
            return MathResult(
                success=False,
                operation=operation,
                input_expr=input_expr,
                execution_time_ms=elapsed,
                code_executed=code,
                stdout=sandbox_result.stdout,
                error=sandbox_result.error_message or sandbox_result.stderr,
            )

        # Extraer JSON estructurado del stdout
        math_data = self._extract_math_result(sandbox_result.stdout)

        if math_data is None:
            logger.warning(f"MathEngine.{operation}: no se encontró resultado estructurado")
            return MathResult(
                success=True,
                operation=operation,
                input_expr=input_expr,
                output_expr=sandbox_result.stdout.strip(),
                execution_time_ms=elapsed,
                code_executed=code,
                stdout=sandbox_result.stdout,
            )

        if "error" in math_data:
            return MathResult(
                success=False,
                operation=operation,
                input_expr=input_expr,
                execution_time_ms=elapsed,
                code_executed=code,
                stdout=sandbox_result.stdout,
                error=math_data["error"],
            )

        return MathResult(
            success=True,
            operation=operation,
            input_expr=input_expr,
            output_expr=math_data.get("output_expr", ""),
            output_latex=math_data.get("output_latex", ""),
            numeric_value=math_data.get("numeric_value"),
            assumptions=math_data.get("assumptions", []),
            execution_time_ms=elapsed,
            code_executed=code,
            stdout=sandbox_result.stdout,
        )

    def _extract_math_result(self, stdout: str) -> Optional[dict]:
        """Extrae JSON delimitado por __MATH_RESULT__ del stdout."""
        if self.RESULT_START not in stdout:
            return None
        try:
            parts = stdout.split(self.RESULT_START)
            json_str = parts[1].split(self.RESULT_END)[0].strip()
            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError) as e:
            logger.warning(f"Error parseando resultado matemático: {e}")
            return None

    @staticmethod
    def _domain_assumptions(domain: str) -> str:
        """Convierte dominio a assumptions de SymPy Symbol."""
        mapping = {
            "real": "real=True",
            "positive": "positive=True",
            "integer": "integer=True",
            "complex": "",
            "nonnegative": "nonnegative=True",
        }
        return mapping.get(domain, "")
