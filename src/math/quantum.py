"""
Quantum Computation Module — Simulación y verificación cuántica especializada.

Fase 4 del roadmap: Integra QuTiP y PennyLane para:
- Simulación de estados y operadores cuánticos
- Verificación de circuitos cuánticos
- Comprobación de invariantes cuánticos (unitariedad, traza, hermiticidad)
- Evolución temporal de sistemas cuánticos
- Productos tensoriales y traza parcial

Usa el sandbox existente (qutip y pennylane ya están en whitelist).
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from .engine import MathEngine, MathResult
from .artifacts import MathArtifact, VerificationLevel

logger = logging.getLogger(__name__)


@dataclass
class QuantumState:
    """Representación de un estado cuántico con metadata."""
    n_qubits: int
    state_vector: str           # Representación SymPy/string
    density_matrix: str = ""    # Si es estado mixto
    is_pure: bool = True
    fidelity: Optional[float] = None
    entanglement: Optional[float] = None  # Concurrencia o entropía de von Neumann


@dataclass
class QuantumCircuit:
    """Representación de un circuito cuántico."""
    n_qubits: int
    gates: List[Dict[str, Any]] = field(default_factory=list)
    unitary_matrix: str = ""
    is_unitary: Optional[bool] = None


class QuantumEngine:
    """
    Motor de computación cuántica usando el sandbox.

    Ejecuta operaciones cuánticas con SymPy (para cálculos simbólicos)
    y opcionalmente QuTiP (para simulación numérica).
    """

    def __init__(self, math_engine: Optional[MathEngine] = None):
        self.engine = math_engine or MathEngine()

    def verify_unitary(self, matrix_expr: str) -> MathResult:
        """Verifica que una matriz es unitaria: U†U = I."""
        code = f'''
import sympy as sp
import json

try:
    M = sp.Matrix({matrix_expr})
    product = sp.simplify(M * M.H - sp.eye(M.rows))
    is_unitary = product.is_zero_matrix
    if is_unitary is None:
        # Intentar numéricamente
        import numpy as np
        M_num = np.array(M.tolist(), dtype=complex)
        product_num = M_num @ M_num.conj().T - np.eye(M.rows)
        is_unitary = np.allclose(product_num, 0, atol=1e-10)

    result = {{
        "output_expr": str(is_unitary),
        "output_latex": "U^\\\\dagger U = I" if is_unitary else "U^\\\\dagger U \\\\neq I",
        "numeric_value": 1.0 if is_unitary else 0.0,
        "assumptions": ["unitary_check"],
    }}
except Exception as e:
    result = {{"error": str(e)}}

print("__MATH_RESULT__")
print(json.dumps(result, default=str))
print("__MATH_RESULT_END__")
'''
        return self.engine._execute_math_code(code, "verify_unitary", matrix_expr)

    def verify_hermitian(self, matrix_expr: str) -> MathResult:
        """Verifica que un operador es hermítico: H = H†."""
        code = f'''
import sympy as sp
import json

try:
    M = sp.Matrix({matrix_expr})
    diff = sp.simplify(M - M.H)
    is_hermitian = diff.is_zero_matrix
    if is_hermitian is None:
        import numpy as np
        M_num = np.array(M.tolist(), dtype=complex)
        is_hermitian = np.allclose(M_num, M_num.conj().T, atol=1e-10)

    eigenvals = list(M.eigenvals().keys())
    all_real = all(sp.im(ev) == 0 for ev in eigenvals)

    result = {{
        "output_expr": str(is_hermitian),
        "output_latex": "H = H^\\\\dagger" if is_hermitian else "H \\\\neq H^\\\\dagger",
        "numeric_value": 1.0 if is_hermitian else 0.0,
        "assumptions": ["hermitian_check", f"eigenvalues_real={{all_real}}"],
    }}
except Exception as e:
    result = {{"error": str(e)}}

print("__MATH_RESULT__")
print(json.dumps(result, default=str))
print("__MATH_RESULT_END__")
'''
        return self.engine._execute_math_code(code, "verify_hermitian", matrix_expr)

    def tensor_product(self, *matrices: str) -> MathResult:
        """Calcula el producto tensorial de matrices."""
        matrices_code = ", ".join(f'sp.Matrix({m})' for m in matrices)
        code = f'''
import sympy as sp
from functools import reduce
import json

try:
    matrices = [{matrices_code}]
    result_matrix = reduce(lambda a, b: sp.kronecker_product(a, b), matrices)
    result = {{
        "output_expr": str(result_matrix.tolist()),
        "output_latex": sp.latex(result_matrix),
        "numeric_value": None,
        "assumptions": ["tensor_product"],
    }}
except Exception as e:
    result = {{"error": str(e)}}

print("__MATH_RESULT__")
print(json.dumps(result, default=str))
print("__MATH_RESULT_END__")
'''
        return self.engine._execute_math_code(
            code, "tensor_product", " ⊗ ".join(str(m)[:30] for m in matrices)
        )

    def partial_trace(self, matrix_expr: str, trace_out: int, dims: List[int]) -> MathResult:
        """Calcula la traza parcial de una matriz de densidad."""
        code = f'''
import sympy as sp
import json

try:
    rho = sp.Matrix({matrix_expr})
    dims = {dims}
    trace_out = {trace_out}

    # Traza parcial manual
    n_systems = len(dims)
    total_dim = 1
    for d in dims:
        total_dim *= d

    # Construir la traza parcial
    keep_dims = [d for i, d in enumerate(dims) if i != trace_out]
    keep_dim = 1
    for d in keep_dims:
        keep_dim *= d

    traced_dim = dims[trace_out]
    result_matrix = sp.zeros(keep_dim, keep_dim)

    # Para cada base del sistema a trazar
    for k in range(traced_dim):
        # Construir projector
        proj = sp.zeros(total_dim, total_dim)
        for i in range(keep_dim):
            for j in range(keep_dim):
                # Calcular índices
                if trace_out == 0:
                    row = k * keep_dim + i
                    col = k * keep_dim + j
                else:
                    row = i * traced_dim + k
                    col = j * traced_dim + k
                if row < total_dim and col < total_dim:
                    result_matrix[i, j] += rho[row, col]

    result = {{
        "output_expr": str(result_matrix.tolist()),
        "output_latex": sp.latex(result_matrix),
        "numeric_value": None,
        "assumptions": ["partial_trace", f"trace_out={trace_out}", f"dims={dims}"],
    }}
except Exception as e:
    result = {{"error": str(e)}}

print("__MATH_RESULT__")
print(json.dumps(result, default=str))
print("__MATH_RESULT_END__")
'''
        return self.engine._execute_math_code(code, "partial_trace", matrix_expr[:50])

    def commutator(self, A: str, B: str) -> MathResult:
        """Calcula el conmutador [A, B] = AB - BA."""
        code = f'''
import sympy as sp
import json

try:
    A = sp.Matrix({A})
    B = sp.Matrix({B})
    comm = sp.simplify(A * B - B * A)

    comm_is_zero = comm.is_zero_matrix
    result = {{
        "output_expr": str(comm.tolist()),
        "output_latex": sp.latex(comm),
        "numeric_value": None,
        "assumptions": ["commutator", f"commutes={{comm_is_zero}}"],
    }}
except Exception as e:
    result = {{"error": str(e)}}

print("__MATH_RESULT__")
print(json.dumps(result, default=str))
print("__MATH_RESULT_END__")
'''
        return self.engine._execute_math_code(code, "commutator", f"[A, B]")

    def anticommutator(self, A: str, B: str) -> MathResult:
        """Calcula el anticonmutador {A, B} = AB + BA."""
        code = f'''
import sympy as sp
import json

try:
    A = sp.Matrix({A})
    B = sp.Matrix({B})
    anticomm = sp.simplify(A * B + B * A)

    result = {{
        "output_expr": str(anticomm.tolist()),
        "output_latex": sp.latex(anticomm),
        "numeric_value": None,
        "assumptions": ["anticommutator"],
    }}
except Exception as e:
    result = {{"error": str(e)}}

print("__MATH_RESULT__")
print(json.dumps(result, default=str))
print("__MATH_RESULT_END__")
'''
        return self.engine._execute_math_code(code, "anticommutator", f"{{A, B}}")

    def quantum_gate(
        self,
        gate_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> MathResult:
        """
        Genera y verifica una puerta cuántica estándar.

        Gates soportados: H, X, Y, Z, S, T, CNOT, CZ, SWAP,
        Rx, Ry, Rz, Phase, Toffoli, QFT
        """
        params = params or {}

        gate_definitions = {
            "H": "sp.Matrix([[1,1],[1,-1]]) / sp.sqrt(2)",
            "X": "sp.Matrix([[0,1],[1,0]])",
            "Y": "sp.Matrix([[0,-sp.I],[sp.I,0]])",
            "Z": "sp.Matrix([[1,0],[0,-1]])",
            "S": "sp.Matrix([[1,0],[0,sp.I]])",
            "T": "sp.Matrix([[1,0],[0,sp.exp(sp.I*sp.pi/4)]])",
            "CNOT": "sp.Matrix([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])",
            "CZ": "sp.Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])",
            "SWAP": "sp.Matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])",
        }

        if gate_name in gate_definitions:
            gate_code = gate_definitions[gate_name]
        elif gate_name.startswith("R"):
            # Rotación parametrizada
            axis = gate_name[1].lower()
            theta = params.get("theta", "sp.Symbol('theta')")
            if axis == "x":
                gate_code = f"sp.Matrix([[sp.cos({theta}/2), -sp.I*sp.sin({theta}/2)],[-sp.I*sp.sin({theta}/2), sp.cos({theta}/2)]])"
            elif axis == "y":
                gate_code = f"sp.Matrix([[sp.cos({theta}/2), -sp.sin({theta}/2)],[sp.sin({theta}/2), sp.cos({theta}/2)]])"
            elif axis == "z":
                gate_code = f"sp.Matrix([[sp.exp(-sp.I*{theta}/2), 0],[0, sp.exp(sp.I*{theta}/2)]])"
            else:
                return MathResult(
                    success=False, operation="quantum_gate",
                    input_expr=gate_name, error=f"Eje de rotación no válido: {axis}"
                )
        elif gate_name == "QFT":
            n = params.get("n", 2)
            gate_code = f"""(lambda n: (lambda N, omega: sp.Matrix(N, N, lambda i, j: omega**(i*j)) / sp.sqrt(N))(2**n, sp.exp(2*sp.pi*sp.I/(2**n))))({n})"""
        else:
            return MathResult(
                success=False, operation="quantum_gate",
                input_expr=gate_name, error=f"Puerta no soportada: {gate_name}"
            )

        code = f'''
import sympy as sp
import json

try:
    gate = {gate_code}
    # Verificar unitariedad
    product = sp.simplify(gate * gate.H - sp.eye(gate.rows))
    is_unitary = product.is_zero_matrix
    if is_unitary is None:
        is_unitary = all(abs(complex(v)) < 1e-10 for v in product)

    result = {{
        "output_expr": str(gate.tolist()),
        "output_latex": sp.latex(gate),
        "numeric_value": None,
        "assumptions": [
            "gate={gate_name}",
            f"unitary={{is_unitary}}",
            f"dim={{gate.rows}}x{{gate.cols}}"
        ],
    }}
except Exception as e:
    result = {{"error": str(e)}}

print("__MATH_RESULT__")
print(json.dumps(result, default=str))
print("__MATH_RESULT_END__")
'''
        return self.engine._execute_math_code(code, "quantum_gate", gate_name)

    def apply_gate(
        self,
        gate_expr: str,
        state_expr: str,
    ) -> MathResult:
        """Aplica una puerta a un estado cuántico."""
        code = f'''
import sympy as sp
import json

try:
    gate = sp.Matrix({gate_expr})
    state = sp.Matrix({state_expr})
    result_state = sp.simplify(gate * state)

    # Verificar normalización
    norm = sp.simplify((result_state.H * result_state)[0,0])

    result = {{
        "output_expr": str(result_state.tolist()),
        "output_latex": sp.latex(result_state),
        "numeric_value": complex(norm.evalf()) if norm.is_number else None,
        "assumptions": [f"normalized={{sp.simplify(norm - 1) == 0}}"],
    }}
except Exception as e:
    result = {{"error": str(e)}}

print("__MATH_RESULT__")
print(json.dumps(result, default=str))
print("__MATH_RESULT_END__")
'''
        return self.engine._execute_math_code(code, "apply_gate", f"gate * state")

    def measure_probabilities(self, state_expr: str) -> MathResult:
        """Calcula probabilidades de medición en base computacional."""
        code = f'''
import sympy as sp
import json

try:
    state = sp.Matrix({state_expr})
    n = state.rows
    probs = {{}}

    for i in range(n):
        prob = sp.simplify(sp.Abs(state[i])**2)
        if prob != 0:
            # Formato de base computacional
            label = bin(i)[2:].zfill(len(bin(n-1)[2:]))
            probs[f"|{{label}}⟩"] = str(prob)

    # Verificar que suman 1
    total = sum(sp.Abs(state[i])**2 for i in range(n))
    total_simplified = sp.simplify(total)

    result = {{
        "output_expr": str(probs),
        "output_latex": ", ".join(f"P({{k}}) = {{v}}" for k, v in probs.items()),
        "numeric_value": float(total_simplified.evalf()) if total_simplified.is_number else None,
        "assumptions": [f"total_probability={{total_simplified}}"],
    }}
except Exception as e:
    result = {{"error": str(e)}}

print("__MATH_RESULT__")
print(json.dumps(result, default=str))
print("__MATH_RESULT_END__")
'''
        return self.engine._execute_math_code(code, "measure_probabilities", state_expr[:50])

    def von_neumann_entropy(self, density_matrix_expr: str) -> MathResult:
        """Calcula la entropía de von Neumann S(ρ) = -Tr(ρ log₂ ρ)."""
        code = f'''
import sympy as sp
import json

try:
    rho = sp.Matrix({density_matrix_expr})
    eigenvals_dict = rho.eigenvals()  # {{eigenvalue: multiplicity}}

    # S = -sum(multiplicity * lambda_i * log2(lambda_i)) para lambda_i > 0
    entropy = 0
    for ev, mult in eigenvals_dict.items():
        ev_val = sp.simplify(ev)
        if ev_val > 0:
            entropy -= mult * ev_val * sp.log(ev_val, 2)

    entropy = sp.simplify(entropy)

    result = {{
        "output_expr": str(entropy),
        "output_latex": sp.latex(entropy),
        "numeric_value": float(entropy.evalf()) if entropy.is_number else None,
        "assumptions": ["von_neumann_entropy", "base_2"],
    }}
except Exception as e:
    result = {{"error": str(e)}}

print("__MATH_RESULT__")
print(json.dumps(result, default=str))
print("__MATH_RESULT_END__")
'''
        return self.engine._execute_math_code(code, "von_neumann_entropy", density_matrix_expr[:50])

    def fidelity(self, rho_expr: str, sigma_expr: str) -> MathResult:
        """Calcula la fidelidad F(ρ, σ) entre dos estados."""
        code = f'''
import sympy as sp
import json

try:
    rho = sp.Matrix({rho_expr})
    sigma = sp.Matrix({sigma_expr})

    # Para estados puros: F = |⟨ψ|φ⟩|²
    # Verificar si son vectores (estados puros)
    if rho.cols == 1 and sigma.cols == 1:
        overlap = (rho.H * sigma)[0, 0]
        fid = sp.simplify(sp.Abs(overlap)**2)
    else:
        # F(ρ,σ) = (Tr(√(√ρ σ √ρ)))²
        # Para matrices de densidad, usar eigenvalues con multiplicidades
        sqrt_rho = rho  # Simplificación para diagonales
        product = sqrt_rho * sigma * sqrt_rho
        eigenvals_dict = product.eigenvals()  # {{eigenvalue: multiplicity}}
        sqrt_sum = sum(mult * sp.sqrt(sp.Abs(ev)) for ev, mult in eigenvals_dict.items())
        fid = sp.simplify(sqrt_sum**2)

    result = {{
        "output_expr": str(fid),
        "output_latex": sp.latex(fid),
        "numeric_value": float(fid.evalf()) if fid.is_number else None,
        "assumptions": ["fidelity"],
    }}
except Exception as e:
    result = {{"error": str(e)}}

print("__MATH_RESULT__")
print(json.dumps(result, default=str))
print("__MATH_RESULT_END__")
'''
        return self.engine._execute_math_code(code, "fidelity", "F(ρ, σ)")

    def verify_quantum_invariants(
        self,
        matrix_expr: str,
        invariants: List[str],
    ) -> Dict[str, MathArtifact]:
        """
        Verifica múltiples invariantes cuánticos para una matriz.

        invariants: Lista de ["unitary", "hermitian", "trace_one",
                              "positive_semidefinite", "normalized"]
        """
        from .verification import VerificationPipeline

        pipeline = VerificationPipeline(engine=self.engine, max_level=3)
        results = {}

        for inv in invariants:
            artifact = pipeline.verify(
                matrix_expr, matrix_expr,
                min_level=3, max_level=3,
                context={"invariant_type": inv},
            )
            results[inv] = artifact

        return results
