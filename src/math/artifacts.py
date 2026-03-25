"""
MathArtifact — Objeto de evidencia computacional.

Cada operación matemática verificada produce un MathArtifact que
documenta el input, output, motor usado, verificación realizada
y cadena de provenance. Constituye la evidencia computacional
que se adjunta a las respuestas junto a las citas textuales.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional


class VerificationLevel(Enum):
    """Niveles de verificación, de menor a mayor rigor."""
    NONE = -1           # Sin verificación
    DIMENSIONAL = 0     # Pint: consistencia dimensional
    NUMERICAL = 1       # NumPy/mpmath: sampling numérico
    SYMBOLIC = 2        # SymPy: equivalencia simbólica (transforms específicos)
    PHYSICAL = 3        # Invariantes físicos (unitariedad, traza, conservación)
    FORMAL = 5          # Lean 4: prueba formal (futuro)


@dataclass
class MathArtifact:
    """Evidencia computacional de un paso matemático."""

    # Entrada
    input_latex: str = ""
    input_sympy: str = ""
    parse_confidence: float = 1.0

    # Contexto
    assumptions: List[str] = field(default_factory=list)
    source_chunks: List[str] = field(default_factory=list)

    # Computación
    engine: str = "sympy"
    operation: str = ""
    code: str = ""
    result: str = ""
    result_latex: str = ""
    numeric_result: Optional[Any] = None

    # Verificación
    verification_level: VerificationLevel = VerificationLevel.NONE
    verification_passed: bool = False
    verification_details: Dict[str, Any] = field(default_factory=dict)

    # Provenance
    content_hash: str = ""
    provenance_chain: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if not self.content_hash and (self.code or self.result):
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """SHA256 del código + resultado para integridad."""
        content = f"{self.code}|{self.result}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_latex": self.input_latex,
            "input_sympy": self.input_sympy,
            "parse_confidence": self.parse_confidence,
            "assumptions": self.assumptions,
            "source_chunks": self.source_chunks,
            "engine": self.engine,
            "operation": self.operation,
            "code": self.code,
            "result": self.result,
            "result_latex": self.result_latex,
            "numeric_result": str(self.numeric_result) if self.numeric_result is not None else None,
            "verification_level": self.verification_level.name,
            "verification_passed": self.verification_passed,
            "verification_details": self.verification_details,
            "content_hash": self.content_hash,
            "provenance_chain": self.provenance_chain,
            "timestamp": self.timestamp,
        }

    def to_evidence_block(self) -> str:
        """Renderiza como bloque markdown de evidencia para incluir en respuestas."""
        status = "PASS" if self.verification_passed else "FAIL"
        level_name = self.verification_level.name

        lines = [
            f"**Verificación computacional** [{status}] (nivel: {level_name})",
            f"- Motor: `{self.engine}` | Operación: `{self.operation}`",
        ]

        if self.result_latex:
            lines.append(f"- Resultado: ${self.result_latex}$")
        elif self.result:
            lines.append(f"- Resultado: `{self.result}`")

        if self.assumptions:
            lines.append(f"- Asunciones: {', '.join(self.assumptions)}")

        if self.source_chunks:
            chunk_refs = ", ".join(self.source_chunks[:3])
            lines.append(f"- Fuentes: {chunk_refs}")

        lines.append(f"- Hash: `{self.content_hash}`")

        return "\n".join(lines)

    @classmethod
    def from_math_result(cls, math_result, source_chunks: Optional[List[str]] = None) -> "MathArtifact":
        """Crea un MathArtifact desde un MathResult del engine."""
        return cls(
            input_sympy=math_result.input_expr,
            engine="sympy",
            operation=math_result.operation,
            code=math_result.code_executed,
            result=math_result.output_expr,
            result_latex=math_result.output_latex,
            numeric_result=math_result.numeric_value,
            assumptions=math_result.assumptions,
            source_chunks=source_chunks or [],
            verification_level=VerificationLevel.NONE,
            verification_passed=math_result.success,
        )
