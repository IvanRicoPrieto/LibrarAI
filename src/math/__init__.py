"""
Motor Matemático de LibrarAI.

Provee capacidades de cálculo simbólico y numérico integradas con el pipeline RAG:

Fase 1: Loop de computación bidireccional
- MathEngine: Operaciones matemáticas estructuradas sobre el sandbox
- MathComputationOrchestrator: Loop bidireccional LLM ↔ sandbox

Fase 2: Verificación integrada
- MathArtifact: Evidencia computacional verificable
- VerificationPipeline: Verificación multi-nivel
- WolframClient: Fallback a Wolfram Alpha API
- LaTeXParser: Conversión LaTeX → SymPy

Fase 3: Razonamiento multi-agente
- MultiAgentOrchestrator: Planner/Calculator/Verifier/Synthesizer
- ProvenanceGraph: Trazabilidad W3C PROV

Fase 4: Computación cuántica
- QuantumEngine: SymPy + QuTiP para estados y operadores cuánticos

Fase 5: Knowledge Graph Computacional
- FormulaGraph: Grafo de fórmulas con fingerprinting y e-graph rules
- FormulaFingerprintEngine: Búsqueda por equivalencia simbólica

Fase 6: Verificación formal
- FormalVerifier: Integración con Lean 4 via LeanInterface
- Autoformalizator: Traducción NL+LaTeX → Lean 4
"""

from .engine import MathEngine, MathResult
from .artifacts import MathArtifact, VerificationLevel

__all__ = [
    "MathEngine",
    "MathResult",
    "MathArtifact",
    "VerificationLevel",
]
