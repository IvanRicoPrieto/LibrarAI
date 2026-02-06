"""
LibrarAI API - Interfaz programática optimizada para agentes.

Provee 5 modos de operación:
- EXPLORE: Descubrir contenido disponible
- RETRIEVE: Obtener contenido exhaustivo
- QUERY: Responder preguntas específicas
- VERIFY: Verificar afirmaciones contra fuentes
- CITE: Generar citas formateadas
"""

from .agent_interface import (
    AgentAPI,
    ExploreResult,
    RetrieveResult,
    QueryResult,
    VerifyResult,
    CiteResult,
    ContentNode,
    VerificationStatus,
)

__all__ = [
    "AgentAPI",
    "ExploreResult",
    "RetrieveResult",
    "QueryResult",
    "VerifyResult",
    "CiteResult",
    "ContentNode",
    "VerificationStatus",
]
