"""
Tipos de datos para claude_max_client.

Define las estructuras de respuesta y configuracion del cliente.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompletionResponse:
    """
    Respuesta de una llamada de completacion a Claude via suscripcion Max.

    Attributes:
        content: Texto de la respuesta generada.
        model: Identificador del modelo usado (e.g. "claude-sonnet-4-5-20250929").
        tokens_input: Tokens de entrada consumidos (estimados o reportados).
        tokens_output: Tokens de salida generados (estimados o reportados).
        cost_usd: Coste reportado por el SDK. Con suscripcion Max sera 0 o None.
        latency_ms: Latencia total de la llamada en milisegundos.
        session_id: ID de sesion del Agent SDK, util para depuracion.
        raw_usage: Datos crudos de uso del ResultMessage del SDK.
        error: Mensaje de error si la llamada fallo (solo en batch con errores parciales).
    """
    content: str
    model: str
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float | None = None
    latency_ms: float = 0.0
    session_id: str | None = None
    raw_usage: dict[str, Any] | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True si la respuesta no contiene errores."""
        return self.error is None and len(self.content) > 0

    def to_dict(self) -> dict[str, Any]:
        """Serializa la respuesta a diccionario."""
        return {
            "content": self.content,
            "model": self.model,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "session_id": self.session_id,
            "raw_usage": self.raw_usage,
            "error": self.error,
        }


# Modelos disponibles con suscripcion Claude Code Max
AVAILABLE_MODELS = {
    "claude-opus-4-5-20251101": {
        "name": "Claude Opus 4.5",
        "context_window": 200_000,
        "description": "Maximo razonamiento, tareas complejas",
    },
    "claude-sonnet-4-5-20250929": {
        "name": "Claude Sonnet 4.5",
        "context_window": 200_000,
        "description": "Mejor balance calidad/velocidad (recomendado)",
    },
    "claude-sonnet-4-20250514": {
        "name": "Claude Sonnet 4",
        "context_window": 200_000,
        "description": "Rapido, buen rendimiento general",
    },
    "claude-haiku-4-5-20251001": {
        "name": "Claude Haiku 4.5",
        "context_window": 200_000,
        "description": "Ultra-rapido, tareas simples y alto volumen",
    },
}

DEFAULT_MODEL = "claude-opus-4-5-20251101"
