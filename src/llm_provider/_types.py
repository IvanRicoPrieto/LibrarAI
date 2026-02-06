"""
Tipos para el adaptador LLM centralizado.
"""

from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Respuesta unificada de cualquier backend LLM."""

    content: str
    model: str
    provider: str
    tokens_input: int = 0
    tokens_output: int = 0
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True si la respuesta no contiene errores y tiene contenido."""
        return self.error is None and len(self.content) > 0


class LLMProviderError(Exception):
    """
    Error del proveedor LLM configurado.

    Nunca se realiza fallback automatico a otro proveedor.
    """

    pass
