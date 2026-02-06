"""
llm_provider - Adaptador LLM centralizado para LibrarAI.

Centraliza todas las llamadas a modelos de lenguaje (chat completion) en un
solo punto. El backend se selecciona mediante la variable de entorno
LLM_PROVIDER:

    - claude_max (default): Usa la suscripcion de Claude Code Max (tarifa plana).
    - openai_api: Usa la API de OpenAI (requiere OPENAI_API_KEY).
    - anthropic_api: Usa la API de Anthropic (requiere ANTHROPIC_API_KEY).

Si el backend falla, se lanza LLMProviderError. Nunca hay fallback
automatico a otro proveedor para evitar costes inesperados.

Uso:
    from src.llm_provider import complete as llm_complete

    response = llm_complete(
        prompt="Explica que es un qubit.",
        system="Eres un experto en computacion cuantica.",
        json_mode=True,
        temperature=0.1,
    )
    print(response.content)

Embeddings NO pasan por este adaptador (siguen usando OpenAI directamente).
"""

import logging
from typing import Callable

from ._config import get_active_provider, get_defaults
from ._backends import dispatch
from ._types import LLMResponse, LLMProviderError

__all__ = ["complete", "get_provider_name", "LLMResponse", "LLMProviderError"]

logger = logging.getLogger(__name__)

# Resolver provider una vez al importar el modulo
_provider = get_active_provider()
_defaults = get_defaults(_provider)

logger.info(f"LLM provider: {_provider.value} (model: {_defaults['model']})")


def complete(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
    stream: bool = False,
    stream_callback: Callable[[str], None] | None = None,
) -> LLMResponse:
    """
    Envia una solicitud de chat completion al proveedor LLM configurado.

    Args:
        prompt: Mensaje del usuario / instruccion principal.
        system: System prompt (opcional).
        temperature: Override de temperatura (usa default del provider si None).
        max_tokens: Override de max tokens (usa default del provider si None).
        json_mode: Si True, instruye al modelo a responder con JSON valido.
        stream: Si True, entrega respuesta incrementalmente via stream_callback.
        stream_callback: Funcion(chunk: str) llamada por cada fragmento de texto.

    Returns:
        LLMResponse con .content, .ok, .model, .provider, .tokens_input, etc.

    Raises:
        LLMProviderError: Si el proveedor falla. Nunca hay fallback automatico.
    """
    effective_temp = temperature if temperature is not None else _defaults["temperature"]
    effective_max = max_tokens if max_tokens is not None else _defaults["max_tokens"]

    return dispatch(
        provider=_provider,
        prompt=prompt,
        system=system,
        temperature=effective_temp,
        max_tokens=effective_max,
        json_mode=json_mode,
        stream=stream,
        stream_callback=stream_callback,
    )


def get_provider_name() -> str:
    """Devuelve el nombre del proveedor activo (e.g. 'claude_max')."""
    return _provider.value
