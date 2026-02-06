"""
Configuracion del proveedor LLM.

Lee LLM_PROVIDER del entorno y proporciona los defaults por proveedor.
"""

import os
from enum import Enum

from ._types import LLMProviderError


class Provider(Enum):
    CLAUDE_MAX = "claude_max"
    OPENAI_API = "openai_api"
    ANTHROPIC_API = "anthropic_api"


PROVIDER_DEFAULTS = {
    Provider.CLAUDE_MAX: {
        "model": "claude-opus-4-5-20251101",
        "temperature": 0.3,
        "max_tokens": 4096,
    },
    Provider.OPENAI_API: {
        "model": "gpt-4.1-mini",
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    Provider.ANTHROPIC_API: {
        "model": "claude-sonnet-4-5-20250929",
        "temperature": 0.3,
        "max_tokens": 4096,
    },
}


def get_active_provider() -> Provider:
    """Lee LLM_PROVIDER del entorno. Default: claude_max."""
    raw = os.getenv("LLM_PROVIDER", "claude_max").lower().strip()
    try:
        return Provider(raw)
    except ValueError:
        valid = ", ".join(p.value for p in Provider)
        raise LLMProviderError(
            f"LLM_PROVIDER='{raw}' no reconocido. Opciones: {valid}"
        )


def get_defaults(provider: Provider) -> dict:
    """Devuelve los defaults (model, temperature, max_tokens) del provider."""
    return PROVIDER_DEFAULTS[provider]
