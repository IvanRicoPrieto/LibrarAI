"""
Implementaciones de backends LLM.

Cada funcion recibe los mismos parametros y devuelve LLMResponse.
"""

import logging
import time
from typing import Callable

from ._config import Provider, get_defaults
from ._types import LLMResponse, LLMProviderError

logger = logging.getLogger(__name__)

# Singletons de clientes (se crean en el primer uso)
_claude_max_instance = None
_openai_instance = None
_anthropic_instance = None


def _complete_claude_max(
    prompt: str,
    system: str | None,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    stream: bool,
    stream_callback: Callable[[str], None] | None,
) -> LLMResponse:
    """Backend: ClaudeMaxClient via suscripcion Claude Code Max."""
    global _claude_max_instance

    try:
        from src.claude_max_client import ClaudeMaxClient
    except ImportError as e:
        raise LLMProviderError(
            "claude_max_client no disponible. "
            "Verifica que src/claude_max_client/ existe y claude-agent-sdk esta instalado."
        ) from e

    if _claude_max_instance is None:
        defaults = get_defaults(Provider.CLAUDE_MAX)
        _claude_max_instance = ClaudeMaxClient(
            model=defaults["model"],
            default_temperature=defaults["temperature"],
            default_max_tokens=defaults["max_tokens"],
        )

    try:
        resp = _claude_max_instance.complete(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            stream=stream,
            stream_callback=stream_callback,
        )
    except Exception as e:
        raise LLMProviderError(f"claude_max: {e}") from e

    if not resp.ok:
        raise LLMProviderError(
            f"claude_max: respuesta vacia o con error: {resp.error or 'sin contenido'}"
        )

    return LLMResponse(
        content=resp.content,
        model=resp.model,
        provider="claude_max",
        tokens_input=resp.tokens_input,
        tokens_output=resp.tokens_output,
        latency_ms=resp.latency_ms,
    )


def _complete_openai_api(
    prompt: str,
    system: str | None,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    stream: bool,
    stream_callback: Callable[[str], None] | None,
) -> LLMResponse:
    """Backend: OpenAI API (pago por token)."""
    import os

    global _openai_instance

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMProviderError(
            "LLM_PROVIDER=openai_api pero OPENAI_API_KEY no esta configurada."
        )

    try:
        from openai import OpenAI
    except ImportError as e:
        raise LLMProviderError(
            "Paquete 'openai' no instalado. Ejecuta: pip install openai"
        ) from e

    if _openai_instance is None:
        _openai_instance = OpenAI(api_key=api_key)

    defaults = get_defaults(Provider.OPENAI_API)
    model = defaults["model"]

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    start = time.time()

    try:
        if stream and stream_callback:
            kwargs["stream"] = True
            resp = _openai_instance.chat.completions.create(**kwargs)
            content_parts: list[str] = []
            for chunk in resp:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    content_parts.append(text)
                    stream_callback(text)
            content = "".join(content_parts)
            tokens_in = len(prompt) // 4
            tokens_out = len(content) // 4
        else:
            resp = _openai_instance.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            tokens_in = resp.usage.prompt_tokens if resp.usage else len(prompt) // 4
            tokens_out = resp.usage.completion_tokens if resp.usage else len(content) // 4
    except Exception as e:
        raise LLMProviderError(f"openai_api: {e}") from e

    latency = (time.time() - start) * 1000

    return LLMResponse(
        content=content,
        model=model,
        provider="openai_api",
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        latency_ms=latency,
    )


def _complete_anthropic_api(
    prompt: str,
    system: str | None,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    stream: bool,
    stream_callback: Callable[[str], None] | None,
) -> LLMResponse:
    """Backend: Anthropic API (pago por token)."""
    import os

    global _anthropic_instance

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMProviderError(
            "LLM_PROVIDER=anthropic_api pero ANTHROPIC_API_KEY no esta configurada."
        )

    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise LLMProviderError(
            "Paquete 'anthropic' no instalado. Ejecuta: pip install anthropic"
        ) from e

    if _anthropic_instance is None:
        _anthropic_instance = Anthropic(api_key=api_key)

    defaults = get_defaults(Provider.ANTHROPIC_API)
    model = defaults["model"]

    effective_prompt = prompt
    if json_mode:
        effective_prompt = (
            prompt
            + "\n\nIMPORTANTE: Responde UNICAMENTE con JSON valido. "
            "No incluyas texto adicional ni bloques de codigo markdown."
        )

    messages = [{"role": "user", "content": effective_prompt}]

    start = time.time()

    try:
        if stream and stream_callback:
            with _anthropic_instance.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "",
                messages=messages,
            ) as stream_response:
                content_parts: list[str] = []
                for text in stream_response.text_stream:
                    content_parts.append(text)
                    stream_callback(text)

            final = stream_response.get_final_message()
            content = "".join(content_parts)
            tokens_in = final.usage.input_tokens
            tokens_out = final.usage.output_tokens
        else:
            resp = _anthropic_instance.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "",
                messages=messages,
            )
            content = resp.content[0].text if resp.content else ""
            tokens_in = resp.usage.input_tokens
            tokens_out = resp.usage.output_tokens
    except Exception as e:
        raise LLMProviderError(f"anthropic_api: {e}") from e

    latency = (time.time() - start) * 1000

    return LLMResponse(
        content=content,
        model=model,
        provider="anthropic_api",
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        latency_ms=latency,
    )


# Dispatch table
_DISPATCH = {
    Provider.CLAUDE_MAX: _complete_claude_max,
    Provider.OPENAI_API: _complete_openai_api,
    Provider.ANTHROPIC_API: _complete_anthropic_api,
}


def dispatch(
    provider: Provider,
    prompt: str,
    system: str | None,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    stream: bool,
    stream_callback: Callable[[str], None] | None,
) -> LLMResponse:
    """Despacha la llamada al backend correcto."""
    fn = _DISPATCH[provider]
    return fn(
        prompt=prompt,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
        stream=stream,
        stream_callback=stream_callback,
    )
