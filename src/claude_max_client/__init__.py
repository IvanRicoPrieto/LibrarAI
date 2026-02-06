"""
claude_max_client - Cliente para usar modelos Claude via suscripcion Max.

Modulo autocontenido y reutilizable que permite hacer llamadas a modelos Claude
(Sonnet, Opus, Haiku) usando la suscripcion de Claude Code Max ($200/mes) en vez
de pagar por token via la API de Anthropic.

Envuelve el Claude Agent SDK oficial (claude-agent-sdk) con una interfaz simple.

Uso rapido:
    from src.claude_max_client import ClaudeMaxClient

    client = ClaudeMaxClient()
    response = client.complete("Explica que es un qubit.")
    print(response.content)

Uso con streaming:
    def on_chunk(text):
        print(text, end="", flush=True)

    response = client.complete(
        prompt="Explica la superposicion cuantica.",
        stream=True,
        stream_callback=on_chunk,
    )

Uso con imagenes (vision):
    response = client.complete(
        prompt="Describe esta figura.",
        images=["data/books/computacion_cuantica/images/fig1.jpg"],
    )

Batch processing:
    responses = client.batch_complete(
        prompts=["Pregunta 1", "Pregunta 2", "Pregunta 3"],
        max_concurrency=5,
        progress_callback=lambda done, total: print(f"{done}/{total}"),
    )

Dependencias:
    - claude-agent-sdk>=0.1.0
    - Python 3.10+
"""

from .client import ClaudeMaxClient
from .types import CompletionResponse, AVAILABLE_MODELS, DEFAULT_MODEL
from .exceptions import (
    ClaudeMaxError,
    AuthenticationError,
    RateLimitError,
    ModelNotAvailableError,
    ImageProcessingError,
    FileProcessingError,
    CompletionError,
)

__all__ = [
    "ClaudeMaxClient",
    "CompletionResponse",
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL",
    "ClaudeMaxError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotAvailableError",
    "ImageProcessingError",
    "FileProcessingError",
    "CompletionError",
]

__version__ = "0.1.0"
