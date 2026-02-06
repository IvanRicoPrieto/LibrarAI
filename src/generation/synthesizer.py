"""
Response Synthesizer - Genera respuestas usando LLMs.

El proveedor LLM se selecciona via la variable de entorno LLM_PROVIDER
a traves del adaptador centralizado src/llm_provider.
"""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ..retrieval.fusion import RetrievalResult
from .prompt_builder import PromptBuilder, QueryType
from ..utils.cost_tracker import get_tracker

logger = logging.getLogger(__name__)


@dataclass
class GeneratedResponse:
    """Respuesta generada con metadatos."""
    content: str
    query: str
    query_type: str
    sources_used: List[str]  # chunk_ids usados
    model: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    abstained: bool = False  # Si el sistema se abstuvo de responder
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "query": self.query,
            "query_type": self.query_type,
            "sources_used": self.sources_used,
            "model": self.model,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "abstained": self.abstained
        }


class ResponseSynthesizer:
    """
    Sintetizador de respuestas usando el adaptador LLM centralizado.

    El proveedor se controla via LLM_PROVIDER (default: claude_max).
    No hay fallback automatico a proveedores de pago.
    """

    def __init__(
        self,
        temperature: float = 0.3,
        max_output_tokens: int = 2000,
    ):
        """
        Args:
            temperature: Temperatura de sampling
            max_output_tokens: Máximo tokens de salida
        """
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.prompt_builder = PromptBuilder()

    def generate(
        self,
        query: str,
        results: List[RetrievalResult],
        query_type: Optional[QueryType] = None,
        stream: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None
    ) -> GeneratedResponse:
        """
        Genera respuesta basada en resultados de retrieval.

        Args:
            query: Consulta del usuario
            results: Resultados del retrieval
            query_type: Tipo de query (auto-detectado si None)
            stream: Si usar streaming
            stream_callback: Función callback para streaming

        Returns:
            Respuesta generada con metadatos
        """
        from src.llm_provider import complete as llm_complete, get_provider_name

        import time
        start_time = time.time()

        # Construir prompt
        prompt = self.prompt_builder.build_prompt(
            query=query,
            results=results,
            query_type=query_type
        )

        # Detectar tipo si no se proporciona
        if query_type is None:
            query_type = self.prompt_builder.detect_query_type(query)

        llm_response = llm_complete(
            prompt=prompt["user"],
            system=prompt["system"],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            stream=stream,
            stream_callback=stream_callback,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Registrar coste (QUERY)
        tracker = get_tracker()
        tracker.record_generation(
            model=llm_response.model,
            tokens_input=llm_response.tokens_input,
            tokens_output=llm_response.tokens_output,
            query=query
        )

        # Construir respuesta
        generated = GeneratedResponse(
            content=llm_response.content,
            query=query,
            query_type=query_type.value if isinstance(query_type, QueryType) else str(query_type),
            sources_used=[r.chunk_id for r in results],
            model=llm_response.model,
            tokens_input=llm_response.tokens_input,
            tokens_output=llm_response.tokens_output,
            latency_ms=latency_ms,
            metadata={
                "temperature": self.temperature,
                "provider": get_provider_name(),
            }
        )

        logger.info(
            f"Respuesta generada: {llm_response.tokens_output} tokens, "
            f"{latency_ms:.0f}ms, modelo={llm_response.model}"
        )

        return generated
