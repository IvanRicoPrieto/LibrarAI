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

        # Para queries MATHEMATICAL con computación habilitada, usar orquestador
        if query_type == QueryType.MATHEMATICAL and self._math_computation_enabled():
            return self._generate_with_computation(
                query=query,
                results=results,
                prompt=prompt,
                query_type=query_type,
                stream=stream,
                stream_callback=stream_callback,
            )

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

    def _math_computation_enabled(self) -> bool:
        """Comprueba si la computación matemática está habilitada en settings."""
        return self._get_math_config().get("enabled", False)

    def _multi_agent_enabled(self) -> bool:
        """Comprueba si el modo multi-agente está habilitado."""
        return self._get_math_config().get("multi_agent", {}).get("enabled", False)

    def _get_math_config(self) -> Dict:
        """Lee la configuración de math_computation de settings.yaml."""
        try:
            from pathlib import Path
            import yaml
            settings_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
            if settings_path.exists():
                with open(settings_path) as f:
                    settings = yaml.safe_load(f)
                return settings.get("math_computation", {})
        except Exception:
            pass
        return {}

    def _generate_with_computation(
        self,
        query: str,
        results: List[RetrievalResult],
        prompt: Dict[str, str],
        query_type: QueryType,
        stream: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> GeneratedResponse:
        """
        Genera respuesta usando computación matemática.

        Dos modos:
        - Básico (default): MathComputationOrchestrator — loop ToRA bidireccional
        - Multi-agente: MultiAgentOrchestrator — Planner/Calculator/Verifier/Synthesizer
          con provenance W3C PROV (habilitado con math_computation.multi_agent.enabled)
        """
        if self._multi_agent_enabled():
            return self._generate_with_multi_agent(
                query, results, prompt, query_type, stream, stream_callback
            )
        return self._generate_with_basic_loop(
            query, results, prompt, query_type, stream, stream_callback
        )

    def _generate_with_basic_loop(
        self,
        query: str,
        results: List[RetrievalResult],
        prompt: Dict[str, str],
        query_type: QueryType,
        stream: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> GeneratedResponse:
        """Loop de computación bidireccional básico (Fase 1)."""
        from src.llm_provider import get_provider_name
        from ..math.orchestrator import MathComputationOrchestrator

        import time
        start_time = time.time()

        orchestrator = MathComputationOrchestrator()

        final_text, steps, artifacts = orchestrator.run(
            query=query,
            retrieval_context="\n\n".join(
                f"[{i+1}] {r.content}" for i, r in enumerate(results)
            ),
            system_prompt=prompt["system"],
            user_template=prompt["user"],
            stream_callback=stream_callback if stream else None,
        )

        latency_ms = (time.time() - start_time) * 1000

        generated = GeneratedResponse(
            content=final_text,
            query=query,
            query_type=query_type.value,
            sources_used=[r.chunk_id for r in results],
            model=get_provider_name(),
            tokens_input=0,
            tokens_output=0,
            latency_ms=latency_ms,
            metadata={
                "temperature": self.temperature,
                "provider": get_provider_name(),
                "math_computation": True,
                "math_mode": "basic_loop",
                "computation_steps": len(steps),
                "computation_iterations": max((s.iteration for s in steps), default=0) + 1,
                "math_artifacts": [a.to_dict() for a in artifacts],
                "computation_trace": [s.to_dict() for s in steps],
            }
        )

        logger.info(
            f"Respuesta matemática (loop básico): {len(steps)} pasos, "
            f"{len(artifacts)} artefactos, {latency_ms:.0f}ms"
        )

        return generated

    def _generate_with_multi_agent(
        self,
        query: str,
        results: List[RetrievalResult],
        prompt: Dict[str, str],
        query_type: QueryType,
        stream: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> GeneratedResponse:
        """
        Generación multi-agente con provenance W3C PROV (Fase 3).

        Usa Planner → Calculator → Verifier → Synthesizer con
        derivaciones paso a paso verificadas.
        """
        from src.llm_provider import complete as llm_complete, get_provider_name
        from ..math.agents import MultiAgentOrchestrator

        import time
        start_time = time.time()

        config = self._get_math_config()
        ma_config = config.get("multi_agent", {})

        orchestrator = MultiAgentOrchestrator(
            max_steps=ma_config.get("max_steps", 10),
        )

        retrieval_context = "\n\n".join(
            f"[{i+1}] {r.content}" for i, r in enumerate(results)
        )

        result = orchestrator.run(
            query=query,
            retrieval_context=retrieval_context,
            llm_fn=llm_complete,
            stream_callback=stream_callback if stream else None,
        )

        latency_ms = (time.time() - start_time) * 1000

        generated = GeneratedResponse(
            content=result["response"],
            query=query,
            query_type=query_type.value,
            sources_used=[r.chunk_id for r in results],
            model=get_provider_name(),
            tokens_input=0,
            tokens_output=0,
            latency_ms=latency_ms,
            metadata={
                "temperature": self.temperature,
                "provider": get_provider_name(),
                "math_computation": True,
                "math_mode": "multi_agent",
                "steps_verified": result["steps_verified"],
                "steps_total": result["steps_total"],
                "math_artifacts": [a.to_dict() for a in result["artifacts"]],
                "provenance": result["provenance"],
                "plan": result["plan"].to_dict(),
            }
        )

        logger.info(
            f"Respuesta matemática (multi-agente): "
            f"{result['steps_verified']}/{result['steps_total']} pasos verificados, "
            f"{len(result['artifacts'])} artefactos, {latency_ms:.0f}ms"
        )

        return generated
