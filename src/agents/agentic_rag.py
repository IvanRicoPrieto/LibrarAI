"""
Agentic RAG — Loop agéntico con razonamiento iterativo.

Pipeline: Retrieve → Reflect → Decide (buscar más o generar).
El agente reformula queries, cambia estrategia y acumula contexto
iterativamente hasta tener suficiente información o agotar iteraciones.

Mejora la respuesta a preguntas que requieren sintetizar información
dispersa en la biblioteca.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Set

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Estado del agente a lo largo de las iteraciones."""
    original_query: str
    current_query: str
    accumulated_results: List[Any] = field(default_factory=list)
    accumulated_chunk_ids: Set[str] = field(default_factory=set)
    iteration: int = 0
    reasoning_trace: List[str] = field(default_factory=list)
    search_strategies_used: List[str] = field(default_factory=list)
    is_sufficient: bool = False
    confidence: float = 0.0
    max_iterations: int = 4


@dataclass
class ReflectionResult:
    """Resultado de la reflexión del agente."""
    is_sufficient: bool
    confidence: float
    missing_aspects: List[str]
    suggested_reformulation: Optional[str]
    suggested_strategy: Optional[str]
    reasoning: str


class AgenticRAGPipeline:
    """
    Pipeline RAG agéntico con loop iterativo.

    Flujo por iteración:
    1. Retrieve: búsqueda con query actual
    2. Accumulate: merge resultados nuevos (dedup)
    3. Reflect: ¿tengo suficiente contexto?
    4. Decide: si suficiente → generar | si no → reformular y volver a 1
    """

    def __init__(
        self,
        retriever,
        synthesizer,
        max_iterations: int = 4,
        confidence_threshold: float = 0.7,
        min_new_results: int = 2,
    ):
        """
        Args:
            retriever: UnifiedRetriever para búsqueda.
            synthesizer: ResponseSynthesizer para generación.
            max_iterations: Máximo de iteraciones del loop.
            confidence_threshold: Confianza mínima para generar.
            min_new_results: Mínimo de resultados nuevos por iteración.
        """
        self.retriever = retriever
        self.synthesizer = synthesizer
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.min_new_results = min_new_results

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def ask(
        self,
        query: str,
        top_k: int = 10,
        stream: bool = False,
        stream_callback: Optional[Callable] = None,
    ):
        """
        Ejecuta el pipeline agéntico.

        Args:
            query: Pregunta del usuario.
            top_k: Resultados por iteración.
            stream: Si usar streaming.
            stream_callback: Callback para streaming.

        Returns:
            Tuple (GeneratedResponse, List[RetrievalResult], AgentState)
        """
        state = AgentState(
            original_query=query,
            current_query=query,
            max_iterations=self.max_iterations,
        )

        logger.info(f"Agentic RAG: iniciando (max {self.max_iterations} iter)")

        for iteration in range(self.max_iterations):
            state.iteration = iteration + 1
            state.reasoning_trace.append(
                f"--- Iteración {state.iteration} ---"
            )
            state.reasoning_trace.append(f"Query: {state.current_query}")

            # 1. Retrieve
            new_results = self._retrieve(state, top_k)

            # 2. Accumulate (dedup)
            n_new = self._accumulate_results(state, new_results)
            state.reasoning_trace.append(
                f"Nuevos resultados: {n_new} "
                f"(total acumulado: {len(state.accumulated_results)})"
            )

            # Si no hay resultados nuevos significativos, parar
            if n_new < self.min_new_results and iteration > 0:
                state.reasoning_trace.append(
                    "Pocos resultados nuevos. Finalizando búsqueda."
                )
                break

            # 3. Reflect
            reflection = self._reflect(state)
            state.confidence = reflection.confidence
            state.reasoning_trace.append(
                f"Reflexión: suficiente={reflection.is_sufficient} "
                f"(confianza={reflection.confidence:.2f})"
            )

            if reflection.is_sufficient:
                state.is_sufficient = True
                state.reasoning_trace.append(
                    "Contexto suficiente. Procediendo a generar."
                )
                break

            # 4. Decide: reformular query
            if reflection.suggested_reformulation:
                state.current_query = reflection.suggested_reformulation
                state.reasoning_trace.append(
                    f"Reformulando: {state.current_query}"
                )
            elif reflection.missing_aspects:
                # Construir query enfocada en lo que falta
                missing = ", ".join(reflection.missing_aspects[:3])
                state.current_query = (
                    f"{query} (enfoque en: {missing})"
                )
                state.reasoning_trace.append(
                    f"Enfocando en aspectos faltantes: {missing}"
                )

        logger.info(
            f"Agentic RAG: {state.iteration} iteraciones, "
            f"{len(state.accumulated_results)} resultados, "
            f"confianza={state.confidence:.2f}"
        )

        # 5. Generate
        response = self._generate(
            state, stream=stream, callback=stream_callback
        )

        # Añadir metadata del agente
        response.metadata["agentic_rag"] = {
            "iterations": state.iteration,
            "total_results": len(state.accumulated_results),
            "confidence": state.confidence,
            "is_sufficient": state.is_sufficient,
            "reasoning_trace": state.reasoning_trace,
            "queries_used": state.search_strategies_used,
        }

        return response, state.accumulated_results[:top_k], state

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def _retrieve(self, state: AgentState, top_k: int) -> list:
        """Retrieval con query actual."""
        state.search_strategies_used.append(state.current_query)

        try:
            results = self.retriever.search(
                state.current_query, top_k=top_k
            )
            return results
        except Exception as e:
            logger.error(f"Error en retrieval (iter {state.iteration}): {e}")
            return []

    # ------------------------------------------------------------------
    # Accumulate
    # ------------------------------------------------------------------

    def _accumulate_results(
        self, state: AgentState, new_results: list
    ) -> int:
        """Acumula resultados nuevos sin duplicados. Retorna count nuevos."""
        n_new = 0
        for r in new_results:
            chunk_id = getattr(r, "chunk_id", None)
            if chunk_id and chunk_id not in state.accumulated_chunk_ids:
                state.accumulated_chunk_ids.add(chunk_id)
                state.accumulated_results.append(r)
                n_new += 1
        return n_new

    # ------------------------------------------------------------------
    # Reflect
    # ------------------------------------------------------------------

    def _reflect(self, state: AgentState) -> ReflectionResult:
        """
        Evalúa si el contexto acumulado es suficiente para responder.
        """
        if not state.accumulated_results:
            return ReflectionResult(
                is_sufficient=False,
                confidence=0.0,
                missing_aspects=["No hay resultados aún"],
                suggested_reformulation=None,
                suggested_strategy=None,
                reasoning="Sin resultados.",
            )

        try:
            from src.llm_provider import complete as llm_complete

            # Preparar resumen de lo que tenemos
            context_summary = ""
            for idx, r in enumerate(state.accumulated_results[:10]):
                content = getattr(r, "content", str(r))[:200]
                title = getattr(r, "doc_title", "?")
                context_summary += (
                    f"\n[{idx + 1}] ({title}) {content}...\n"
                )

            response = llm_complete(
                prompt=(
                    f'Pregunta original: "{state.original_query}"\n'
                    f"Query actual: \"{state.current_query}\"\n"
                    f"Iteración: {state.iteration}/{state.max_iterations}\n\n"
                    f"Contexto acumulado ({len(state.accumulated_results)} "
                    f"fragmentos):{context_summary}\n\n"
                    "Evalúa:\n"
                    "1. ¿El contexto es suficiente para responder?\n"
                    "2. ¿Qué aspectos de la pregunta NO están cubiertos?\n"
                    "3. Si no es suficiente, ¿cómo reformularías la búsqueda?\n\n"
                    "Responde con JSON:\n"
                    "{\n"
                    '  "is_sufficient": true/false,\n'
                    '  "confidence": 0.0-1.0,\n'
                    '  "missing_aspects": ["..."],\n'
                    '  "suggested_reformulation": "..." o null,\n'
                    '  "reasoning": "..."\n'
                    "}"
                ),
                system=(
                    "Evalúas si un conjunto de fragmentos recuperados es "
                    "suficiente para responder una pregunta. Sé riguroso: "
                    "solo marca como suficiente si la información cubre los "
                    "aspectos principales de la pregunta. Responde SOLO con JSON."
                ),
                temperature=0,
                max_tokens=400,
                json_mode=True,
            )

            return self._parse_reflection(response.content)

        except Exception as e:
            logger.warning(f"Error en reflexión: {e}")
            # Fallback heurístico
            n_results = len(state.accumulated_results)
            confidence = min(0.9, n_results * 0.15)
            return ReflectionResult(
                is_sufficient=confidence >= self.confidence_threshold,
                confidence=confidence,
                missing_aspects=[],
                suggested_reformulation=None,
                suggested_strategy=None,
                reasoning=f"Heurístico: {n_results} resultados → {confidence:.2f}",
            )

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def _generate(
        self,
        state: AgentState,
        stream: bool = False,
        callback=None,
    ):
        """Genera respuesta final con todo el contexto acumulado."""
        # Ordenar por score
        sorted_results = sorted(
            state.accumulated_results,
            key=lambda r: getattr(r, "score", 0),
            reverse=True,
        )

        return self.synthesizer.generate(
            query=state.original_query,
            results=sorted_results,
            stream=stream,
            stream_callback=callback,
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_reflection(raw: str) -> ReflectionResult:
        """Parsea la respuesta JSON de reflexión."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return ReflectionResult(
                is_sufficient=False,
                confidence=0.3,
                missing_aspects=["Error parseando reflexión"],
                suggested_reformulation=None,
                suggested_strategy=None,
                reasoning="JSON parse error",
            )

        return ReflectionResult(
            is_sufficient=bool(parsed.get("is_sufficient", False)),
            confidence=float(parsed.get("confidence", 0.3)),
            missing_aspects=parsed.get("missing_aspects", []),
            suggested_reformulation=parsed.get("suggested_reformulation"),
            suggested_strategy=parsed.get("suggested_strategy"),
            reasoning=parsed.get("reasoning", ""),
        )
