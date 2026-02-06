"""
Self-RAG - RAG Auto-Reflexivo con tokens especiales de reflexión.

Implementa el paradigma Self-RAG donde el modelo:
1. Decide si necesita recuperar información
2. Evalúa la relevancia de lo recuperado
3. Genera con crítica integrada
4. Decide si la respuesta está fundamentada

Basado en: "Self-RAG: Learning to Retrieve, Generate, and Critique"
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class RetrievalDecision(Enum):
    """Decisión sobre si recuperar."""
    RETRIEVE = "retrieve"       # Necesita información externa
    NO_RETRIEVE = "no_retrieve" # Puede responder sin retrieval
    UNCERTAIN = "uncertain"     # Necesita más análisis


class RelevanceScore(Enum):
    """Score de relevancia del contexto recuperado."""
    RELEVANT = "relevant"           # Directamente relevante
    PARTIALLY = "partially"         # Parcialmente relevante
    IRRELEVANT = "irrelevant"       # No relevante


class SupportScore(Enum):
    """Grado de soporte de la respuesta por el contexto."""
    FULLY_SUPPORTED = "fully_supported"     # Totalmente respaldada
    PARTIALLY_SUPPORTED = "partially"       # Parcialmente respaldada
    NO_SUPPORT = "no_support"               # Sin respaldo
    CONTRADICTS = "contradicts"             # Contradice el contexto


class UsefulnessScore(Enum):
    """Utilidad de la respuesta para el usuario."""
    USEFUL = "useful"           # Útil y completa
    PARTIAL = "partial"         # Parcialmente útil
    NOT_USEFUL = "not_useful"   # No útil


@dataclass
class SelfRAGCritique:
    """Crítica auto-generada de la respuesta."""
    retrieval_decision: RetrievalDecision
    relevance_scores: List[RelevanceScore]  # Por cada fuente
    support_score: SupportScore
    usefulness_score: UsefulnessScore
    reasoning: str
    confidence: float  # 0-1
    suggestions: List[str] = field(default_factory=list)

    def is_acceptable(self) -> bool:
        """Determina si la respuesta es aceptable."""
        return (
            self.support_score in [SupportScore.FULLY_SUPPORTED, SupportScore.PARTIALLY_SUPPORTED]
            and self.usefulness_score in [UsefulnessScore.USEFUL, UsefulnessScore.PARTIAL]
            and self.confidence >= 0.6
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval_decision": self.retrieval_decision.value,
            "relevance_scores": [r.value for r in self.relevance_scores],
            "support_score": self.support_score.value,
            "usefulness_score": self.usefulness_score.value,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "suggestions": self.suggestions,
            "is_acceptable": self.is_acceptable()
        }


@dataclass
class SelfRAGState:
    """Estado del pipeline Self-RAG."""
    query: str
    iteration: int = 0
    max_iterations: int = 3
    retrieved_contexts: List[Any] = field(default_factory=list)
    critiques: List[SelfRAGCritique] = field(default_factory=list)
    responses: List[str] = field(default_factory=list)
    final_response: Optional[str] = None
    action_trace: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "iterations": self.iteration,
            "critiques": [c.to_dict() for c in self.critiques],
            "action_trace": self.action_trace,
            "final_response": self.final_response
        }


class SelfRAGPipeline:
    """
    Pipeline Self-RAG con reflexión integrada.

    Flujo:
    1. Analizar query → decidir si necesita retrieval
    2. Si retrieval: recuperar → evaluar relevancia
    3. Generar respuesta con contexto
    4. Criticar respuesta → ¿fundamentada? ¿útil?
    5. Si no aceptable: iterar con más contexto o reformular

    Uso:
        pipeline = SelfRAGPipeline(retriever, synthesizer)
        response, state = pipeline.ask("¿Qué es la decoherencia?")
    """

    RETRIEVAL_DECISION_PROMPT = """Analiza si necesitas información externa para responder esta pregunta.

PREGUNTA: {query}

CONOCIMIENTO PREVIO DISPONIBLE:
{prior_context}

Decide:
- RETRIEVE: Si necesitas información específica de documentos
- NO_RETRIEVE: Si puedes responder con conocimiento general
- UNCERTAIN: Si no estás seguro

Responde JSON:
{{
  "decision": "retrieve|no_retrieve|uncertain",
  "reasoning": "Por qué esta decisión",
  "confidence": 0.0-1.0
}}"""

    RELEVANCE_EVAL_PROMPT = """Evalúa la relevancia de estos fragmentos para responder la pregunta.

PREGUNTA: {query}

FRAGMENTOS:
{contexts}

Para cada fragmento, asigna:
- relevant: Directamente útil para responder
- partially: Algo útil pero no completo
- irrelevant: No ayuda a responder

Responde JSON:
{{
  "evaluations": [
    {{"index": 1, "score": "relevant|partially|irrelevant", "reason": "..."}},
    ...
  ],
  "overall_sufficiency": 0.0-1.0,
  "missing_aspects": ["aspecto que falta", ...]
}}"""

    CRITIQUE_PROMPT = """Critica esta respuesta generada.

PREGUNTA: {query}

CONTEXTO USADO:
{context}

RESPUESTA GENERADA:
{response}

Evalúa:
1. SUPPORT: ¿La respuesta está respaldada por el contexto?
   - fully_supported: Cada afirmación tiene soporte
   - partially: Algunas afirmaciones sin soporte
   - no_support: La mayoría no tiene soporte
   - contradicts: Contradice el contexto

2. USEFULNESS: ¿Responde la pregunta del usuario?
   - useful: Responde completamente
   - partial: Responde parcialmente
   - not_useful: No responde

Responde JSON:
{{
  "support_score": "fully_supported|partially|no_support|contradicts",
  "usefulness_score": "useful|partial|not_useful",
  "confidence": 0.0-1.0,
  "unsupported_claims": ["claim sin soporte", ...],
  "suggestions": ["cómo mejorar", ...],
  "reasoning": "Análisis detallado"
}}"""

    def __init__(
        self,
        retriever,
        synthesizer,
        max_iterations: int = 3,
        min_relevance_ratio: float = 0.5,
        require_support: bool = True
    ):
        """
        Args:
            retriever: Retriever a usar (UnifiedRetriever o similar)
            synthesizer: Sintetizador de respuestas
            max_iterations: Máximo de iteraciones de refinamiento
            min_relevance_ratio: Ratio mínimo de contextos relevantes
            require_support: Si exigir que la respuesta esté fundamentada
        """
        self.retriever = retriever
        self.synthesizer = synthesizer
        self.max_iterations = max_iterations
        self.min_relevance_ratio = min_relevance_ratio
        self.require_support = require_support

    def ask(
        self,
        query: str,
        top_k: int = 10,
        stream: bool = False,
        stream_callback=None
    ) -> Tuple[Any, SelfRAGState]:
        """
        Procesa una consulta con reflexión Self-RAG.

        Args:
            query: Pregunta del usuario
            top_k: Número de contextos a recuperar
            stream: Si usar streaming para la respuesta final
            stream_callback: Callback para streaming

        Returns:
            Tuple (GeneratedResponse, SelfRAGState)
        """
        state = SelfRAGState(
            query=query,
            max_iterations=self.max_iterations
        )

        # 1. Decidir si necesita retrieval
        decision = self._decide_retrieval(query, state)
        state.action_trace.append(f"Retrieval decision: {decision.value}")

        if decision == RetrievalDecision.NO_RETRIEVE:
            # Generar sin retrieval (raro para RAG, pero posible)
            state.action_trace.append("Generating without retrieval")
            response = self._generate_without_context(query, stream, stream_callback)
            state.final_response = response.content
            return response, state

        # 2. Loop de retrieval-generate-critique
        while state.iteration < self.max_iterations:
            state.iteration += 1
            state.action_trace.append(f"Iteration {state.iteration}")

            # Recuperar contexto
            contexts = self._retrieve(query, top_k, state)
            state.retrieved_contexts = contexts

            if not contexts:
                state.action_trace.append("No contexts found, generating best effort")
                break

            # Evaluar relevancia
            relevance_scores = self._evaluate_relevance(query, contexts)
            state.action_trace.append(
                f"Relevance: {sum(1 for r in relevance_scores if r == RelevanceScore.RELEVANT)}/{len(relevance_scores)} relevant"
            )

            # Filtrar contextos irrelevantes
            relevant_contexts = [
                c for c, r in zip(contexts, relevance_scores)
                if r != RelevanceScore.IRRELEVANT
            ]

            if not relevant_contexts:
                state.action_trace.append("No relevant contexts, reformulating query")
                query = self._reformulate_query(query, state)
                continue

            # Generar respuesta
            response = self.synthesizer.generate(
                query=query,
                results=relevant_contexts,
                stream=False  # No stream en iteraciones intermedias
            )
            state.responses.append(response.content)

            # Criticar respuesta
            critique = self._critique_response(query, relevant_contexts, response.content)
            critique.relevance_scores = relevance_scores
            state.critiques.append(critique)

            state.action_trace.append(
                f"Critique: support={critique.support_score.value}, "
                f"useful={critique.usefulness_score.value}, "
                f"confidence={critique.confidence:.2f}"
            )

            # ¿Respuesta aceptable?
            if critique.is_acceptable():
                state.action_trace.append("Response accepted")
                state.final_response = response.content

                # Re-generar con stream si se solicitó
                if stream and stream_callback:
                    response = self.synthesizer.generate(
                        query=query,
                        results=relevant_contexts,
                        stream=True,
                        stream_callback=stream_callback
                    )

                response.metadata["self_rag"] = state.to_dict()
                return response, state

            # No aceptable: intentar mejorar
            if critique.support_score == SupportScore.NO_SUPPORT:
                # Necesita más/mejor contexto
                state.action_trace.append("Need better context, expanding search")
                top_k = int(top_k * 1.5)
            elif critique.usefulness_score == UsefulnessScore.NOT_USEFUL:
                # Query mal interpretada
                state.action_trace.append("Query misunderstood, reformulating")
                query = self._reformulate_query(query, state, critique.suggestions)

        # Max iteraciones alcanzadas, usar última respuesta
        state.action_trace.append("Max iterations reached, using best response")
        if state.responses:
            state.final_response = state.responses[-1]
            # Generar respuesta final
            response = self.synthesizer.generate(
                query=state.query,
                results=state.retrieved_contexts,
                stream=stream,
                stream_callback=stream_callback
            )
            response.metadata["self_rag"] = state.to_dict()
            return response, state
        else:
            # Fallback: generar sin reflexión
            response = self.synthesizer.generate(
                query=state.query,
                results=state.retrieved_contexts or [],
                stream=stream,
                stream_callback=stream_callback
            )
            response.metadata["self_rag"] = state.to_dict()
            return response, state

    def _decide_retrieval(
        self,
        query: str,
        state: SelfRAGState
    ) -> RetrievalDecision:
        """Decide si la query necesita retrieval."""
        from src.llm_provider import complete as llm_complete

        # Para un sistema RAG, casi siempre queremos retrieval
        # Pero evaluamos por si es una pregunta muy general
        try:
            prior_context = "Sistema RAG sobre computación cuántica e información cuántica."

            response = llm_complete(
                prompt=self.RETRIEVAL_DECISION_PROMPT.format(
                    query=query,
                    prior_context=prior_context
                ),
                system="Eres un analizador de queries para RAG. Responde JSON.",
                temperature=0,
                max_tokens=200,
                json_mode=True
            )

            data = json.loads(response.content)
            decision = data.get("decision", "retrieve").lower()

            if decision == "no_retrieve":
                return RetrievalDecision.NO_RETRIEVE
            elif decision == "uncertain":
                return RetrievalDecision.UNCERTAIN
            else:
                return RetrievalDecision.RETRIEVE

        except Exception as e:
            logger.warning(f"Error en decisión de retrieval: {e}")
            return RetrievalDecision.RETRIEVE  # Default seguro

    def _retrieve(
        self,
        query: str,
        top_k: int,
        state: SelfRAGState
    ) -> List[Any]:
        """Recupera contexto relevante."""
        try:
            return self.retriever.search(query, top_k=top_k)
        except Exception as e:
            logger.error(f"Error en retrieval: {e}")
            return []

    def _evaluate_relevance(
        self,
        query: str,
        contexts: List[Any]
    ) -> List[RelevanceScore]:
        """Evalúa relevancia de cada contexto."""
        from src.llm_provider import complete as llm_complete

        try:
            # Formatear contextos
            contexts_text = "\n\n".join([
                f"[{i+1}] {c.doc_title}: {c.content[:500]}..."
                for i, c in enumerate(contexts[:10])
            ])

            response = llm_complete(
                prompt=self.RELEVANCE_EVAL_PROMPT.format(
                    query=query,
                    contexts=contexts_text
                ),
                system="Eres un evaluador de relevancia para RAG. Responde JSON.",
                temperature=0,
                max_tokens=500,
                json_mode=True
            )

            data = json.loads(response.content)
            evaluations = data.get("evaluations", [])

            scores = []
            for i in range(len(contexts)):
                eval_data = next(
                    (e for e in evaluations if e.get("index") == i + 1),
                    {"score": "partially"}
                )
                score_str = eval_data.get("score", "partially").lower()

                if score_str == "relevant":
                    scores.append(RelevanceScore.RELEVANT)
                elif score_str == "irrelevant":
                    scores.append(RelevanceScore.IRRELEVANT)
                else:
                    scores.append(RelevanceScore.PARTIALLY)

            return scores

        except Exception as e:
            logger.warning(f"Error evaluando relevancia: {e}")
            # Default: todo parcialmente relevante
            return [RelevanceScore.PARTIALLY] * len(contexts)

    def _critique_response(
        self,
        query: str,
        contexts: List[Any],
        response: str
    ) -> SelfRAGCritique:
        """Critica la respuesta generada."""
        from src.llm_provider import complete as llm_complete

        try:
            context_text = "\n\n".join([
                f"[{i+1}] {c.content[:300]}..."
                for i, c in enumerate(contexts[:5])
            ])

            llm_response = llm_complete(
                prompt=self.CRITIQUE_PROMPT.format(
                    query=query,
                    context=context_text,
                    response=response
                ),
                system="Eres un crítico de respuestas RAG. Responde JSON.",
                temperature=0,
                max_tokens=500,
                json_mode=True
            )

            data = json.loads(llm_response.content)

            support_str = data.get("support_score", "partially").lower()
            support_map = {
                "fully_supported": SupportScore.FULLY_SUPPORTED,
                "partially": SupportScore.PARTIALLY_SUPPORTED,
                "no_support": SupportScore.NO_SUPPORT,
                "contradicts": SupportScore.CONTRADICTS
            }
            support = support_map.get(support_str, SupportScore.PARTIALLY_SUPPORTED)

            useful_str = data.get("usefulness_score", "partial").lower()
            useful_map = {
                "useful": UsefulnessScore.USEFUL,
                "partial": UsefulnessScore.PARTIAL,
                "not_useful": UsefulnessScore.NOT_USEFUL
            }
            usefulness = useful_map.get(useful_str, UsefulnessScore.PARTIAL)

            return SelfRAGCritique(
                retrieval_decision=RetrievalDecision.RETRIEVE,
                relevance_scores=[],  # Se llena después
                support_score=support,
                usefulness_score=usefulness,
                reasoning=data.get("reasoning", ""),
                confidence=data.get("confidence", 0.7),
                suggestions=data.get("suggestions", [])
            )

        except Exception as e:
            logger.warning(f"Error en crítica: {e}")
            return SelfRAGCritique(
                retrieval_decision=RetrievalDecision.RETRIEVE,
                relevance_scores=[],
                support_score=SupportScore.PARTIALLY_SUPPORTED,
                usefulness_score=UsefulnessScore.PARTIAL,
                reasoning=f"Error en evaluación: {e}",
                confidence=0.5,
                suggestions=[]
            )

    def _reformulate_query(
        self,
        query: str,
        state: SelfRAGState,
        suggestions: List[str] = None
    ) -> str:
        """Reformula la query basándose en feedback."""
        from src.llm_provider import complete as llm_complete

        try:
            suggestions_text = "\n".join(suggestions or [])
            previous_attempts = "\n".join(state.responses[-2:]) if state.responses else "Ninguno"

            response = llm_complete(
                prompt=f"""Reformula esta pregunta para obtener mejores resultados de búsqueda.

PREGUNTA ORIGINAL: {query}

SUGERENCIAS DE MEJORA:
{suggestions_text}

INTENTOS PREVIOS QUE NO FUNCIONARON:
{previous_attempts}

Genera una reformulación más específica y clara.
Responde SOLO con la nueva pregunta, sin explicaciones.""",
                system="Eres un experto en reformulación de queries.",
                temperature=0.3,
                max_tokens=150
            )

            reformulated = response.content.strip()
            if reformulated and reformulated != query:
                logger.info(f"Query reformulada: {query} -> {reformulated}")
                return reformulated

        except Exception as e:
            logger.warning(f"Error reformulando query: {e}")

        return query

    def _generate_without_context(
        self,
        query: str,
        stream: bool,
        stream_callback
    ) -> Any:
        """Genera respuesta sin contexto (caso raro)."""
        from ..generation.synthesizer import GeneratedResponse

        # Esto no debería pasar frecuentemente en un RAG
        return GeneratedResponse(
            content="Esta pregunta podría requerir información específica de tu biblioteca. "
                   "¿Podrías reformularla o especificar qué aspecto te interesa?",
            query=query,
            query_type="no_retrieval",
            sources_used=[],
            model="self_rag_decision",
            tokens_input=0,
            tokens_output=0,
            latency_ms=0,
            metadata={"self_rag": {"decision": "no_retrieve"}}
        )
