"""
Corrective RAG (CRAG) — Validación pre-generación de relevancia.

Después del retrieval y antes de la generación, evalúa la relevancia de
cada documento recuperado. Clasifica como CORRECT / AMBIGUOUS / INCORRECT.

Si la mayoría son INCORRECT → reformula la query y re-busca.
Si son mixtos → filtra y hace merge.
Si son CORRECT → pasa directo.

Esto evita gastar tokens generando sobre contexto irrelevante y reduce
alucinaciones en origen.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class RelevanceVerdict(Enum):
    """Veredicto de relevancia de un resultado."""
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@dataclass
class RelevanceAssessment:
    """Evaluación de relevancia de un chunk para una query."""
    chunk_id: str
    verdict: RelevanceVerdict
    confidence: float
    reasoning: str
    relevance_score: float


@dataclass
class CRAGResult:
    """Resultado del proceso Corrective RAG."""
    original_results: List[Any]
    corrected_results: List[Any]
    assessments: List[RelevanceAssessment]
    action_taken: str  # "pass", "filter", "reformulate", "merge"
    reformulated_query: Optional[str]
    stats: Dict[str, Any] = field(default_factory=dict)


class CorrectiveRAG:
    """
    Implementa Corrective RAG: evalúa relevancia antes de generar.

    Flujo:
    1. Evaluar relevancia de cada resultado (LLM o heurístico)
    2. Decidir acción: pass / filter / reformulate / merge
    3. Si reformulate → reformular query y re-buscar
    4. Retornar resultados corregidos
    """

    def __init__(
        self,
        retriever=None,
        correct_threshold: float = 0.7,
        incorrect_threshold: float = 0.3,
        majority_threshold: float = 0.5,
        use_llm_assessment: bool = True,
    ):
        """
        Args:
            retriever: UnifiedRetriever para re-búsqueda.
            correct_threshold: Score mínimo para CORRECT.
            incorrect_threshold: Score máximo para INCORRECT.
            majority_threshold: Fracción de INCORRECT para reformular.
            use_llm_assessment: Si usar LLM para evaluar relevancia.
        """
        self.retriever = retriever
        self.correct_threshold = correct_threshold
        self.incorrect_threshold = incorrect_threshold
        self.majority_threshold = majority_threshold
        self.use_llm_assessment = use_llm_assessment

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def correct(
        self,
        query: str,
        results: List[Any],
        top_k: int = 10,
    ) -> CRAGResult:
        """
        Evalúa y corrige los resultados del retrieval.

        Args:
            query: Query original.
            results: Resultados del retrieval.
            top_k: Número de resultados finales deseados.

        Returns:
            CRAGResult con resultados corregidos.
        """
        if not results:
            return CRAGResult(
                original_results=results,
                corrected_results=results,
                assessments=[],
                action_taken="pass",
                reformulated_query=None,
                stats={"reason": "no_results"},
            )

        # 1. Evaluar relevancia
        if self.use_llm_assessment:
            assessments = self._assess_relevance_llm(query, results)
        else:
            assessments = self._assess_relevance_heuristic(query, results)

        # 2. Contar veredictos
        n_correct = sum(
            1 for a in assessments if a.verdict == RelevanceVerdict.CORRECT
        )
        n_ambiguous = sum(
            1 for a in assessments if a.verdict == RelevanceVerdict.AMBIGUOUS
        )
        n_incorrect = sum(
            1 for a in assessments if a.verdict == RelevanceVerdict.INCORRECT
        )
        total = len(assessments)

        stats = {
            "correct": n_correct,
            "ambiguous": n_ambiguous,
            "incorrect": n_incorrect,
            "total": total,
        }

        # 3. Decidir acción
        incorrect_ratio = n_incorrect / total if total > 0 else 0

        if incorrect_ratio <= 0.2:
            # Mayoría correcta → pasar tal cual
            action = "pass"
            corrected = results[:top_k]
            reformulated = None

        elif incorrect_ratio >= self.majority_threshold:
            # Mayoría incorrecta → reformular y re-buscar
            action = "reformulate"
            reformulated = self._reformulate_query(query, assessments)

            if self.retriever and reformulated:
                new_results = self.retriever.search(
                    reformulated, top_k=top_k
                )
                # Merge: mantener los correctos originales + nuevos
                relevant_originals = self._filter_relevant(
                    results, assessments
                )
                corrected = self._merge_and_dedup(
                    relevant_originals, new_results, top_k
                )
            else:
                corrected = self._filter_relevant(results, assessments)
                corrected = corrected[:top_k]

        else:
            # Mixto → filtrar irrelevantes
            action = "filter"
            corrected = self._filter_relevant(results, assessments)
            corrected = corrected[:top_k]
            reformulated = None

        stats["action"] = action
        if reformulated:
            stats["reformulated_query"] = reformulated

        logger.info(
            f"CRAG: {action} "
            f"(C={n_correct}, A={n_ambiguous}, I={n_incorrect})"
        )

        return CRAGResult(
            original_results=results,
            corrected_results=corrected,
            assessments=assessments,
            action_taken=action,
            reformulated_query=reformulated,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Evaluación de relevancia con LLM
    # ------------------------------------------------------------------

    def _assess_relevance_llm(
        self, query: str, results: List[Any]
    ) -> List[RelevanceAssessment]:
        """Evalúa relevancia usando LLM en batch."""
        try:
            from src.llm_provider import complete as llm_complete

            # Construir prompt con todos los chunks
            chunks_text = ""
            for idx, r in enumerate(results[:15]):
                content_preview = getattr(r, "content", str(r))[:300]
                chunks_text += (
                    f"\n--- Documento {idx} ---\n"
                    f"Contenido: {content_preview}\n"
                )

            response = llm_complete(
                prompt=(
                    f'Query del usuario: "{query}"\n\n'
                    f"Documentos recuperados:{chunks_text}\n\n"
                    "Para cada documento, evalúa si es relevante para la query.\n"
                    "Responde con un JSON array donde cada elemento tiene:\n"
                    '- "index": número del documento\n'
                    '- "verdict": "correct", "ambiguous" o "incorrect"\n'
                    '- "confidence": 0.0 a 1.0\n'
                    '- "reasoning": explicación breve\n'
                ),
                system=(
                    "Evalúas la relevancia de documentos recuperados para una query. "
                    "CORRECT = claramente relevante. "
                    "AMBIGUOUS = parcialmente relevante o tangencial. "
                    "INCORRECT = no relevante para la query. "
                    "Responde SOLO con JSON."
                ),
                temperature=0,
                max_tokens=100 * len(results),
                json_mode=True,
            )

            return self._parse_assessments(response.content, results)

        except Exception as e:
            logger.warning(f"Error en evaluación LLM: {e}. Usando heurística.")
            return self._assess_relevance_heuristic(query, results)

    def _assess_relevance_heuristic(
        self, query: str, results: List[Any]
    ) -> List[RelevanceAssessment]:
        """Evaluación heurística basada en keyword overlap."""
        import re
        import unicodedata

        def normalize(text: str) -> set:
            text = unicodedata.normalize("NFD", text.lower())
            text = re.sub(r"[\u0300-\u036f]", "", text)
            words = re.findall(r"\b\w{3,}\b", text)
            return set(words)

        query_terms = normalize(query)
        assessments = []

        for r in results:
            content = getattr(r, "content", str(r))
            doc_terms = normalize(content)

            if not query_terms:
                overlap = 0.0
            else:
                overlap = len(query_terms & doc_terms) / len(query_terms)

            if overlap >= self.correct_threshold:
                verdict = RelevanceVerdict.CORRECT
            elif overlap <= self.incorrect_threshold:
                verdict = RelevanceVerdict.INCORRECT
            else:
                verdict = RelevanceVerdict.AMBIGUOUS

            chunk_id = getattr(r, "chunk_id", "unknown")
            assessments.append(
                RelevanceAssessment(
                    chunk_id=chunk_id,
                    verdict=verdict,
                    confidence=overlap,
                    reasoning=f"Keyword overlap: {overlap:.2f}",
                    relevance_score=overlap,
                )
            )

        return assessments

    # ------------------------------------------------------------------
    # Reformulación
    # ------------------------------------------------------------------

    def _reformulate_query(
        self, query: str, assessments: List[RelevanceAssessment]
    ) -> Optional[str]:
        """Reformula la query basándose en los problemas detectados."""
        try:
            from src.llm_provider import complete as llm_complete

            # Resumir problemas
            issues = [
                a.reasoning
                for a in assessments
                if a.verdict == RelevanceVerdict.INCORRECT
            ][:5]

            response = llm_complete(
                prompt=(
                    f'Query original: "{query}"\n\n'
                    f"Problemas con los documentos recuperados:\n"
                    + "\n".join(f"- {i}" for i in issues)
                    + "\n\n"
                    "Reformula la query para obtener resultados más relevantes. "
                    "Usa términos más específicos o sinónimos. "
                    "Responde SOLO con la query reformulada, sin explicación."
                ),
                system=(
                    "Reformulas queries de búsqueda para mejorar la relevancia "
                    "de los resultados. Responde solo con la query reformulada."
                ),
                temperature=0.3,
                max_tokens=200,
            )
            reformulated = response.content.strip().strip('"')
            logger.info(f"CRAG reformulación: '{query}' → '{reformulated}'")
            return reformulated

        except Exception as e:
            logger.warning(f"Error reformulando query: {e}")
            return None

    # ------------------------------------------------------------------
    # Filtrado y merge
    # ------------------------------------------------------------------

    def _filter_relevant(
        self, results: List[Any], assessments: List[RelevanceAssessment]
    ) -> List[Any]:
        """Mantiene solo resultados CORRECT y AMBIGUOUS."""
        relevant = []
        for r, a in zip(results, assessments):
            if a.verdict != RelevanceVerdict.INCORRECT:
                relevant.append(r)
        return relevant

    @staticmethod
    def _merge_and_dedup(
        originals: List[Any],
        new_results: List[Any],
        top_k: int,
    ) -> List[Any]:
        """Merge sin duplicados, priorizando originales."""
        seen_ids = set()
        merged = []

        for r in originals:
            chunk_id = getattr(r, "chunk_id", id(r))
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged.append(r)

        for r in new_results:
            chunk_id = getattr(r, "chunk_id", id(r))
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged.append(r)

        return merged[:top_k]

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_assessments(
        self, raw: str, results: List[Any]
    ) -> List[RelevanceAssessment]:
        """Parsea la respuesta JSON del LLM de evaluación."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return self._assess_relevance_heuristic("", results)

        if isinstance(parsed, dict):
            for val in parsed.values():
                if isinstance(val, list):
                    parsed = val
                    break

        if not isinstance(parsed, list):
            return self._assess_relevance_heuristic("", results)

        assessments = []
        for item in parsed:
            if not isinstance(item, dict):
                continue

            idx = item.get("index", len(assessments))
            if idx >= len(results):
                continue

            verdict_str = item.get("verdict", "ambiguous").lower()
            verdict_map = {
                "correct": RelevanceVerdict.CORRECT,
                "ambiguous": RelevanceVerdict.AMBIGUOUS,
                "incorrect": RelevanceVerdict.INCORRECT,
            }
            verdict = verdict_map.get(verdict_str, RelevanceVerdict.AMBIGUOUS)

            chunk_id = getattr(results[idx], "chunk_id", f"idx_{idx}")
            assessments.append(
                RelevanceAssessment(
                    chunk_id=chunk_id,
                    verdict=verdict,
                    confidence=float(item.get("confidence", 0.5)),
                    reasoning=item.get("reasoning", ""),
                    relevance_score=float(item.get("confidence", 0.5)),
                )
            )

        # Rellenar si faltan
        while len(assessments) < len(results):
            idx = len(assessments)
            chunk_id = getattr(results[idx], "chunk_id", f"idx_{idx}")
            assessments.append(
                RelevanceAssessment(
                    chunk_id=chunk_id,
                    verdict=RelevanceVerdict.AMBIGUOUS,
                    confidence=0.5,
                    reasoning="No evaluado",
                    relevance_score=0.5,
                )
            )

        return assessments
