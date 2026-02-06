"""
Multi-Query RAG - Expansión de queries para mejorar recall.

Genera múltiples variaciones de la query del usuario para recuperar
chunks que podrían perderse con una sola formulación.

Estrategias:
1. Reformulación semántica (parafraseo)
2. Descomposición en sub-preguntas
3. Expansión con términos relacionados
4. Generación de queries hipotéticas (estilo HyDE pero para queries)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class ExpansionStrategy(Enum):
    """Estrategias de expansión de query."""
    PARAPHRASE = "paraphrase"           # Reformulación semántica
    DECOMPOSE = "decompose"              # Sub-preguntas
    KEYWORD_EXPAND = "keyword_expand"    # Términos relacionados
    PERSPECTIVE = "perspective"          # Diferentes ángulos/perspectivas
    ALL = "all"                          # Todas las estrategias


@dataclass
class ExpandedQuery:
    """Query expandida con metadata."""
    original: str
    variations: List[str]
    strategy_used: ExpansionStrategy
    reasoning: str = ""

    def all_queries(self) -> List[str]:
        """Retorna original + variaciones sin duplicados."""
        seen = {self.original.lower().strip()}
        result = [self.original]
        for v in self.variations:
            normalized = v.lower().strip()
            if normalized not in seen and normalized:
                seen.add(normalized)
                result.append(v)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "variations": self.variations,
            "strategy": self.strategy_used.value,
            "reasoning": self.reasoning,
            "total_queries": len(self.all_queries())
        }


class MultiQueryExpander:
    """
    Expande queries usando LLM para mejorar recall en retrieval.

    Uso:
        expander = MultiQueryExpander()
        expanded = expander.expand("¿Qué es el algoritmo de Shor?")
        for query in expanded.all_queries():
            results.extend(retriever.search(query))
    """

    EXPANSION_PROMPT = """Eres un experto en reformulación de preguntas para sistemas de búsqueda.

Dada la siguiente pregunta del usuario sobre computación cuántica/información cuántica, genera {n_variations} variaciones que:
1. Mantengan la intención original
2. Usen diferentes términos y formulaciones
3. Cubran diferentes aspectos o ángulos de la pregunta
4. Puedan recuperar información complementaria

PREGUNTA ORIGINAL: {query}

ESTRATEGIA: {strategy_description}

Responde con un JSON:
{{
  "variations": ["variación 1", "variación 2", ...],
  "reasoning": "Breve explicación de por qué estas variaciones ayudan"
}}

IMPORTANTE:
- Las variaciones deben ser preguntas completas y autocontenidas
- No repitas la pregunta original
- Cada variación debe aportar algo diferente
- Máximo {n_variations} variaciones

Responde SOLO con el JSON."""

    STRATEGY_DESCRIPTIONS = {
        ExpansionStrategy.PARAPHRASE:
            "Genera paráfrasis semánticas usando diferentes palabras y estructuras gramaticales.",
        ExpansionStrategy.DECOMPOSE:
            "Descompón en sub-preguntas más específicas que en conjunto respondan la original.",
        ExpansionStrategy.KEYWORD_EXPAND:
            "Expande con términos técnicos relacionados, sinónimos y conceptos asociados.",
        ExpansionStrategy.PERSPECTIVE:
            "Reformula desde diferentes perspectivas: teórica, práctica, histórica, comparativa.",
        ExpansionStrategy.ALL:
            "Combina: paráfrasis, sub-preguntas, términos relacionados y diferentes perspectivas."
    }

    def __init__(
        self,
        n_variations: int = 4,
        default_strategy: ExpansionStrategy = ExpansionStrategy.ALL,
        use_llm: bool = True
    ):
        """
        Args:
            n_variations: Número de variaciones a generar
            default_strategy: Estrategia por defecto
            use_llm: Si usar LLM (False = heurísticas simples)
        """
        self.n_variations = n_variations
        self.default_strategy = default_strategy
        self.use_llm = use_llm

    def expand(
        self,
        query: str,
        strategy: Optional[ExpansionStrategy] = None,
        n_variations: Optional[int] = None
    ) -> ExpandedQuery:
        """
        Expande una query en múltiples variaciones.

        Args:
            query: Query original del usuario
            strategy: Estrategia de expansión (default: self.default_strategy)
            n_variations: Número de variaciones (default: self.n_variations)

        Returns:
            ExpandedQuery con variaciones
        """
        strategy = strategy or self.default_strategy
        n_variations = n_variations or self.n_variations

        if self.use_llm:
            return self._expand_with_llm(query, strategy, n_variations)
        else:
            return self._expand_heuristic(query, strategy, n_variations)

    def _expand_with_llm(
        self,
        query: str,
        strategy: ExpansionStrategy,
        n_variations: int
    ) -> ExpandedQuery:
        """Expansión usando LLM."""
        from src.llm_provider import complete as llm_complete

        prompt = self.EXPANSION_PROMPT.format(
            query=query,
            n_variations=n_variations,
            strategy_description=self.STRATEGY_DESCRIPTIONS[strategy]
        )

        try:
            response = llm_complete(
                prompt=prompt,
                system="Eres un experto en reformulación de queries para RAG. Responde solo JSON válido.",
                temperature=0.7,  # Un poco de creatividad
                max_tokens=500,
                json_mode=True
            )

            data = json.loads(response.content)
            variations = data.get("variations", [])[:n_variations]
            reasoning = data.get("reasoning", "")

            logger.debug(f"Multi-Query: {len(variations)} variaciones generadas")

            return ExpandedQuery(
                original=query,
                variations=variations,
                strategy_used=strategy,
                reasoning=reasoning
            )

        except Exception as e:
            logger.warning(f"Error en expansión LLM, usando heurística: {e}")
            return self._expand_heuristic(query, strategy, n_variations)

    def _expand_heuristic(
        self,
        query: str,
        strategy: ExpansionStrategy,
        n_variations: int
    ) -> ExpandedQuery:
        """Expansión usando heurísticas simples (fallback sin LLM)."""
        variations = []

        # Heurística básica: diferentes formulaciones
        query_lower = query.lower().strip()

        # 1. Convertir pregunta a afirmación buscable
        if query_lower.startswith("¿qué es"):
            concept = query_lower.replace("¿qué es", "").replace("?", "").strip()
            variations.append(f"definición de {concept}")
            variations.append(f"{concept} explicación")

        elif query_lower.startswith("¿cómo"):
            action = query_lower.replace("¿cómo", "").replace("?", "").strip()
            variations.append(f"procedimiento para {action}")
            variations.append(f"pasos para {action}")

        elif "diferencia" in query_lower or "compara" in query_lower:
            variations.append(query.replace("diferencia", "comparación"))
            variations.append(query.replace("compara", "contrasta"))

        # 2. Añadir variantes con términos comunes en quantum computing
        quantum_synonyms = {
            "qubit": ["bit cuántico", "sistema de dos niveles"],
            "entrelazamiento": ["entanglement", "correlación cuántica"],
            "superposición": ["superposition", "estado superpuesto"],
            "medición": ["medida", "observación", "colapso"],
            "algoritmo": ["protocolo", "procedimiento"],
        }

        for term, synonyms in quantum_synonyms.items():
            if term in query_lower:
                for syn in synonyms[:1]:
                    variations.append(query.replace(term, syn))

        # 3. Variante en inglés (muchos textos están en inglés)
        # Solo si la query parece estar en español
        if "¿" in query or "qué" in query_lower:
            # Simplificación: no traducimos, solo añadimos términos clave
            pass

        # Limitar y deduplicar
        variations = list(dict.fromkeys(variations))[:n_variations]

        return ExpandedQuery(
            original=query,
            variations=variations,
            strategy_used=strategy,
            reasoning="Expansión heurística basada en patrones de pregunta"
        )

    def expand_batch(
        self,
        queries: List[str],
        strategy: Optional[ExpansionStrategy] = None
    ) -> List[ExpandedQuery]:
        """Expande múltiples queries."""
        return [self.expand(q, strategy) for q in queries]


class MultiQueryRetriever:
    """
    Wrapper que usa MultiQueryExpander para mejorar retrieval.

    Ejecuta búsqueda con query original + variaciones, fusiona y deduplica.
    """

    def __init__(
        self,
        base_retriever,  # UnifiedRetriever o similar
        expander: Optional[MultiQueryExpander] = None,
        fusion_method: str = "rrf",  # "rrf" o "max_score"
        dedupe_threshold: float = 0.95
    ):
        """
        Args:
            base_retriever: Retriever base a usar
            expander: MultiQueryExpander (se crea uno si None)
            fusion_method: Método para fusionar resultados
            dedupe_threshold: Umbral de similitud para deduplicar
        """
        self.retriever = base_retriever
        self.expander = expander or MultiQueryExpander()
        self.fusion_method = fusion_method
        self.dedupe_threshold = dedupe_threshold

    def search(
        self,
        query: str,
        top_k: int = 10,
        expand: bool = True,
        **kwargs
    ) -> Tuple[List[Any], ExpandedQuery]:
        """
        Busca usando múltiples variaciones de la query.

        Args:
            query: Query del usuario
            top_k: Resultados finales a retornar
            expand: Si expandir la query (False = búsqueda normal)
            **kwargs: Parámetros adicionales para el retriever base

        Returns:
            Tuple (resultados fusionados, ExpandedQuery usada)
        """
        if not expand:
            results = self.retriever.search(query, top_k=top_k, **kwargs)
            expanded = ExpandedQuery(
                original=query,
                variations=[],
                strategy_used=ExpansionStrategy.ALL
            )
            return results, expanded

        # Expandir query
        expanded = self.expander.expand(query)
        all_queries = expanded.all_queries()

        logger.info(f"Multi-Query: buscando con {len(all_queries)} queries")

        # Buscar con cada query
        all_results = []
        for q in all_queries:
            results = self.retriever.search(
                q,
                top_k=top_k * 2,  # Más resultados por query
                **kwargs
            )
            all_results.append((q, results))

        # Fusionar resultados
        fused = self._fuse_results(all_results, top_k)

        return fused, expanded

    def _fuse_results(
        self,
        query_results: List[Tuple[str, List[Any]]],
        top_k: int
    ) -> List[Any]:
        """
        Fusiona resultados de múltiples queries.

        Args:
            query_results: Lista de (query, resultados)
            top_k: Número de resultados finales

        Returns:
            Resultados fusionados y deduplicados
        """
        # Agrupar por chunk_id
        chunk_scores: Dict[str, Dict[str, Any]] = {}

        for query, results in query_results:
            for rank, result in enumerate(results):
                chunk_id = result.chunk_id

                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = {
                        "result": result,
                        "scores": [],
                        "ranks": [],
                        "queries": []
                    }

                chunk_scores[chunk_id]["scores"].append(result.score)
                chunk_scores[chunk_id]["ranks"].append(rank + 1)
                chunk_scores[chunk_id]["queries"].append(query)

        # Calcular score final según método
        final_scores = []

        for chunk_id, data in chunk_scores.items():
            if self.fusion_method == "rrf":
                # Reciprocal Rank Fusion
                rrf_score = sum(1.0 / (60 + r) for r in data["ranks"])
                final_score = rrf_score
            else:  # max_score
                final_score = max(data["scores"])

            # Bonus por aparecer en múltiples queries
            query_bonus = 1 + 0.1 * (len(set(data["queries"])) - 1)
            final_score *= query_bonus

            final_scores.append((final_score, data["result"]))

        # Ordenar y retornar top_k
        final_scores.sort(key=lambda x: -x[0])

        # Actualizar scores en resultados
        results = []
        for score, result in final_scores[:top_k]:
            # Crear copia con score actualizado
            result_copy = type(result)(
                chunk_id=result.chunk_id,
                content=result.content,
                score=score,
                doc_id=result.doc_id,
                doc_title=result.doc_title,
                header_path=result.header_path,
                sources=getattr(result, 'sources', []),
                source_scores=getattr(result, 'source_scores', {}),
                parent_id=getattr(result, 'parent_id', None),
                level=getattr(result, 'level', 'MICRO'),
                token_count=getattr(result, 'token_count', 0),
                metadata=getattr(result, 'metadata', {})
            )
            results.append(result_copy)

        return results
