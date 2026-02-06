"""
Query Router - Enruta consultas a la estrategia óptima.

Decide:
- Qué retrievers usar (vector, BM25, grafo)
- Pesos de fusión
- Si requiere sub-queries
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Estrategias de retrieval disponibles."""
    VECTOR_ONLY = "vector_only"
    BM25_ONLY = "bm25_only"
    HYBRID = "hybrid"
    HYBRID_WITH_GRAPH = "hybrid_with_graph"
    MULTI_HOP = "multi_hop"


@dataclass
class RoutingDecision:
    """Decisión de routing."""
    strategy: RetrievalStrategy
    vector_weight: float
    bm25_weight: float
    graph_weight: float
    top_k: int
    sub_queries: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "strategy": self.strategy.value,
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "graph_weight": self.graph_weight,
            "top_k": self.top_k,
            "sub_queries": self.sub_queries,
            "filters": self.filters,
            "reasoning": self.reasoning
        }


class QueryRouter:
    """
    Router inteligente que decide cómo procesar cada consulta.
    
    Puede usar:
    - Heurísticas basadas en patrones
    - LLM para decisiones complejas
    """
    
    # Patrones para detección de tipo de query
    PATTERNS = {
        "exact_term": [
            r"\b(?:BB84|E91|B92|SARG04|COW|DPS)\b",
            r"\b(?:Shor|Grover|Deutsch|Simon)\b",
            r"\b(?:CNOT|Hadamard|Pauli|Toffoli)\b"
        ],
        "conceptual": [
            r"(?:qué es|what is|explicar?|definición)",
            r"(?:concepto de|meaning of)",
            r"(?:introducción|overview)"
        ],
        "relational": [
            r"(?:relación entre|relationship between)",
            r"(?:conecta con|relates to)",
            r"(?:depende de|depends on)"
        ],
        "comparative": [
            r"(?:diferencia entre|difference between)",
            r"(?:comparar?|versus|vs\.?)",
            r"(?:mejor|advantages|desventajas)"
        ],
        "multi_hop": [
            r"(?:y además|and also|también)",
            r"(?:primero.*luego|first.*then)",
            r"(?:por qué.*cómo|why.*how)"
        ]
    }
    
    def __init__(
        self,
        use_llm_router: bool = False,
        default_top_k: int = 10
    ):
        """
        Args:
            use_llm_router: Si usar LLM para routing
            default_top_k: top_k por defecto
        """
        self.use_llm_router = use_llm_router
        self.default_top_k = default_top_k
    
    def route(self, query: str) -> RoutingDecision:
        """
        Decide la estrategia de retrieval para una query.
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Decisión de routing
        """
        if self.use_llm_router:
            return self._route_with_llm(query)
        else:
            return self._route_with_heuristics(query)
    
    def _route_with_heuristics(self, query: str) -> RoutingDecision:
        """Routing basado en heurísticas."""
        query_lower = query.lower()
        
        # Detectar patrones
        has_exact_term = any(
            re.search(p, query, re.IGNORECASE) 
            for p in self.PATTERNS["exact_term"]
        )
        
        has_relational = any(
            re.search(p, query_lower) 
            for p in self.PATTERNS["relational"]
        )
        
        has_multi_hop = any(
            re.search(p, query_lower) 
            for p in self.PATTERNS["multi_hop"]
        )
        
        has_comparative = any(
            re.search(p, query_lower) 
            for p in self.PATTERNS["comparative"]
        )
        
        # Decidir estrategia
        if has_multi_hop:
            # Query compleja: dividir en sub-queries
            sub_queries = self._extract_sub_queries(query)
            return RoutingDecision(
                strategy=RetrievalStrategy.MULTI_HOP,
                vector_weight=0.4,
                bm25_weight=0.3,
                graph_weight=0.3,
                top_k=self.default_top_k * 2,
                sub_queries=sub_queries,
                reasoning="Query multi-hop detectada, dividiendo en sub-consultas"
            )
        
        if has_relational:
            # Usar grafo para relaciones
            return RoutingDecision(
                strategy=RetrievalStrategy.HYBRID_WITH_GRAPH,
                vector_weight=0.3,
                bm25_weight=0.2,
                graph_weight=0.5,
                top_k=self.default_top_k,
                reasoning="Query relacional: priorizando grafo de conocimiento"
            )
        
        if has_exact_term:
            # Priorizar BM25 para términos exactos
            return RoutingDecision(
                strategy=RetrievalStrategy.HYBRID,
                vector_weight=0.3,
                bm25_weight=0.6,
                graph_weight=0.1,
                top_k=self.default_top_k,
                reasoning="Términos exactos detectados: priorizando BM25"
            )
        
        if has_comparative:
            # Comparative: necesita múltiples perspectivas
            return RoutingDecision(
                strategy=RetrievalStrategy.HYBRID_WITH_GRAPH,
                vector_weight=0.4,
                bm25_weight=0.3,
                graph_weight=0.3,
                top_k=self.default_top_k + 5,
                reasoning="Comparación: búsqueda amplia con múltiples fuentes"
            )
        
        # Default: híbrido balanceado
        return RoutingDecision(
            strategy=RetrievalStrategy.HYBRID,
            vector_weight=0.5,
            bm25_weight=0.3,
            graph_weight=0.2,
            top_k=self.default_top_k,
            reasoning="Query general: estrategia híbrida balanceada"
        )
    
    def _extract_sub_queries(self, query: str) -> List[str]:
        """Extrae sub-queries de una query compleja."""
        sub_queries = []
        
        # Dividir por conectores
        connectors = [
            r'\s+y\s+(?:además|también)\s+',
            r'\s+and\s+(?:also|additionally)\s+',
            r'\s*[,;]\s*',
            r'\s+así como\s+',
            r'\s+as well as\s+'
        ]
        
        parts = [query]
        for connector in connectors:
            new_parts = []
            for part in parts:
                new_parts.extend(re.split(connector, part, flags=re.IGNORECASE))
            parts = new_parts
        
        # Limpiar y filtrar
        for part in parts:
            part = part.strip()
            if len(part) > 10:  # Mínimo viable
                sub_queries.append(part)
        
        return sub_queries if len(sub_queries) > 1 else [query]
    
    def _route_with_llm(self, query: str) -> RoutingDecision:
        """Routing usando LLM."""
        from src.llm_provider import complete as llm_complete

        prompt = f"""Analiza esta consulta y decide la mejor estrategia de búsqueda.

CONSULTA: {query}

OPCIONES:
1. VECTOR_ONLY: Para queries conceptuales generales
2. BM25_ONLY: Para búsqueda de términos exactos
3. HYBRID: Combinación vector + BM25
4. HYBRID_WITH_GRAPH: Incluye grafo de conocimiento para relaciones
5. MULTI_HOP: Query compleja que requiere dividirse

Responde en JSON:
{{
    "strategy": "HYBRID",
    "vector_weight": 0.5,
    "bm25_weight": 0.3,
    "graph_weight": 0.2,
    "top_k": 10,
    "sub_queries": [],
    "reasoning": "Explicación breve"
}}"""

        try:
            response = llm_complete(prompt=prompt, json_mode=True, temperature=0.1)

            import json
            result = json.loads(response.content)

            return RoutingDecision(
                strategy=RetrievalStrategy(result.get("strategy", "HYBRID").lower()),
                vector_weight=result.get("vector_weight", 0.5),
                bm25_weight=result.get("bm25_weight", 0.3),
                graph_weight=result.get("graph_weight", 0.2),
                top_k=result.get("top_k", self.default_top_k),
                sub_queries=result.get("sub_queries", []),
                reasoning=result.get("reasoning", "")
            )
        except Exception as e:
            logger.warning(f"LLM routing failed: {e}, using heuristics")
            return self._route_with_heuristics(query)
