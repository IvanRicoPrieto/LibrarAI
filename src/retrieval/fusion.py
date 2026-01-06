"""
Hybrid Fusion - Combina resultados de múltiples retrievers.

Implementa:
- Reciprocal Rank Fusion (RRF)
- Score normalization
- Deduplicación inteligente
"""

from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from .vector_retriever import VectorSearchResult
from .bm25_retriever import BM25SearchResult
from .graph_retriever import GraphSearchResult

logger = logging.getLogger(__name__)


class RetrieverType(Enum):
    """Tipos de retrievers disponibles."""
    VECTOR = "vector"
    BM25 = "bm25"
    GRAPH = "graph"


@dataclass
class RetrievalResult:
    """Resultado unificado de retrieval."""
    chunk_id: str
    content: str
    score: float
    doc_id: str
    doc_title: str
    header_path: str
    sources: List[RetrieverType]  # Retrievers que encontraron este chunk
    source_scores: Dict[str, float]  # Score por cada retriever
    parent_id: Optional[str] = None
    level: str = "MICRO"
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "header_path": self.header_path,
            "sources": [s.value for s in self.sources],
            "source_scores": self.source_scores,
            "parent_id": self.parent_id,
            "level": self.level,
            "token_count": self.token_count,
            "metadata": self.metadata
        }


class HybridFusion:
    """
    Fusiona resultados de múltiples retrievers usando RRF.
    
    Reciprocal Rank Fusion (RRF):
    score(d) = Σ 1 / (k + rank_i(d))
    
    Donde k es un parámetro (típicamente 60) y rank_i(d) es
    el ranking del documento d en el retriever i.
    """
    
    def __init__(
        self,
        k: int = 60,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.3,
        graph_weight: float = 0.2
    ):
        """
        Args:
            k: Parámetro RRF (mayor k = menos peso a posiciones altas)
            vector_weight: Peso del retriever vectorial
            bm25_weight: Peso del retriever BM25
            graph_weight: Peso del retriever de grafo
        """
        self.k = k
        self.weights = {
            RetrieverType.VECTOR: vector_weight,
            RetrieverType.BM25: bm25_weight,
            RetrieverType.GRAPH: graph_weight
        }
    
    def fuse(
        self,
        vector_results: Optional[List[VectorSearchResult]] = None,
        bm25_results: Optional[List[BM25SearchResult]] = None,
        graph_results: Optional[List[GraphSearchResult]] = None,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Fusiona resultados de múltiples retrievers.
        
        Args:
            vector_results: Resultados de búsqueda vectorial
            bm25_results: Resultados de búsqueda BM25
            graph_results: Resultados de búsqueda en grafo
            top_k: Número de resultados finales
            
        Returns:
            Lista unificada ordenada por score fusionado
        """
        # Acumular scores RRF por chunk_id
        chunk_scores: Dict[str, Dict] = {}
        
        # Procesar resultados vectoriales
        if vector_results:
            for rank, result in enumerate(vector_results, 1):
                self._add_rrf_score(
                    chunk_scores,
                    result.chunk_id,
                    rank,
                    RetrieverType.VECTOR,
                    result
                )
        
        # Procesar resultados BM25
        if bm25_results:
            for rank, result in enumerate(bm25_results, 1):
                self._add_rrf_score(
                    chunk_scores,
                    result.chunk_id,
                    rank,
                    RetrieverType.BM25,
                    result
                )
        
        # Procesar resultados del grafo (expandir chunk_ids)
        if graph_results:
            for rank, result in enumerate(graph_results, 1):
                # El grafo retorna entidades con múltiples chunk_ids
                for chunk_id in result.chunk_ids[:3]:  # Limitar por entidad
                    self._add_rrf_score_from_graph(
                        chunk_scores,
                        chunk_id,
                        rank,
                        result
                    )
        
        # Calcular score final y construir resultados
        final_results = []
        
        for chunk_id, data in chunk_scores.items():
            total_score = sum(
                score * self.weights[source]
                for source, score in data["rrf_scores"].items()
            )
            
            result = RetrievalResult(
                chunk_id=chunk_id,
                content=data.get("content", ""),
                score=total_score,
                doc_id=data.get("doc_id", ""),
                doc_title=data.get("doc_title", ""),
                header_path=data.get("header_path", ""),
                sources=list(data["rrf_scores"].keys()),
                source_scores={
                    s.value: data["original_scores"].get(s, 0)
                    for s in data["rrf_scores"].keys()
                },
                parent_id=data.get("parent_id"),
                level=data.get("level", "MICRO"),
                token_count=data.get("token_count", 0),
                metadata=data.get("metadata", {})
            )
            final_results.append(result)
        
        # Ordenar por score
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(
            f"Fusión híbrida: {len(final_results)} resultados únicos "
            f"(V:{len(vector_results or [])}, B:{len(bm25_results or [])}, "
            f"G:{len(graph_results or [])})"
        )
        
        return final_results[:top_k]
    
    def _add_rrf_score(
        self,
        chunk_scores: Dict,
        chunk_id: str,
        rank: int,
        source: RetrieverType,
        result: Union[VectorSearchResult, BM25SearchResult]
    ):
        """Añade score RRF de un resultado."""
        rrf_score = 1.0 / (self.k + rank)
        
        if chunk_id not in chunk_scores:
            chunk_scores[chunk_id] = {
                "content": result.content,
                "doc_id": result.doc_id,
                "doc_title": result.doc_title,
                "header_path": result.header_path,
                "parent_id": getattr(result, "parent_id", None),
                "level": result.level,
                "token_count": result.token_count,
                "metadata": getattr(result, "metadata", {}),
                "rrf_scores": {},
                "original_scores": {}
            }
        
        chunk_scores[chunk_id]["rrf_scores"][source] = rrf_score
        chunk_scores[chunk_id]["original_scores"][source] = result.score
    
    def _add_rrf_score_from_graph(
        self,
        chunk_scores: Dict,
        chunk_id: str,
        rank: int,
        result: GraphSearchResult
    ):
        """Añade score RRF desde resultado de grafo."""
        rrf_score = 1.0 / (self.k + rank)
        
        if chunk_id not in chunk_scores:
            # Solo tenemos chunk_id, el contenido se debe buscar después
            chunk_scores[chunk_id] = {
                "content": "",  # Se rellenará después
                "doc_id": "",
                "doc_title": "",
                "header_path": "",
                "parent_id": None,
                "level": "MICRO",
                "token_count": 0,
                "metadata": {
                    "from_entity": result.entity,
                    "entity_type": result.entity_type
                },
                "rrf_scores": {},
                "original_scores": {}
            }
        
        source = RetrieverType.GRAPH
        chunk_scores[chunk_id]["rrf_scores"][source] = rrf_score
        chunk_scores[chunk_id]["original_scores"][source] = result.score
    
    def rerank_with_cross_encoder(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 10,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> List[RetrievalResult]:
        """
        Re-rankea resultados usando un cross-encoder.
        
        El cross-encoder evalúa query-documento de forma conjunta,
        lo que típicamente da mejores resultados que bi-encoders.
        
        Args:
            query: Consulta original
            results: Resultados a re-rankear
            top_k: Número de resultados finales
            model_name: Modelo de cross-encoder
            
        Returns:
            Resultados re-rankeados
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.warning(
                "sentence-transformers no disponible, "
                "omitiendo re-ranking"
            )
            return results[:top_k]
        
        if not results:
            return []
        
        # Cargar modelo
        model = CrossEncoder(model_name)
        
        # Preparar pares query-documento
        pairs = [(query, r.content) for r in results]
        
        # Obtener scores
        scores = model.predict(pairs)
        
        # Asignar scores y re-ordenar
        for result, score in zip(results, scores):
            result.metadata["cross_encoder_score"] = float(score)
            # Combinar con score original
            result.score = 0.7 * float(score) + 0.3 * result.score
        
        results.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Re-ranking con cross-encoder: top score = {results[0].score:.3f}")
        
        return results[:top_k]


class UnifiedRetriever:
    """
    Retriever unificado que coordina todos los métodos de búsqueda.
    
    Proporciona una interfaz simple para:
    - Búsqueda híbrida (vector + BM25 + grafo)
    - Configuración de pesos
    - Re-ranking opcional
    """
    
    def __init__(
        self,
        indices_dir,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.3,
        graph_weight: float = 0.2,
        use_graph: bool = True,
        use_reranker: bool = False
    ):
        """
        Args:
            indices_dir: Directorio con los índices
            vector_weight: Peso búsqueda vectorial
            bm25_weight: Peso búsqueda BM25
            graph_weight: Peso búsqueda en grafo
            use_graph: Si usar retriever de grafo
            use_reranker: Si aplicar cross-encoder
        """
        from pathlib import Path
        from .vector_retriever import VectorRetriever
        from .bm25_retriever import BM25Retriever
        from .graph_retriever import GraphRetriever
        
        self.indices_dir = Path(indices_dir)
        self.use_graph = use_graph
        self.use_reranker = use_reranker
        
        # Inicializar retrievers
        self.vector_retriever = VectorRetriever(self.indices_dir)
        self.bm25_retriever = BM25Retriever(self.indices_dir)
        self.graph_retriever = GraphRetriever(self.indices_dir) if use_graph else None
        
        # Fusión
        self.fusion = HybridFusion(
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            graph_weight=graph_weight
        )
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        vector_top_k: int = 20,
        bm25_top_k: int = 20,
        graph_top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        Realiza búsqueda híbrida.
        
        Args:
            query: Consulta en lenguaje natural
            top_k: Número de resultados finales
            vector_top_k: Resultados de búsqueda vectorial
            bm25_top_k: Resultados de BM25
            graph_top_k: Resultados de grafo
            filters: Filtros opcionales
            
        Returns:
            Resultados fusionados y ordenados
        """
        # Búsqueda vectorial
        vector_results = self.vector_retriever.search(
            query,
            top_k=vector_top_k,
            filters=filters
        )
        
        # Búsqueda BM25
        bm25_results = self.bm25_retriever.search(
            query,
            top_k=bm25_top_k
        )
        
        # Búsqueda en grafo
        graph_results = None
        if self.use_graph and self.graph_retriever:
            graph_results = self.graph_retriever.search(
                query,
                top_k=graph_top_k
            )
        
        # Fusionar resultados
        results = self.fusion.fuse(
            vector_results=vector_results,
            bm25_results=bm25_results,
            graph_results=graph_results,
            top_k=top_k * 2 if self.use_reranker else top_k
        )
        
        # Re-ranking opcional
        if self.use_reranker and results:
            results = self.fusion.rerank_with_cross_encoder(
                query,
                results,
                top_k=top_k
            )
        
        return results
