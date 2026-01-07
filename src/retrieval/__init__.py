"""
Módulo de Retrieval - Recuperación de información.

Exporta los retrievers y la fusión híbrida.
"""

from .vector_retriever import VectorRetriever
from .bm25_retriever import BM25Retriever
from .graph_retriever import GraphRetriever
from .fusion import HybridFusion, RetrievalResult, UnifiedRetriever
from .reranker import CrossEncoderReranker, RerankerConfig, RerankerFactory
from .cache import EmbeddingCache
from .hyde import HyDEExpander, HyDEConfig, get_hyde_expander
from .semantic_cache import SemanticCache, SemanticCacheConfig, CachedResponse

__all__ = [
    "VectorRetriever",
    "BM25Retriever",
    "GraphRetriever",
    "HybridFusion",
    "RetrievalResult",
    "UnifiedRetriever",
    "CrossEncoderReranker",
    "RerankerConfig",
    "RerankerFactory",
    "EmbeddingCache",
    "HyDEExpander",
    "HyDEConfig",
    "get_hyde_expander",
    "SemanticCache",
    "SemanticCacheConfig",
    "CachedResponse"
]
