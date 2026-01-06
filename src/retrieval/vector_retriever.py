"""
Vector Retriever - Búsqueda semántica usando Qdrant.

Características:
- Búsqueda por similitud coseno
- Soporte para filtros de metadatos
- Auto-merge de chunks jerárquicos
- Cache de embeddings para reducir costes 70-90%
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import logging
import pickle

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Resultado de búsqueda vectorial."""
    chunk_id: str
    content: str
    score: float
    doc_id: str
    doc_title: str
    header_path: str
    parent_id: Optional[str] = None
    level: str = "MICRO"
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorRetriever:
    """
    Retriever basado en búsqueda vectorial con Qdrant.
    
    Soporta:
    - Búsqueda semántica por similitud
    - Filtrado por documento/sección
    - Auto-merge de chunks
    - Cache de embeddings (reduce costes 70-90%)
    """
    
    def __init__(
        self,
        indices_dir: Path,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-large",
        collection_name: str = "quantum_library",
        use_cache: bool = True
    ):
        """
        Args:
            indices_dir: Directorio con los índices
            embedding_provider: "openai" o "local"
            embedding_model: Modelo de embeddings
            collection_name: Nombre de colección en Qdrant
            use_cache: Habilitar cache de embeddings
        """
        self.indices_dir = Path(indices_dir)
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.use_cache = use_cache
        
        self._qdrant_client = None
        self._embedding_client = None
        self._chunks_store = None
        self._embedding_cache = None
    
    def _init_qdrant(self):
        """Inicializa cliente de Qdrant."""
        if self._qdrant_client is not None:
            return
        
        from qdrant_client import QdrantClient
        
        qdrant_path = self.indices_dir / "qdrant"
        self._qdrant_client = QdrantClient(path=str(qdrant_path))
        logger.debug("Qdrant inicializado")
    
    def _init_embeddings(self):
        """Inicializa cliente de embeddings."""
        if self._embedding_client is not None:
            return
        
        if self.embedding_provider == "openai":
            from openai import OpenAI
            import os
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY no configurada")
            
            self._embedding_client = OpenAI(api_key=api_key)
        else:
            from sentence_transformers import SentenceTransformer
            self._embedding_client = SentenceTransformer(self.embedding_model)
    
    def _init_cache(self):
        """Inicializa cache de embeddings."""
        if self._embedding_cache is not None or not self.use_cache:
            return
        
        try:
            from .cache import EmbeddingCache, CacheConfig
            
            config = CacheConfig(
                memory_cache_size=10000,
                persistent=True,
                cache_file="embedding_cache.db",
            )
            self._embedding_cache = EmbeddingCache(
                indices_dir=self.indices_dir,
                config=config,
            )
            logger.debug("Embedding cache inicializado")
        except Exception as e:
            logger.warning(f"No se pudo inicializar cache de embeddings: {e}")
            self.use_cache = False
    
    def _get_embedding(self, text: str) -> List[float]:
        """Genera embedding para una query, usando cache si está disponible."""
        self._init_embeddings()
        
        # Check cache first
        if self.use_cache:
            self._init_cache()
            if self._embedding_cache:
                cached = self._embedding_cache.get(text, model=self.embedding_model)
                if cached is not None:
                    logger.debug(f"Cache hit para query: {text[:50]}...")
                    return cached
        
        # Compute embedding
        if self.embedding_provider == "openai":
            response = self._embedding_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embedding = response.data[0].embedding
        else:
            embedding = self._embedding_client.encode(text).tolist()
        
        # Store in cache
        if self.use_cache and self._embedding_cache:
            self._embedding_cache.set(text, embedding, model=self.embedding_model)
            logger.debug(f"Cached embedding para query: {text[:50]}...")
        
        return embedding
    
    def get_cache_stats(self) -> dict:
        """Obtiene estadísticas del cache de embeddings."""
        if self._embedding_cache:
            return self._embedding_cache.get_stats()
        return {"enabled": False}
    
    def _load_chunks_store(self):
        """Carga el almacén de chunks para auto-merge."""
        if self._chunks_store is not None:
            return
        
        chunks_path = self.indices_dir / "chunks.pkl"
        if chunks_path.exists():
            with open(chunks_path, 'rb') as f:
                self._chunks_store = pickle.load(f)
        else:
            self._chunks_store = {}
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
        auto_merge: bool = True
    ) -> List[VectorSearchResult]:
        """
        Realiza búsqueda semántica.
        
        Args:
            query: Consulta en lenguaje natural
            top_k: Número de resultados a retornar
            score_threshold: Umbral mínimo de similitud
            filters: Filtros de metadatos (doc_id, header_path, etc.)
            auto_merge: Si expandir chunks usando jerarquía
            
        Returns:
            Lista de resultados ordenados por relevancia
        """
        self._init_qdrant()
        
        # Generar embedding de la query
        query_embedding = self._get_embedding(query)
        
        # Construir filtros para Qdrant
        qdrant_filter = None
        if filters:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            qdrant_filter = Filter(must=conditions)
        
        # Búsqueda usando query_points (qdrant-client 1.16+)
        from qdrant_client.models import QueryResponse
        
        response = self._qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True
        )
        
        # Convertir a resultados
        search_results = []
        for hit in response.points:
            payload = hit.payload
            result = VectorSearchResult(
                chunk_id=payload.get("chunk_id", ""),
                content=payload.get("content", ""),
                score=hit.score,
                doc_id=payload.get("doc_id", ""),
                doc_title=payload.get("doc_title", ""),
                header_path=payload.get("header_path", ""),
                parent_id=payload.get("parent_id"),
                level=payload.get("level", "MICRO"),
                token_count=payload.get("token_count", 0),
                metadata=payload
            )
            search_results.append(result)
        
        # Auto-merge: expandir contexto si es necesario
        if auto_merge and search_results:
            search_results = self._auto_merge_results(search_results)
        
        logger.info(f"Vector search: {len(search_results)} resultados para '{query[:50]}...'")
        return search_results
    
    def _auto_merge_results(
        self, 
        results: List[VectorSearchResult]
    ) -> List[VectorSearchResult]:
        """
        Expande chunks MICRO a MESO/MACRO si hay suficientes hermanos.
        
        Usa el algoritmo de auto-merge de la propuesta:
        - Si ≥3 hermanos MICRO del mismo padre → usa chunk MESO
        - Si ≥2 hermanos MESO del mismo padre → usa chunk MACRO
        """
        self._load_chunks_store()
        
        if not self._chunks_store:
            return results
        
        from collections import defaultdict
        
        # Agrupar por parent_id
        parent_groups = defaultdict(list)
        for result in results:
            if result.parent_id:
                parent_groups[result.parent_id].append(result)
        
        merged_results = []
        merged_chunk_ids = set()
        
        for parent_id, siblings in parent_groups.items():
            # Si hay ≥3 hermanos, buscar el padre
            if len(siblings) >= 3 and parent_id in self._chunks_store:
                parent_chunk = self._chunks_store[parent_id]
                
                # Crear resultado con el padre
                merged_result = VectorSearchResult(
                    chunk_id=parent_chunk.chunk_id,
                    content=parent_chunk.content,
                    score=max(s.score for s in siblings),  # Mejor score
                    doc_id=parent_chunk.doc_id,
                    doc_title=parent_chunk.doc_title,
                    header_path=parent_chunk.header_path,
                    parent_id=parent_chunk.parent_id,
                    level=parent_chunk.level.value,
                    token_count=parent_chunk.token_count,
                    metadata={"merged_from": [s.chunk_id for s in siblings]}
                )
                merged_results.append(merged_result)
                
                # Marcar hijos como ya procesados
                for sibling in siblings:
                    merged_chunk_ids.add(sibling.chunk_id)
        
        # Agregar resultados no merged
        for result in results:
            if result.chunk_id not in merged_chunk_ids:
                merged_results.append(result)
        
        # Re-ordenar por score
        merged_results.sort(key=lambda x: x.score, reverse=True)
        
        return merged_results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[VectorSearchResult]:
        """Obtiene un chunk específico por ID."""
        self._init_qdrant()
        
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        results = self._qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="chunk_id",
                        match=MatchValue(value=chunk_id)
                    )
                ]
            ),
            limit=1
        )[0]
        
        if results:
            payload = results[0].payload
            return VectorSearchResult(
                chunk_id=payload.get("chunk_id", ""),
                content=payload.get("content", ""),
                score=1.0,
                doc_id=payload.get("doc_id", ""),
                doc_title=payload.get("doc_title", ""),
                header_path=payload.get("header_path", ""),
                parent_id=payload.get("parent_id"),
                level=payload.get("level", "MICRO"),
                token_count=payload.get("token_count", 0)
            )
        return None
