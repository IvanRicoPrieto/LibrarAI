"""
BM25 Retriever - Búsqueda léxica basada en términos.

Complementa la búsqueda vectorial para:
- Encontrar términos exactos (nombres, fórmulas, siglas)
- Mejorar recall para queries específicas
- Detección de keywords técnicos
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import logging
import pickle

logger = logging.getLogger(__name__)


@dataclass
class BM25SearchResult:
    """Resultado de búsqueda BM25."""
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


class BM25Retriever:
    """
    Retriever basado en BM25 (Best Matching 25).
    
    BM25 es un algoritmo de ranking probabilístico que:
    - Considera frecuencia de términos (TF)
    - Aplica IDF (Inverse Document Frequency)
    - Normaliza por longitud del documento
    
    Ideal para:
    - Búsqueda de términos exactos
    - Nombres de algoritmos/protocolos
    - Símbolos y fórmulas
    """
    
    def __init__(self, indices_dir: Path):
        """
        Args:
            indices_dir: Directorio con los índices
        """
        self.indices_dir = Path(indices_dir)
        
        self._bm25_index = None
        self._chunks_list = None  # Lista ordenada de chunks (mismo orden que BM25)
        self._chunks_store = None  # Dict con todos los chunks
    
    def _load_index(self):
        """Carga índice BM25 y chunks."""
        if self._bm25_index is not None:
            return
        
        bm25_path = self.indices_dir / "bm25_index.pkl"
        chunks_path = self.indices_dir / "chunks.pkl"
        
        if not bm25_path.exists():
            raise FileNotFoundError(
                f"Índice BM25 no encontrado en {bm25_path}. "
                "Ejecuta primero la indexación."
            )
        
        with open(bm25_path, 'rb') as f:
            self._bm25_index = pickle.load(f)
        
        with open(chunks_path, 'rb') as f:
            self._chunks_store = pickle.load(f)
        
        # Filtrar solo MICRO chunks y mantener orden
        from ..ingestion.chunker import ChunkLevel
        self._chunks_list = [
            c for c in self._chunks_store.values()
            if c.level == ChunkLevel.MICRO
        ]
        
        logger.debug(f"BM25 cargado: {len(self._chunks_list)} chunks")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0
    ) -> List[BM25SearchResult]:
        """
        Realiza búsqueda léxica con BM25.
        
        Args:
            query: Consulta en lenguaje natural
            top_k: Número de resultados a retornar
            score_threshold: Umbral mínimo de score BM25
            
        Returns:
            Lista de resultados ordenados por relevancia
        """
        self._load_index()
        
        # Tokenizar query (misma lógica que el indexer)
        from ..utils.text_processing import tokenize_for_bm25
        query_tokens = tokenize_for_bm25(query)
        
        # Obtener scores BM25
        scores = self._bm25_index.get_scores(query_tokens)
        
        # Ordenar por score
        scored_chunks = list(zip(self._chunks_list, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Filtrar y convertir
        results = []
        for chunk, score in scored_chunks[:top_k]:
            if score < score_threshold:
                continue
            
            result = BM25SearchResult(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                score=score,
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                header_path=chunk.header_path,
                parent_id=chunk.parent_id,
                level=chunk.level.value,
                token_count=chunk.token_count
            )
            results.append(result)
        
        logger.info(f"BM25 search: {len(results)} resultados para '{query[:50]}...'")
        return results
    
    def search_exact_terms(
        self,
        terms: List[str],
        top_k: int = 10
    ) -> List[BM25SearchResult]:
        """
        Búsqueda enfocada en términos exactos.
        
        Útil para nombres específicos, siglas, etc.
        
        Args:
            terms: Lista de términos a buscar
            top_k: Número de resultados
            
        Returns:
            Resultados que contienen los términos
        """
        self._load_index()
        
        # Buscar chunks que contengan los términos
        results = []
        
        for chunk in self._chunks_list:
            content_lower = chunk.content.lower()
            
            # Contar matches de términos (word boundary para evitar substrings)
            match_count = sum(
                1 for term in terms
                if re.search(r'\b' + re.escape(term.lower()) + r'\b', content_lower)
            )
            
            if match_count > 0:
                # Score proporcional a matches
                score = match_count / len(terms)
                
                result = BM25SearchResult(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    score=score,
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    header_path=chunk.header_path,
                    parent_id=chunk.parent_id,
                    level=chunk.level.value,
                    token_count=chunk.token_count
                )
                results.append(result)
        
        # Ordenar y limitar
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def get_documents_with_term(self, term: str) -> List[str]:
        """
        Obtiene IDs de documentos que contienen un término.
        
        Args:
            term: Término a buscar
            
        Returns:
            Lista de doc_ids únicos
        """
        self._load_index()
        
        doc_ids = set()
        term_lower = term.lower()
        
        for chunk in self._chunks_list:
            if re.search(r'\b' + re.escape(term_lower) + r'\b', chunk.content.lower()):
                doc_ids.add(chunk.doc_id)
        
        return list(doc_ids)
