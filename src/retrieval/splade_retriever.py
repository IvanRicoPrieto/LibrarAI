"""
SPLADE Retriever - Learned Sparse Representations.

SPLADE (Sparse Lexical and Expansion) aprende representaciones sparse
que combinan las ventajas de:
- Búsqueda léxica (BM25): matching exacto de términos
- Búsqueda semántica: expansión de términos relacionados

Mejor que BM25 puro, complementa bien embeddings densos.

Requiere:
    pip install transformers torch
    # Modelo: naver/splade-cocondenser-ensembledistil
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import pickle
import json
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


@dataclass
class SPLADEResult:
    """Resultado de búsqueda SPLADE."""
    chunk_id: str
    content: str
    score: float
    doc_id: str = ""
    doc_title: str = ""
    header_path: str = ""
    matched_terms: List[str] = field(default_factory=list)
    expanded_terms: List[str] = field(default_factory=list)

    def to_retrieval_result(self, metadata: Dict = None):
        """Convierte a RetrievalResult estándar."""
        from .fusion import RetrievalResult, RetrieverType

        return RetrievalResult(
            chunk_id=self.chunk_id,
            content=self.content,
            score=self.score,
            doc_id=self.doc_id,
            doc_title=self.doc_title,
            header_path=self.header_path,
            sources=[RetrieverType.SPLADE],
            source_scores={"splade": self.score},
            metadata={
                **(metadata or {}),
                "matched_terms": self.matched_terms,
                "expanded_terms": self.expanded_terms
            }
        )


class SPLADERetriever:
    """
    Retriever basado en SPLADE para representaciones sparse aprendidas.

    Características:
    - Expande queries y documentos con términos semánticamente relacionados
    - Matching sparse eficiente (inverted index)
    - Mejor que BM25 en recall, comparable en precisión
    - Interpretable: puedes ver qué términos matchean

    Uso:
        retriever = SPLADERetriever(indices_dir)
        retriever.index(chunks)
        results = retriever.search("entrelazamiento cuántico", top_k=10)
    """

    def __init__(
        self,
        indices_dir: Path,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        max_length: int = 256,
        top_k_tokens: int = 256,  # Tokens sparse por documento
        device: str = "cpu"
    ):
        """
        Args:
            indices_dir: Directorio para índices
            model_name: Modelo SPLADE de HuggingFace
            max_length: Longitud máxima de tokens
            top_k_tokens: Número de tokens sparse a retener
            device: "cpu" o "cuda"
        """
        self.indices_dir = Path(indices_dir)
        self.model_name = model_name
        self.max_length = max_length
        self.top_k_tokens = top_k_tokens
        self.device = device

        self._index_path = self.indices_dir / "splade"
        self._inverted_index_path = self._index_path / "inverted_index.pkl"
        self._doc_vectors_path = self._index_path / "doc_vectors.pkl"
        self._chunks_meta_path = self._index_path / "chunks_meta.pkl"

        self._model = None
        self._tokenizer = None
        self._inverted_index: Dict[str, List[Tuple[str, float]]] = {}  # term -> [(chunk_id, weight)]
        self._doc_vectors: Dict[str, Dict[str, float]] = {}  # chunk_id -> {term: weight}
        self._chunks_meta: Dict[str, Dict] = {}  # chunk_id -> metadata

        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Verifica disponibilidad de dependencias."""
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            return True
        except ImportError as e:
            logger.warning(f"SPLADE no disponible: {e}")
            return False

    def is_available(self) -> bool:
        """Retorna True si SPLADE está disponible."""
        return self._available

    def _load_model(self):
        """Carga modelo SPLADE."""
        if self._model is not None:
            return

        if not self._available:
            return

        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        logger.info(f"Cargando modelo SPLADE: {self.model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

        logger.info("Modelo SPLADE cargado")

    def _encode_sparse(self, text: str) -> Dict[str, float]:
        """
        Codifica texto a representación sparse SPLADE.

        Args:
            text: Texto a codificar

        Returns:
            Dict token -> weight
        """
        import torch

        self._load_model()

        # Tokenizar
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits  # (batch, seq_len, vocab_size)

        # SPLADE: log(1 + ReLU(logits)) * attention_mask
        # Luego max pooling sobre secuencia
        relu_log = torch.log1p(torch.relu(logits))
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        weighted = relu_log * attention_mask

        # Max pooling
        sparse_vec = weighted.max(dim=1).values.squeeze()  # (vocab_size,)

        # Convertir a dict sparse
        sparse_dict = {}
        indices = torch.nonzero(sparse_vec > 0).squeeze(-1)

        if indices.numel() > 0:
            values = sparse_vec[indices]

            # Ordenar por valor y tomar top_k
            if len(indices) > self.top_k_tokens:
                top_values, top_indices = torch.topk(values, self.top_k_tokens)
                indices = indices[top_indices]
                values = top_values

            # Convertir a tokens
            tokens = self._tokenizer.convert_ids_to_tokens(indices.tolist())
            for token, value in zip(tokens, values.tolist()):
                if token not in ['[PAD]', '[CLS]', '[SEP]', '[UNK]']:
                    sparse_dict[token] = value

        return sparse_dict

    def index(
        self,
        chunks: List[Any],
        batch_size: int = 16,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Indexa chunks con SPLADE.

        Args:
            chunks: Lista de chunks a indexar
            batch_size: Tamaño de batch
            force: Forzar reindexación

        Returns:
            Estadísticas de indexación
        """
        if not self.is_available():
            return {"error": "SPLADE no disponible"}

        if self._inverted_index_path.exists() and not force:
            self._load_index()
            return {"skipped": True, "loaded": len(self._doc_vectors)}

        logger.info(f"Indexando {len(chunks)} chunks con SPLADE...")

        self._load_model()
        self._inverted_index = defaultdict(list)
        self._doc_vectors = {}
        self._chunks_meta = {}

        total_terms = 0

        for i, chunk in enumerate(chunks):
            if i % 100 == 0:
                logger.info(f"SPLADE indexando: {i}/{len(chunks)}")

            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            chunk_id = chunk.chunk_id if hasattr(chunk, 'chunk_id') else f"chunk_{i}"

            # Codificar
            sparse_vec = self._encode_sparse(content)
            self._doc_vectors[chunk_id] = sparse_vec
            total_terms += len(sparse_vec)

            # Añadir a índice invertido
            for term, weight in sparse_vec.items():
                self._inverted_index[term].append((chunk_id, weight))

            # Metadata
            self._chunks_meta[chunk_id] = {
                "content": content,
                "doc_id": getattr(chunk, 'doc_id', ''),
                "doc_title": getattr(chunk, 'doc_title', ''),
                "header_path": getattr(chunk, 'header_path', ''),
            }

        # Guardar índice
        self._save_index()

        avg_terms = total_terms / len(chunks) if chunks else 0
        logger.info(
            f"SPLADE indexación completada: {len(chunks)} docs, "
            f"{len(self._inverted_index)} términos únicos, "
            f"{avg_terms:.1f} términos/doc promedio"
        )

        return {
            "indexed": len(chunks),
            "unique_terms": len(self._inverted_index),
            "avg_terms_per_doc": avg_terms
        }

    def _save_index(self):
        """Guarda índice a disco."""
        self._index_path.mkdir(parents=True, exist_ok=True)

        with open(self._inverted_index_path, 'wb') as f:
            pickle.dump(dict(self._inverted_index), f)

        with open(self._doc_vectors_path, 'wb') as f:
            pickle.dump(self._doc_vectors, f)

        with open(self._chunks_meta_path, 'wb') as f:
            pickle.dump(self._chunks_meta, f)

    def _load_index(self):
        """Carga índice desde disco."""
        if self._inverted_index_path.exists():
            with open(self._inverted_index_path, 'rb') as f:
                self._inverted_index = defaultdict(list, pickle.load(f))

        if self._doc_vectors_path.exists():
            with open(self._doc_vectors_path, 'rb') as f:
                self._doc_vectors = pickle.load(f)

        if self._chunks_meta_path.exists():
            with open(self._chunks_meta_path, 'rb') as f:
                self._chunks_meta = pickle.load(f)

        logger.info(
            f"Índice SPLADE cargado: {len(self._doc_vectors)} docs, "
            f"{len(self._inverted_index)} términos"
        )

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SPLADEResult]:
        """
        Busca usando SPLADE.

        Args:
            query: Query de búsqueda
            top_k: Número de resultados

        Returns:
            Lista de SPLADEResult
        """
        if not self.is_available():
            return []

        # Cargar índice si no está cargado
        if not self._inverted_index:
            self._load_index()

        if not self._inverted_index:
            logger.warning("Índice SPLADE vacío")
            return []

        self._load_model()

        # Codificar query
        query_vec = self._encode_sparse(query)

        if not query_vec:
            return []

        # Scoring con dot product sparse
        scores: Dict[str, float] = defaultdict(float)
        matched_terms: Dict[str, List[str]] = defaultdict(list)

        for term, q_weight in query_vec.items():
            if term in self._inverted_index:
                for chunk_id, d_weight in self._inverted_index[term]:
                    scores[chunk_id] += q_weight * d_weight
                    matched_terms[chunk_id].append(term)

        if not scores:
            return []

        # Ordenar y retornar top_k
        sorted_results = sorted(scores.items(), key=lambda x: -x[1])[:top_k]

        results = []
        for chunk_id, score in sorted_results:
            meta = self._chunks_meta.get(chunk_id, {})

            # Términos expandidos (en doc pero no en query original)
            doc_terms = set(self._doc_vectors.get(chunk_id, {}).keys())
            query_terms = set(query.lower().split())
            expanded = [t for t in matched_terms[chunk_id] if t.lower() not in query_terms]

            results.append(SPLADEResult(
                chunk_id=chunk_id,
                content=meta.get('content', ''),
                score=score,
                doc_id=meta.get('doc_id', ''),
                doc_title=meta.get('doc_title', ''),
                header_path=meta.get('header_path', ''),
                matched_terms=matched_terms[chunk_id],
                expanded_terms=expanded[:10]  # Top 10 expansiones
            ))

        return results

    def search_as_retrieval_results(
        self,
        query: str,
        top_k: int = 10,
        chunks_store: Dict = None
    ) -> List[Any]:
        """
        Busca y retorna como RetrievalResult estándar.

        Args:
            query: Query de búsqueda
            top_k: Número de resultados
            chunks_store: Dict chunk_id -> Chunk para metadata extra

        Returns:
            Lista de RetrievalResult
        """
        results = self.search(query, top_k)

        retrieval_results = []
        for r in results:
            metadata = {
                "matched_terms": r.matched_terms,
                "expanded_terms": r.expanded_terms
            }
            if chunks_store and r.chunk_id in chunks_store:
                chunk = chunks_store[r.chunk_id]
                metadata.update({
                    "section_hierarchy": getattr(chunk, 'section_hierarchy', []),
                    "section_number": getattr(chunk, 'section_number', ''),
                    "topic_summary": getattr(chunk, 'topic_summary', ''),
                })

            retrieval_results.append(r.to_retrieval_result(metadata))

        return retrieval_results

    def explain_match(
        self,
        query: str,
        chunk_id: str
    ) -> Dict[str, Any]:
        """
        Explica por qué un chunk matchea con una query.

        Útil para debugging e interpretabilidad.

        Args:
            query: Query original
            chunk_id: ID del chunk

        Returns:
            Dict con explicación del match
        """
        if not self._doc_vectors or chunk_id not in self._doc_vectors:
            return {"error": "Chunk no encontrado en índice"}

        self._load_model()
        query_vec = self._encode_sparse(query)
        doc_vec = self._doc_vectors[chunk_id]

        # Encontrar términos que contribuyen al score
        contributions = []
        total_score = 0

        for term in set(query_vec.keys()) & set(doc_vec.keys()):
            contrib = query_vec[term] * doc_vec[term]
            total_score += contrib
            contributions.append({
                "term": term,
                "query_weight": query_vec[term],
                "doc_weight": doc_vec[term],
                "contribution": contrib
            })

        contributions.sort(key=lambda x: -x['contribution'])

        return {
            "total_score": total_score,
            "top_contributions": contributions[:20],
            "query_terms": len(query_vec),
            "doc_terms": len(doc_vec),
            "matching_terms": len(contributions),
            "query_expansion": [
                t for t in query_vec.keys()
                if t.lower() not in query.lower()
            ][:10]
        }

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del índice."""
        if not self._inverted_index:
            self._load_index()

        return {
            "available": self.is_available(),
            "model": self.model_name,
            "documents": len(self._doc_vectors),
            "unique_terms": len(self._inverted_index),
            "index_path": str(self._index_path),
            "index_exists": self._inverted_index_path.exists()
        }
