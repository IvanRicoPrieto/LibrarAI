"""
ColBERT Retriever - Late Interaction para matching preciso token-a-token.

ColBERT (Contextualized Late Interaction over BERT) computa embeddings
a nivel de token y hace matching con MaxSim, logrando mejor precisión
que embeddings de documento completo.

Usa RAGatouille o colbert-ai según disponibilidad.

Requiere:
    pip install ragatouille
    # o
    pip install colbert-ai
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import pickle

logger = logging.getLogger(__name__)


@dataclass
class ColBERTResult:
    """Resultado de búsqueda ColBERT."""
    chunk_id: str
    content: str
    score: float
    doc_id: str = ""
    doc_title: str = ""
    header_path: str = ""
    rank: int = 0
    passage_id: int = 0

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
            sources=[RetrieverType.COLBERT],
            source_scores={"colbert": self.score},
            metadata=metadata or {}
        )


class ColBERTRetriever:
    """
    Retriever basado en ColBERT para late interaction.

    Características:
    - Embeddings a nivel de token para matching preciso
    - MaxSim scoring entre tokens de query y documento
    - Mejor precisión que embeddings densos (~10-15% mejora)
    - Más lento pero más preciso

    Uso:
        retriever = ColBERTRetriever(indices_dir)
        retriever.index(chunks)
        results = retriever.search("¿Qué es un qubit?", top_k=10)
    """

    def __init__(
        self,
        indices_dir: Path,
        model_name: str = "colbert-ir/colbertv2.0",
        index_name: str = "quantum_library_colbert",
        use_gpu: bool = False
    ):
        """
        Args:
            indices_dir: Directorio para almacenar índices
            model_name: Modelo ColBERT a usar
            index_name: Nombre del índice
            use_gpu: Si usar GPU para indexación/búsqueda
        """
        self.indices_dir = Path(indices_dir)
        self.model_name = model_name
        self.index_name = index_name
        self.use_gpu = use_gpu

        self._index_path = self.indices_dir / "colbert" / index_name
        self._chunks_map_path = self._index_path / "chunks_map.pkl"
        self._rag = None
        self._chunks_map: Dict[int, str] = {}  # passage_id -> chunk_id

        self._check_availability()

    def _check_availability(self):
        """Verifica que ColBERT está disponible."""
        self._backend = None

        try:
            from ragatouille import RAGPretrainedModel
            self._backend = "ragatouille"
            logger.info("ColBERT backend: RAGatouille")
        except ImportError:
            pass

        if not self._backend:
            try:
                import colbert
                self._backend = "colbert-ai"
                logger.info("ColBERT backend: colbert-ai")
            except ImportError:
                pass

        if not self._backend:
            logger.warning(
                "ColBERT no disponible. Instala: pip install ragatouille"
            )

    def is_available(self) -> bool:
        """Retorna True si ColBERT está disponible."""
        return self._backend is not None

    def _get_rag(self):
        """Obtiene o inicializa el modelo RAGatouille."""
        if self._rag is not None:
            return self._rag

        if self._backend == "ragatouille":
            from ragatouille import RAGPretrainedModel

            # Intentar cargar índice existente
            if self._index_path.exists():
                try:
                    self._rag = RAGPretrainedModel.from_index(
                        str(self._index_path)
                    )
                    self._load_chunks_map()
                    logger.info(f"Índice ColBERT cargado: {self._index_path}")
                    return self._rag
                except Exception as e:
                    logger.warning(f"Error cargando índice: {e}")

            # Crear nuevo modelo
            self._rag = RAGPretrainedModel.from_pretrained(self.model_name)
            return self._rag

        elif self._backend == "colbert-ai":
            # Implementación con colbert-ai directa
            raise NotImplementedError(
                "Backend colbert-ai no implementado. Usa RAGatouille."
            )

        return None

    def _load_chunks_map(self):
        """Carga el mapeo passage_id -> chunk_id."""
        if self._chunks_map_path.exists():
            with open(self._chunks_map_path, 'rb') as f:
                self._chunks_map = pickle.load(f)

    def _save_chunks_map(self):
        """Guarda el mapeo passage_id -> chunk_id."""
        self._chunks_map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._chunks_map_path, 'wb') as f:
            pickle.dump(self._chunks_map, f)

    def index(
        self,
        chunks: List[Any],
        batch_size: int = 32,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Indexa chunks con ColBERT.

        Args:
            chunks: Lista de chunks a indexar
            batch_size: Tamaño de batch para indexación
            force: Si forzar reindexación

        Returns:
            Dict con estadísticas
        """
        if not self.is_available():
            return {"error": "ColBERT no disponible"}

        if self._index_path.exists() and not force:
            logger.info("Índice ColBERT ya existe, saltando indexación")
            return {"skipped": True, "reason": "index exists"}

        logger.info(f"Indexando {len(chunks)} chunks con ColBERT...")

        rag = self._get_rag()

        # Preparar documentos y metadata
        documents = []
        document_ids = []
        document_metadatas = []

        for i, chunk in enumerate(chunks):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            chunk_id = chunk.chunk_id if hasattr(chunk, 'chunk_id') else f"chunk_{i}"

            documents.append(content)
            document_ids.append(chunk_id)
            document_metadatas.append({
                "chunk_id": chunk_id,
                "doc_id": getattr(chunk, 'doc_id', ''),
                "doc_title": getattr(chunk, 'doc_title', ''),
                "header_path": getattr(chunk, 'header_path', ''),
            })

            # Guardar mapeo
            self._chunks_map[i] = chunk_id

        # Indexar
        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)

            rag.index(
                collection=documents,
                document_ids=document_ids,
                document_metadatas=document_metadatas,
                index_name=self.index_name,
                max_document_length=512,
                split_documents=False,  # Ya están chunkeados
            )

            self._save_chunks_map()

            logger.info(f"Indexación ColBERT completada: {len(documents)} documentos")

            return {
                "indexed": len(documents),
                "index_path": str(self._index_path)
            }

        except Exception as e:
            logger.error(f"Error en indexación ColBERT: {e}")
            return {"error": str(e)}

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[ColBERTResult]:
        """
        Busca usando ColBERT late interaction.

        Args:
            query: Query de búsqueda
            top_k: Número de resultados

        Returns:
            Lista de ColBERTResult
        """
        if not self.is_available():
            logger.warning("ColBERT no disponible para búsqueda")
            return []

        rag = self._get_rag()
        if rag is None:
            return []

        try:
            results = rag.search(query=query, k=top_k)

            colbert_results = []
            for i, result in enumerate(results):
                # RAGatouille retorna dict o objeto según versión
                if isinstance(result, dict):
                    content = result.get('content', '')
                    score = result.get('score', 0.0)
                    doc_id = result.get('document_id', '')
                    metadata = result.get('document_metadata', {})
                else:
                    content = getattr(result, 'content', '')
                    score = getattr(result, 'score', 0.0)
                    doc_id = getattr(result, 'document_id', '')
                    metadata = getattr(result, 'document_metadata', {})

                chunk_id = metadata.get('chunk_id', doc_id)

                colbert_results.append(ColBERTResult(
                    chunk_id=chunk_id,
                    content=content,
                    score=score,
                    doc_id=metadata.get('doc_id', ''),
                    doc_title=metadata.get('doc_title', ''),
                    header_path=metadata.get('header_path', ''),
                    rank=i + 1,
                    passage_id=i
                ))

            return colbert_results

        except Exception as e:
            logger.error(f"Error en búsqueda ColBERT: {e}")
            return []

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
            metadata = {}
            if chunks_store and r.chunk_id in chunks_store:
                chunk = chunks_store[r.chunk_id]
                metadata = {
                    "section_hierarchy": getattr(chunk, 'section_hierarchy', []),
                    "section_number": getattr(chunk, 'section_number', ''),
                    "topic_summary": getattr(chunk, 'topic_summary', ''),
                }

            retrieval_results.append(r.to_retrieval_result(metadata))

        return retrieval_results

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del índice."""
        stats = {
            "available": self.is_available(),
            "backend": self._backend,
            "index_exists": self._index_path.exists(),
            "index_path": str(self._index_path),
            "chunks_mapped": len(self._chunks_map)
        }

        return stats


class ColBERTReranker:
    """
    Usa ColBERT para re-rankear resultados de otro retriever.

    Más ligero que indexar todo con ColBERT - solo re-rankea
    los top-N resultados de búsqueda vectorial/BM25.
    """

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        """
        Args:
            model_name: Modelo ColBERT a usar
        """
        self.model_name = model_name
        self._model = None
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Verifica disponibilidad."""
        try:
            from ragatouille import RAGPretrainedModel
            return True
        except ImportError:
            return False

    def _get_model(self):
        """Obtiene modelo para reranking."""
        if self._model is None and self._available:
            from ragatouille import RAGPretrainedModel
            self._model = RAGPretrainedModel.from_pretrained(
                self.model_name,
                index_root=None  # No necesitamos índice para reranking
            )
        return self._model

    def rerank(
        self,
        query: str,
        results: List[Any],
        top_k: Optional[int] = None
    ) -> List[Any]:
        """
        Re-rankea resultados usando ColBERT scoring.

        Args:
            query: Query original
            results: Resultados a re-rankear (con .content)
            top_k: Número de resultados a retornar (None = todos)

        Returns:
            Resultados re-rankeados
        """
        if not self._available or not results:
            return results

        model = self._get_model()
        if model is None:
            return results

        try:
            # Preparar documentos para reranking
            documents = [r.content for r in results]

            # Rerank
            reranked = model.rerank(
                query=query,
                documents=documents,
                k=top_k or len(results)
            )

            # Mapear scores a resultados originales
            result_map = {r.content[:100]: r for r in results}
            reranked_results = []

            for item in reranked:
                if isinstance(item, dict):
                    content = item.get('content', '')[:100]
                    score = item.get('score', 0.0)
                else:
                    content = getattr(item, 'content', '')[:100]
                    score = getattr(item, 'score', 0.0)

                if content in result_map:
                    original = result_map[content]
                    # Crear copia con nuevo score
                    reranked_results.append(type(original)(
                        chunk_id=original.chunk_id,
                        content=original.content,
                        score=score,  # Score de ColBERT
                        doc_id=original.doc_id,
                        doc_title=original.doc_title,
                        header_path=original.header_path,
                        sources=getattr(original, 'sources', []),
                        source_scores={
                            **getattr(original, 'source_scores', {}),
                            "colbert_rerank": score
                        },
                        parent_id=getattr(original, 'parent_id', None),
                        level=getattr(original, 'level', 'MICRO'),
                        token_count=getattr(original, 'token_count', 0),
                        metadata=getattr(original, 'metadata', {})
                    ))

            return reranked_results if reranked_results else results

        except Exception as e:
            logger.warning(f"Error en ColBERT reranking: {e}")
            return results
