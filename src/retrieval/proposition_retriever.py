"""
Proposition Retriever — Búsqueda en la colección de proposiciones atómicas.

Busca en la colección Qdrant de proposiciones (separada de la colección
principal de chunks) y opcionalmente expande los resultados a los chunks
padre para proporcionar más contexto.
"""

import os
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


def _prop_id_to_qdrant_id(prop_id: str) -> int:
    """Mapeo determinístico de proposition_id a Qdrant int ID."""
    digest = hashlib.sha256(prop_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


@dataclass
class PropositionSearchResult:
    """Resultado de búsqueda de proposición."""
    proposition_id: str
    content: str
    score: float
    parent_chunk_id: str
    doc_id: str
    doc_title: str
    header_path: str
    parent_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PropositionRetriever:
    """
    Retriever para proposiciones atómicas almacenadas en Qdrant.

    Busca en la colección de proposiciones y opcionalmente expande
    resultados a sus chunks padre para contexto adicional.
    """

    COLLECTION_SUFFIX = "_propositions"

    def __init__(
        self,
        indices_dir: Path,
        collection_name: str = "quantum_library",
        embedding_model: str = "text-embedding-3-large",
        embedding_dimensions: int = 3072,
        use_cache: bool = True,
    ):
        """
        Args:
            indices_dir: Directorio de índices.
            collection_name: Nombre base de la colección (se añade _propositions).
            embedding_model: Modelo de embeddings.
            embedding_dimensions: Dimensiones del vector.
            use_cache: Si cachear embeddings de queries.
        """
        self.indices_dir = Path(indices_dir)
        self.collection_name = collection_name + self.COLLECTION_SUFFIX
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.use_cache = use_cache

        self._qdrant_client = None
        self._embedding_client = None
        self._chunks_store = None

    # ------------------------------------------------------------------
    # Inicialización lazy
    # ------------------------------------------------------------------

    def _init_qdrant(self):
        """Inicializa cliente Qdrant."""
        if self._qdrant_client is not None:
            return

        from qdrant_client import QdrantClient

        qdrant_url = os.getenv("QDRANT_URL")
        if qdrant_url:
            self._qdrant_client = QdrantClient(url=qdrant_url)
        else:
            qdrant_path = self.indices_dir / "qdrant"
            self._qdrant_client = QdrantClient(path=str(qdrant_path))

        # Verificar que la colección existe
        collections = self._qdrant_client.get_collections().collections
        if self.collection_name not in [c.name for c in collections]:
            logger.warning(
                f"Colección '{self.collection_name}' no encontrada. "
                f"Ejecuta indexación con --propositions primero."
            )

    def _init_embeddings(self):
        """Inicializa cliente de embeddings."""
        if self._embedding_client is not None:
            return

        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY no configurada.")
        self._embedding_client = OpenAI(api_key=api_key)

    def _load_chunks_store(self):
        """Carga almacén de chunks para expandir a padres."""
        if self._chunks_store is not None:
            return

        import pickle

        chunks_path = self.indices_dir / "chunks.pkl"
        if chunks_path.exists():
            with open(chunks_path, "rb") as f:
                self._chunks_store = pickle.load(f)
        else:
            self._chunks_store = {}

    # ------------------------------------------------------------------
    # Búsqueda
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
        expand_to_parent: bool = True,
    ) -> List[PropositionSearchResult]:
        """
        Busca proposiciones relevantes.

        Args:
            query: Consulta.
            top_k: Número máximo de resultados.
            score_threshold: Score mínimo.
            expand_to_parent: Si incluir contenido del chunk padre.

        Returns:
            Lista de resultados.
        """
        self._init_qdrant()
        self._init_embeddings()

        # Generar embedding de la query
        query_embedding = self._get_query_embedding(query)

        # Buscar en Qdrant
        try:
            results = self._qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
            )
        except Exception as e:
            logger.error(f"Error buscando proposiciones: {e}")
            return []

        # Convertir a PropositionSearchResult
        search_results = []
        for hit in results:
            payload = hit.payload or {}
            result = PropositionSearchResult(
                proposition_id=payload.get("proposition_id", ""),
                content=payload.get("content", ""),
                score=hit.score,
                parent_chunk_id=payload.get("parent_chunk_id", ""),
                doc_id=payload.get("doc_id", ""),
                doc_title=payload.get("doc_title", ""),
                header_path=payload.get("header_path", ""),
            )
            search_results.append(result)

        # Expandir a chunks padre si se solicita
        if expand_to_parent and search_results:
            search_results = self._expand_to_parents(search_results)

        return search_results

    # ------------------------------------------------------------------
    # Expansión a chunks padre
    # ------------------------------------------------------------------

    def _expand_to_parents(
        self, results: List[PropositionSearchResult]
    ) -> List[PropositionSearchResult]:
        """Añade contenido del chunk padre a cada resultado."""
        self._load_chunks_store()

        for result in results:
            parent_id = result.parent_chunk_id
            if parent_id and parent_id in self._chunks_store:
                parent_chunk = self._chunks_store[parent_id]
                result.parent_content = parent_chunk.content

        return results

    def to_retrieval_results(
        self,
        results: List[PropositionSearchResult],
        use_parent_content: bool = True,
    ) -> list:
        """
        Convierte PropositionSearchResult a RetrievalResult para
        integración con el pipeline existente.

        Agrupa por chunk padre y usa el contenido del padre como contenido.
        """
        from .fusion import RetrievalResult, RetrieverType

        # Agrupar por chunk padre
        parent_groups: Dict[str, List[PropositionSearchResult]] = {}
        for r in results:
            key = r.parent_chunk_id or r.proposition_id
            if key not in parent_groups:
                parent_groups[key] = []
            parent_groups[key].append(r)

        retrieval_results = []
        for parent_id, props in parent_groups.items():
            # Usar el mejor score del grupo
            best = max(props, key=lambda p: p.score)

            if use_parent_content and best.parent_content:
                content = best.parent_content
            else:
                content = "\n".join(p.content for p in props)

            retrieval_results.append(
                RetrievalResult(
                    chunk_id=parent_id,
                    content=content,
                    score=best.score,
                    doc_id=best.doc_id,
                    doc_title=best.doc_title,
                    header_path=best.header_path,
                    sources=[RetrieverType.VECTOR],
                    source_scores={"proposition": best.score},
                    metadata={
                        "from_propositions": True,
                        "proposition_count": len(props),
                        "proposition_ids": [p.proposition_id for p in props],
                    },
                )
            )

        retrieval_results.sort(key=lambda x: x.score, reverse=True)
        return retrieval_results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_query_embedding(self, query: str) -> List[float]:
        """Genera embedding para una query."""
        response = self._embedding_client.embeddings.create(
            input=query,
            model=self.embedding_model,
        )
        return response.data[0].embedding
