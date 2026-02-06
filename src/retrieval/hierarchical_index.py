"""
Hierarchical Summarization Index - Resúmenes jerárquicos para routing.

Crea índices de resúmenes a múltiples niveles:
- Documento: Resumen general del libro/paper
- Capítulo: Resumen de cada capítulo/sección principal
- Sección: Resumen de subsecciones

Permite routing inteligente: primero encuentra documentos/capítulos relevantes,
luego busca chunks específicos solo en esas secciones.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from enum import Enum
import logging
import pickle
import json

logger = logging.getLogger(__name__)


class SummaryLevel(Enum):
    """Niveles de resumen."""
    DOCUMENT = "document"    # Libro/paper completo
    CHAPTER = "chapter"      # Capítulo/sección principal
    SECTION = "section"      # Subsección


@dataclass
class HierarchicalSummary:
    """Resumen en la jerarquía."""
    summary_id: str
    level: SummaryLevel
    title: str
    summary: str
    parent_id: Optional[str] = None  # ID del nivel superior
    children_ids: List[str] = field(default_factory=list)  # IDs de niveles inferiores
    chunk_ids: List[str] = field(default_factory=list)  # Chunks contenidos
    doc_id: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary_id": self.summary_id,
            "level": self.level.value,
            "title": self.title,
            "summary": self.summary,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "chunk_ids": self.chunk_ids,
            "doc_id": self.doc_id,
            "metadata": self.metadata
        }


@dataclass
class RoutingResult:
    """Resultado del routing jerárquico."""
    query: str
    matched_documents: List[Tuple[str, float]]  # (doc_id, score)
    matched_chapters: List[Tuple[str, float]]   # (chapter_id, score)
    matched_sections: List[Tuple[str, float]]   # (section_id, score)
    candidate_chunk_ids: Set[str]               # Chunks a buscar
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "matched_documents": self.matched_documents,
            "matched_chapters": self.matched_chapters,
            "matched_sections": self.matched_sections,
            "candidate_chunks": len(self.candidate_chunk_ids),
            "reasoning": self.reasoning
        }


class HierarchicalSummaryIndex:
    """
    Índice de resúmenes jerárquicos para routing.

    Flujo de uso:
    1. Durante indexación: generar resúmenes por nivel
    2. Durante query: routing top-down (doc -> cap -> sec -> chunks)
    3. Búsqueda final solo en chunks candidatos

    Uso:
        index = HierarchicalSummaryIndex(indices_dir)
        index.build_from_chunks(chunks)

        # En query time
        routing = index.route_query("¿Qué es la QFT?")
        # Buscar solo en routing.candidate_chunk_ids
    """

    DOCUMENT_SUMMARY_PROMPT = """Resume este documento académico en 2-3 párrafos.

TÍTULO: {title}

TABLA DE CONTENIDOS / SECCIONES:
{toc}

MUESTRA DEL CONTENIDO:
{sample}

El resumen debe:
1. Capturar el tema principal y alcance del documento
2. Mencionar los conceptos clave cubiertos
3. Indicar el nivel (introductorio, avanzado, técnico)

Responde SOLO con el resumen, sin preámbulos."""

    CHAPTER_SUMMARY_PROMPT = """Resume este capítulo/sección en 1-2 párrafos.

DOCUMENTO: {doc_title}
CAPÍTULO: {chapter_title}

SUBSECCIONES:
{subsections}

MUESTRA DEL CONTENIDO:
{sample}

El resumen debe:
1. Explicar qué cubre este capítulo específicamente
2. Mencionar los conceptos principales introducidos
3. Indicar prerequisitos o conexiones con otros capítulos

Responde SOLO con el resumen."""

    def __init__(
        self,
        indices_dir: Path,
        embedding_provider: str = "openai"
    ):
        """
        Args:
            indices_dir: Directorio para índices
            embedding_provider: Proveedor de embeddings
        """
        self.indices_dir = Path(indices_dir)
        self.embedding_provider = embedding_provider

        self._index_path = self.indices_dir / "hierarchical_summaries"
        self._summaries_path = self._index_path / "summaries.pkl"
        self._embeddings_path = self._index_path / "summary_embeddings.pkl"

        self._summaries: Dict[str, HierarchicalSummary] = {}
        self._summary_embeddings: Dict[str, List[float]] = {}

        # Índices por nivel para búsqueda rápida
        self._by_level: Dict[SummaryLevel, List[str]] = {
            SummaryLevel.DOCUMENT: [],
            SummaryLevel.CHAPTER: [],
            SummaryLevel.SECTION: []
        }
        self._by_doc: Dict[str, List[str]] = {}  # doc_id -> summary_ids

    def build_from_chunks(
        self,
        chunks: List[Any],
        generate_summaries: bool = True,
        batch_size: int = 5
    ) -> Dict[str, Any]:
        """
        Construye índice jerárquico desde chunks.

        Args:
            chunks: Lista de chunks con metadata de jerarquía
            generate_summaries: Si generar resúmenes con LLM
            batch_size: Chunks por batch para LLM

        Returns:
            Estadísticas de construcción
        """
        logger.info(f"Construyendo índice jerárquico desde {len(chunks)} chunks")

        # 1. Agrupar chunks por documento y sección
        doc_chunks: Dict[str, List[Any]] = {}
        doc_sections: Dict[str, Dict[str, List[Any]]] = {}  # doc -> section -> chunks

        for chunk in chunks:
            doc_id = chunk.doc_id
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
                doc_sections[doc_id] = {}

            doc_chunks[doc_id].append(chunk)

            # Extraer sección del header_path
            section = self._extract_section(chunk.header_path)
            if section not in doc_sections[doc_id]:
                doc_sections[doc_id][section] = []
            doc_sections[doc_id][section].append(chunk)

        # 2. Crear resúmenes por nivel
        stats = {"documents": 0, "chapters": 0, "sections": 0}

        for doc_id, chunks_list in doc_chunks.items():
            doc_title = chunks_list[0].doc_title

            # Resumen de documento
            doc_summary = self._create_document_summary(
                doc_id, doc_title, chunks_list, generate_summaries
            )
            self._summaries[doc_summary.summary_id] = doc_summary
            self._by_level[SummaryLevel.DOCUMENT].append(doc_summary.summary_id)
            self._by_doc[doc_id] = [doc_summary.summary_id]
            stats["documents"] += 1

            # Resúmenes de secciones
            for section, section_chunks in doc_sections[doc_id].items():
                if not section:
                    continue

                section_summary = self._create_section_summary(
                    doc_id, doc_title, section, section_chunks, generate_summaries
                )
                section_summary.parent_id = doc_summary.summary_id
                doc_summary.children_ids.append(section_summary.summary_id)

                self._summaries[section_summary.summary_id] = section_summary
                self._by_level[SummaryLevel.SECTION].append(section_summary.summary_id)
                self._by_doc[doc_id].append(section_summary.summary_id)
                stats["sections"] += 1

        # 3. Generar embeddings para los resúmenes
        if generate_summaries:
            self._generate_summary_embeddings()

        # 4. Guardar
        self._save_index()

        logger.info(
            f"Índice jerárquico construido: {stats['documents']} docs, "
            f"{stats['sections']} secciones"
        )

        return stats

    def _extract_section(self, header_path: str) -> str:
        """Extrae la sección principal del header_path."""
        if not header_path:
            return ""

        # Tomar primer nivel del path
        parts = header_path.split(" > ")
        return parts[0] if parts else ""

    def _create_document_summary(
        self,
        doc_id: str,
        doc_title: str,
        chunks: List[Any],
        generate: bool
    ) -> HierarchicalSummary:
        """Crea resumen de documento."""
        summary_id = f"doc_{doc_id}"

        # Extraer TOC de headers únicos
        headers = list(dict.fromkeys([c.header_path for c in chunks if c.header_path]))
        toc = "\n".join(headers[:20])

        # Muestra del contenido
        sample_chunks = chunks[:5] + chunks[len(chunks)//2:len(chunks)//2+3]
        sample = "\n---\n".join([c.content[:300] for c in sample_chunks])

        if generate:
            summary_text = self._generate_summary(
                self.DOCUMENT_SUMMARY_PROMPT.format(
                    title=doc_title,
                    toc=toc,
                    sample=sample
                )
            )
        else:
            # Fallback: usar primeros chunks
            summary_text = f"Documento: {doc_title}. " + " ".join([
                c.content[:100] for c in chunks[:3]
            ])

        return HierarchicalSummary(
            summary_id=summary_id,
            level=SummaryLevel.DOCUMENT,
            title=doc_title,
            summary=summary_text,
            chunk_ids=[c.chunk_id for c in chunks],
            doc_id=doc_id,
            metadata={"total_chunks": len(chunks), "headers": headers[:10]}
        )

    def _create_section_summary(
        self,
        doc_id: str,
        doc_title: str,
        section_title: str,
        chunks: List[Any],
        generate: bool
    ) -> HierarchicalSummary:
        """Crea resumen de sección."""
        summary_id = f"sec_{doc_id}_{hash(section_title) % 10000:04d}"

        # Subsecciones
        subsections = list(dict.fromkeys([
            c.header_path.replace(section_title + " > ", "")
            for c in chunks if c.header_path.startswith(section_title)
        ]))

        sample = "\n---\n".join([c.content[:200] for c in chunks[:4]])

        if generate and len(chunks) >= 3:
            summary_text = self._generate_summary(
                self.CHAPTER_SUMMARY_PROMPT.format(
                    doc_title=doc_title,
                    chapter_title=section_title,
                    subsections="\n".join(subsections[:10]),
                    sample=sample
                )
            )
        else:
            summary_text = f"Sección: {section_title}. " + chunks[0].content[:200]

        return HierarchicalSummary(
            summary_id=summary_id,
            level=SummaryLevel.SECTION,
            title=section_title,
            summary=summary_text,
            chunk_ids=[c.chunk_id for c in chunks],
            doc_id=doc_id,
            metadata={"subsections": subsections[:5]}
        )

    def _generate_summary(self, prompt: str) -> str:
        """Genera resumen usando LLM."""
        from src.llm_provider import complete as llm_complete

        try:
            response = llm_complete(
                prompt=prompt,
                system="Eres un asistente que genera resúmenes concisos de textos académicos.",
                temperature=0.3,
                max_tokens=500
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Error generando resumen: {e}")
            return ""

    def _generate_summary_embeddings(self):
        """Genera embeddings para todos los resúmenes."""
        from ..ingestion.embeddings import get_embedding

        logger.info(f"Generando embeddings para {len(self._summaries)} resúmenes")

        for summary_id, summary in self._summaries.items():
            try:
                # Combinar título y resumen para embedding
                text = f"{summary.title}\n\n{summary.summary}"
                embedding = get_embedding(text)
                self._summary_embeddings[summary_id] = embedding
                summary.embedding = embedding
            except Exception as e:
                logger.warning(f"Error embedding {summary_id}: {e}")

    def _save_index(self):
        """Guarda índice a disco."""
        self._index_path.mkdir(parents=True, exist_ok=True)

        with open(self._summaries_path, 'wb') as f:
            pickle.dump(self._summaries, f)

        with open(self._embeddings_path, 'wb') as f:
            pickle.dump(self._summary_embeddings, f)

        # Guardar metadata en JSON para inspección
        meta = {
            "documents": len(self._by_level[SummaryLevel.DOCUMENT]),
            "chapters": len(self._by_level[SummaryLevel.CHAPTER]),
            "sections": len(self._by_level[SummaryLevel.SECTION]),
            "total_summaries": len(self._summaries)
        }
        with open(self._index_path / "meta.json", 'w') as f:
            json.dump(meta, f, indent=2)

    def _load_index(self):
        """Carga índice desde disco."""
        if self._summaries_path.exists():
            with open(self._summaries_path, 'rb') as f:
                self._summaries = pickle.load(f)

            # Reconstruir índices
            self._by_level = {level: [] for level in SummaryLevel}
            self._by_doc = {}

            for sid, summary in self._summaries.items():
                self._by_level[summary.level].append(sid)
                if summary.doc_id not in self._by_doc:
                    self._by_doc[summary.doc_id] = []
                self._by_doc[summary.doc_id].append(sid)

        if self._embeddings_path.exists():
            with open(self._embeddings_path, 'rb') as f:
                self._summary_embeddings = pickle.load(f)

    def route_query(
        self,
        query: str,
        top_docs: int = 3,
        top_sections: int = 5,
        min_score: float = 0.5
    ) -> RoutingResult:
        """
        Enruta query a través de la jerarquía.

        Args:
            query: Query del usuario
            top_docs: Documentos top a considerar
            top_sections: Secciones top por documento
            min_score: Score mínimo para incluir

        Returns:
            RoutingResult con chunks candidatos
        """
        if not self._summaries:
            self._load_index()

        from ..ingestion.embeddings import get_embedding
        import numpy as np

        # Embedding de query
        query_embedding = np.array(get_embedding(query))

        # 1. Buscar documentos relevantes
        doc_scores = []
        for doc_sid in self._by_level[SummaryLevel.DOCUMENT]:
            if doc_sid in self._summary_embeddings:
                doc_emb = np.array(self._summary_embeddings[doc_sid])
                score = float(np.dot(query_embedding, doc_emb) /
                            (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)))
                doc_scores.append((doc_sid, score))

        doc_scores.sort(key=lambda x: -x[1])
        matched_docs = [(sid, score) for sid, score in doc_scores[:top_docs] if score >= min_score]

        if not matched_docs:
            # Fallback: usar todos los documentos
            matched_docs = doc_scores[:top_docs]

        # 2. Buscar secciones relevantes dentro de docs seleccionados
        candidate_chunk_ids = set()
        matched_sections = []

        for doc_sid, doc_score in matched_docs:
            doc_summary = self._summaries[doc_sid]

            # Añadir todos los chunks del doc como candidatos
            candidate_chunk_ids.update(doc_summary.chunk_ids)

            # Buscar en secciones hijas
            for section_sid in doc_summary.children_ids:
                if section_sid in self._summary_embeddings:
                    sec_emb = np.array(self._summary_embeddings[section_sid])
                    score = float(np.dot(query_embedding, sec_emb) /
                                (np.linalg.norm(query_embedding) * np.linalg.norm(sec_emb)))

                    if score >= min_score * 0.8:
                        matched_sections.append((section_sid, score))
                        # Priorizar chunks de secciones relevantes
                        sec_summary = self._summaries[section_sid]
                        candidate_chunk_ids.update(sec_summary.chunk_ids)

        matched_sections.sort(key=lambda x: -x[1])
        matched_sections = matched_sections[:top_sections]

        # Construir reasoning
        doc_titles = [self._summaries[sid].title for sid, _ in matched_docs]
        sec_titles = [self._summaries[sid].title for sid, _ in matched_sections[:3]]

        reasoning = (
            f"Query relacionada con documentos: {', '.join(doc_titles[:2])}. "
            f"Secciones más relevantes: {', '.join(sec_titles)}. "
            f"{len(candidate_chunk_ids)} chunks candidatos identificados."
        )

        return RoutingResult(
            query=query,
            matched_documents=matched_docs,
            matched_chapters=[],  # No usamos nivel chapter en esta implementación
            matched_sections=matched_sections,
            candidate_chunk_ids=candidate_chunk_ids,
            reasoning=reasoning
        )

    def get_summary(self, summary_id: str) -> Optional[HierarchicalSummary]:
        """Obtiene un resumen por ID."""
        if not self._summaries:
            self._load_index()
        return self._summaries.get(summary_id)

    def get_document_summaries(self) -> List[HierarchicalSummary]:
        """Obtiene todos los resúmenes de documentos."""
        if not self._summaries:
            self._load_index()
        return [
            self._summaries[sid]
            for sid in self._by_level[SummaryLevel.DOCUMENT]
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del índice."""
        if not self._summaries:
            self._load_index()

        return {
            "documents": len(self._by_level[SummaryLevel.DOCUMENT]),
            "chapters": len(self._by_level[SummaryLevel.CHAPTER]),
            "sections": len(self._by_level[SummaryLevel.SECTION]),
            "total_summaries": len(self._summaries),
            "embeddings": len(self._summary_embeddings),
            "index_path": str(self._index_path)
        }


class HierarchicalRetriever:
    """
    Retriever que usa routing jerárquico antes de búsqueda fina.

    Combina:
    1. Routing top-down por resúmenes
    2. Búsqueda densa/BM25 solo en chunks candidatos
    """

    def __init__(
        self,
        base_retriever,
        hierarchical_index: HierarchicalSummaryIndex
    ):
        """
        Args:
            base_retriever: Retriever base (UnifiedRetriever)
            hierarchical_index: Índice de resúmenes jerárquicos
        """
        self.base_retriever = base_retriever
        self.hier_index = hierarchical_index

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_routing: bool = True,
        **kwargs
    ) -> Tuple[List[Any], Optional[RoutingResult]]:
        """
        Busca con routing jerárquico.

        Args:
            query: Query del usuario
            top_k: Resultados a retornar
            use_routing: Si usar routing (False = búsqueda normal)
            **kwargs: Parámetros para retriever base

        Returns:
            Tuple (resultados, routing_result)
        """
        if not use_routing:
            return self.base_retriever.search(query, top_k=top_k, **kwargs), None

        # 1. Routing jerárquico
        routing = self.hier_index.route_query(query)

        if not routing.candidate_chunk_ids:
            # Fallback a búsqueda normal
            logger.info("Routing sin candidatos, usando búsqueda completa")
            return self.base_retriever.search(query, top_k=top_k, **kwargs), routing

        logger.info(
            f"Routing: {len(routing.candidate_chunk_ids)} chunks candidatos "
            f"de {len(routing.matched_documents)} docs"
        )

        # 2. Búsqueda en chunks candidatos
        # Nota: esto requiere que el retriever base soporte filtrado por chunk_id
        # Si no, hacemos búsqueda normal y filtramos después

        results = self.base_retriever.search(
            query,
            top_k=top_k * 3,  # Más para compensar filtrado
            **kwargs
        )

        # Filtrar solo chunks candidatos y boost
        filtered_results = []
        for r in results:
            if r.chunk_id in routing.candidate_chunk_ids:
                # Pequeño boost por estar en routing
                r.score *= 1.1
                filtered_results.append(r)

        # Ordenar y limitar
        filtered_results.sort(key=lambda x: -x.score)
        final_results = filtered_results[:top_k]

        # Si muy pocos resultados, añadir de búsqueda general
        if len(final_results) < top_k // 2:
            for r in results:
                if r.chunk_id not in routing.candidate_chunk_ids:
                    if len(final_results) < top_k:
                        final_results.append(r)

        return final_results, routing
