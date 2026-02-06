"""
Agent Interface - API optimizada para consumo por agentes IA.

Diseñada para que Claude Code u otros agentes puedan:
1. Explorar qué contenido existe sobre un tema
2. Recuperar contenido exhaustivo (no limitado a top-k)
3. Hacer preguntas específicas con respuestas citadas
4. Verificar afirmaciones contra las fuentes
5. Generar citas formateadas para referencias

Todos los outputs son dataclasses serializables a JSON estructurado.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class VerificationStatus(Enum):
    """Estado de verificación de una afirmación."""
    SUPPORTED = "supported"      # Encontrada y confirmada en fuentes
    CONTRADICTED = "contradicted"  # Contradice las fuentes
    NOT_FOUND = "not_found"      # No hay información en las fuentes
    PARTIAL = "partial"          # Parcialmente soportada


class CitationStyle(Enum):
    """Estilos de citación soportados."""
    APA = "apa"
    IEEE = "ieee"
    CHICAGO = "chicago"
    MARKDOWN = "markdown"
    INLINE = "inline"


# =============================================================================
# Dataclasses para outputs estructurados
# =============================================================================

@dataclass
class ContentNode:
    """Nodo de contenido en la jerarquía del documento."""
    id: str
    title: str
    level: str  # "document", "chapter", "section", "subsection", "chunk"
    path: str  # "Document > Chapter > Section"
    summary: Optional[str] = None
    token_count: int = 0
    children_count: int = 0
    relevance_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "path": self.path,
            "summary": self.summary,
            "token_count": self.token_count,
            "children_count": self.children_count,
            "relevance_score": self.relevance_score,
        }


@dataclass
class SourceChunk:
    """Chunk de contenido con metadatos completos."""
    chunk_id: str
    content: str
    doc_id: str
    doc_title: str
    section_path: str
    section_number: Optional[str] = None
    page_number: Optional[int] = None
    relevance_score: float = 0.0
    token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "section_path": self.section_path,
            "section_number": self.section_number,
            "page_number": self.page_number,
            "relevance_score": self.relevance_score,
            "token_count": self.token_count,
        }


@dataclass
class ExploreResult:
    """Resultado de exploración: qué contenido existe sobre un tema."""
    query: str
    total_documents: int
    total_relevant_chunks: int
    content_tree: List[ContentNode]
    topic_clusters: List[Dict[str, Any]]
    suggested_queries: List[str]
    coverage_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "total_documents": self.total_documents,
            "total_relevant_chunks": self.total_relevant_chunks,
            "content_tree": [n.to_dict() for n in self.content_tree],
            "topic_clusters": self.topic_clusters,
            "suggested_queries": self.suggested_queries,
            "coverage_summary": self.coverage_summary,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class RetrieveResult:
    """Resultado de recuperación exhaustiva."""
    query: str
    total_chunks: int
    chunks: List[SourceChunk]
    documents_covered: List[str]
    sections_covered: List[str]
    total_tokens: int
    retrieval_strategy: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "total_chunks": self.total_chunks,
            "chunks": [c.to_dict() for c in self.chunks],
            "documents_covered": self.documents_covered,
            "sections_covered": self.sections_covered,
            "total_tokens": self.total_tokens,
            "retrieval_strategy": self.retrieval_strategy,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class CitedClaim:
    """Afirmación con sus citas correspondientes."""
    claim: str
    citations: List[str]  # chunk_ids que soportan la afirmación
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "citations": self.citations,
            "confidence": self.confidence,
        }


@dataclass
class QueryResult:
    """Resultado de una consulta con respuesta citada."""
    query: str
    answer: str
    claims: List[CitedClaim]
    sources_used: List[SourceChunk]
    confidence_score: float
    abstained: bool = False
    abstention_reason: Optional[str] = None
    model: str = ""
    tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "claims": [c.to_dict() for c in self.claims],
            "sources_used": [s.to_dict() for s in self.sources_used],
            "confidence_score": self.confidence_score,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "model": self.model,
            "tokens_used": self.tokens_used,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class VerificationEvidence:
    """Evidencia para verificación de una afirmación."""
    chunk_id: str
    content_excerpt: str
    relevance: str  # "supports", "contradicts", "related"
    quote: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content_excerpt": self.content_excerpt,
            "relevance": self.relevance,
            "quote": self.quote,
        }


@dataclass
class VerifyResult:
    """Resultado de verificación de una afirmación."""
    claim: str
    status: VerificationStatus
    confidence: float
    evidence: List[VerificationEvidence]
    explanation: str
    sources_checked: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "status": self.status.value,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
            "explanation": self.explanation,
            "sources_checked": self.sources_checked,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class FormattedCitation:
    """Cita formateada en un estilo específico."""
    chunk_id: str
    formatted: str
    style: str
    raw_components: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "formatted": self.formatted,
            "style": self.style,
            "raw_components": self.raw_components,
        }


@dataclass
class CiteResult:
    """Resultado de generación de citas."""
    citations: List[FormattedCitation]
    style: str
    total_citations: int
    bibliography: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citations": [c.to_dict() for c in self.citations],
            "style": self.style,
            "total_citations": self.total_citations,
            "bibliography": self.bibliography,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# =============================================================================
# AgentAPI - Interfaz principal
# =============================================================================

class AgentAPI:
    """
    API optimizada para agentes IA.

    Provee acceso estructurado a la biblioteca con outputs tipados
    y formatos JSON consistentes para fácil parsing por agentes.

    Usage:
        api = AgentAPI(indices_dir=Path("indices"))

        # Explorar contenido
        explore = api.explore("algoritmo de Shor")

        # Recuperar contenido exhaustivo
        retrieve = api.retrieve("QFT en Shor", exhaustive=True)

        # Hacer pregunta
        query = api.query("¿Cuál es la complejidad de Shor?")

        # Verificar afirmación
        verify = api.verify("Shor factoriza en tiempo O(n³)")

        # Generar citas
        cite = api.cite(chunk_ids=["nc_5.2.1"], style="apa")
    """

    def __init__(
        self,
        indices_dir: Path,
        config: Optional[Dict] = None,
        use_agentic: bool = True,
        use_colbert: bool = True,
        use_grounding: bool = True,
    ):
        """
        Args:
            indices_dir: Directorio con índices de la biblioteca
            config: Configuración opcional (carga settings.yaml si no se provee)
            use_agentic: Usar loop agéntico para queries complejas
            use_colbert: Usar ColBERT para re-ranking
            use_grounding: Usar verificación de citas
        """
        self.indices_dir = Path(indices_dir)
        self.config = config or self._load_config()
        self.use_agentic = use_agentic
        self.use_colbert = use_colbert
        self.use_grounding = use_grounding

        # Lazy init de componentes
        self._retriever = None
        self._synthesizer = None
        self._graph = None
        self._hierarchical_index = None
        self._nli_verifier = None
        self._chunks_store = None

    def _load_config(self) -> Dict:
        """Carga configuración desde settings.yaml."""
        config_path = self.indices_dir.parent / "config" / "settings.yaml"
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def _init_retriever(self):
        """Inicializa retriever bajo demanda."""
        if self._retriever is not None:
            return

        from ..retrieval.fusion import UnifiedRetriever

        retrieval_config = self.config.get("retrieval", {})
        self._retriever = UnifiedRetriever(
            indices_dir=self.indices_dir,
            use_graph=True,
            use_reranker=False,  # Usamos ColBERT separado
            use_cache=True,
        )

    def _init_synthesizer(self):
        """Inicializa sintetizador bajo demanda."""
        if self._synthesizer is not None:
            return

        from ..generation.synthesizer import ResponseSynthesizer

        gen_config = self.config.get("generation", {})
        self._synthesizer = ResponseSynthesizer(
            temperature=gen_config.get("temperature", 0.3),
            max_output_tokens=gen_config.get("max_tokens", 2000)
        )

    def _init_graph(self):
        """Inicializa grafo de conocimiento bajo demanda."""
        if self._graph is not None:
            return

        try:
            from ..retrieval.graph_retriever import KnowledgeGraph
            graph_path = self.indices_dir / "knowledge_graph.gpickle"
            if graph_path.exists():
                self._graph = KnowledgeGraph.load(graph_path)
        except Exception as e:
            logger.warning(f"No se pudo cargar grafo: {e}")
            self._graph = None

    def _init_hierarchical(self):
        """Inicializa índice jerárquico bajo demanda."""
        if self._hierarchical_index is not None:
            return

        try:
            from ..retrieval.hierarchical_index import HierarchicalSummaryIndex
            self._hierarchical_index = HierarchicalSummaryIndex(self.indices_dir)
        except Exception as e:
            logger.warning(f"No se pudo cargar índice jerárquico: {e}")
            self._hierarchical_index = None

    def _load_chunks_store(self):
        """Carga almacén de chunks."""
        if self._chunks_store is not None:
            return

        import pickle
        chunks_path = self.indices_dir / "chunks.pkl"
        if chunks_path.exists():
            with open(chunks_path, 'rb') as f:
                self._chunks_store = pickle.load(f)
        else:
            self._chunks_store = {}

    def _count_tokens(self, text: str) -> int:
        """Estima tokens en un texto."""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            return len(text) // 4

    # =========================================================================
    # EXPLORE: Descubrir qué contenido existe
    # =========================================================================

    def explore(
        self,
        topic: str,
        max_depth: int = 3,
        include_summaries: bool = True,
    ) -> ExploreResult:
        """
        Explora qué contenido existe sobre un tema.

        Devuelve una estructura jerárquica de documentos, capítulos y secciones
        relevantes, con resúmenes y scores de relevancia.

        Args:
            topic: Tema a explorar
            max_depth: Profundidad máxima de la jerarquía (1=docs, 2=+chapters, 3=+sections)
            include_summaries: Incluir resúmenes de contenido

        Returns:
            ExploreResult con árbol de contenido y metadatos
        """
        self._init_retriever()
        self._init_hierarchical()
        self._init_graph()

        # Búsqueda inicial para identificar documentos relevantes
        initial_results = self._retriever.search(topic, top_k=50)

        # Agrupar por documento
        docs_map: Dict[str, List] = {}
        for r in initial_results:
            if r.doc_id not in docs_map:
                docs_map[r.doc_id] = []
            docs_map[r.doc_id].append(r)

        # Construir árbol de contenido
        content_tree: List[ContentNode] = []

        for doc_id, chunks in docs_map.items():
            # Nivel documento
            doc_title = chunks[0].doc_title if chunks else doc_id
            doc_score = max(c.score for c in chunks)

            doc_node = ContentNode(
                id=doc_id,
                title=doc_title,
                level="document",
                path=doc_title,
                relevance_score=doc_score,
                children_count=len(chunks),
            )

            # Obtener resumen del documento si existe índice jerárquico
            if include_summaries and self._hierarchical_index:
                try:
                    doc_summary = self._hierarchical_index.get_document_summary(doc_id)
                    if doc_summary:
                        doc_node.summary = doc_summary.get("summary", "")
                except Exception:
                    pass

            content_tree.append(doc_node)

            if max_depth >= 2:
                # Nivel capítulo/sección
                sections_map: Dict[str, List] = {}
                for c in chunks:
                    section = c.header_path.split(" > ")[0] if c.header_path else "Sin sección"
                    if section not in sections_map:
                        sections_map[section] = []
                    sections_map[section].append(c)

                for section, section_chunks in sections_map.items():
                    section_score = max(c.score for c in section_chunks)
                    section_node = ContentNode(
                        id=f"{doc_id}_{section}",
                        title=section,
                        level="section",
                        path=f"{doc_title} > {section}",
                        relevance_score=section_score,
                        children_count=len(section_chunks),
                    )
                    content_tree.append(section_node)

        # Generar clusters de temas usando grafo si disponible
        topic_clusters = []
        if self._graph:
            try:
                related_entities = self._graph.get_related_entities(topic, max_hops=2)
                if related_entities:
                    topic_clusters = [
                        {"entity": e, "relation": r, "score": s}
                        for e, r, s in related_entities[:10]
                    ]
            except Exception:
                pass

        # Sugerir queries de seguimiento
        suggested_queries = self._generate_suggested_queries(topic, initial_results)

        # Generar resumen de cobertura
        coverage_summary = (
            f"Encontrados {len(docs_map)} documentos relevantes con "
            f"{len(initial_results)} fragmentos. "
            f"Principales fuentes: {', '.join(list(docs_map.keys())[:3])}."
        )

        return ExploreResult(
            query=topic,
            total_documents=len(docs_map),
            total_relevant_chunks=len(initial_results),
            content_tree=content_tree,
            topic_clusters=topic_clusters,
            suggested_queries=suggested_queries,
            coverage_summary=coverage_summary,
        )

    def _generate_suggested_queries(
        self,
        topic: str,
        results: List,
    ) -> List[str]:
        """Genera queries sugeridas basadas en el contenido encontrado."""
        suggestions = []

        # Extraer términos frecuentes del contenido
        all_content = " ".join(r.content for r in results[:10])

        # Sugerencias básicas basadas en el tema
        suggestions.extend([
            f"¿Qué es {topic}?",
            f"Aplicaciones de {topic}",
            f"Historia de {topic}",
            f"Ejemplos de {topic}",
        ])

        return suggestions[:6]

    # =========================================================================
    # RETRIEVE: Obtener contenido exhaustivo
    # =========================================================================

    def retrieve(
        self,
        query: str,
        exhaustive: bool = False,
        max_chunks: int = 50,
        min_score: float = 0.001,
        expand_context: bool = True,
        difficulty_level: Optional[str] = None,
        math_aware: bool = False,
    ) -> RetrieveResult:
        """
        Recupera contenido relevante para un tema.

        A diferencia de QUERY, no genera respuesta. Solo recupera y devuelve
        los chunks ordenados por relevancia. En modo exhaustivo, recupera
        TODO el contenido relevante sin límite de top-k tradicional.

        Args:
            query: Query de búsqueda
            exhaustive: Si True, recupera todo sin límite estricto
            max_chunks: Máximo de chunks a devolver (solo si exhaustive=False)
            min_score: Score mínimo de relevancia
            expand_context: Expandir chunks a contexto coherente
            difficulty_level: Filtrar por nivel (introductory, intermediate, advanced, research)
                             Puede ser múltiples valores separados por coma: "introductory,intermediate"
            math_aware: Si True, expande query con términos matemáticos equivalentes

        Returns:
            RetrieveResult con chunks y metadatos
        """
        self._init_retriever()

        if exhaustive:
            # Modo exhaustivo: múltiples pasadas con diferentes estrategias
            all_chunks = self._exhaustive_retrieve(
                query, min_score,
                difficulty_level=difficulty_level,
                math_aware=math_aware
            )
        else:
            # Modo normal: top-k estándar
            results = self._retriever.search(
                query, top_k=max_chunks,
                difficulty_level=difficulty_level,
                math_aware=math_aware
            )
            all_chunks = [
                SourceChunk(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    doc_id=r.doc_id,
                    doc_title=r.doc_title,
                    section_path=r.header_path,
                    relevance_score=r.score,
                    token_count=self._count_tokens(r.content),
                )
                for r in results
                if r.score >= min_score
            ]

        # Expandir contexto si se solicita
        if expand_context and all_chunks:
            all_chunks = self._expand_chunks_context(all_chunks, query)

        # Recolectar metadatos
        docs_covered = list(set(c.doc_id for c in all_chunks))
        sections_covered = list(set(c.section_path for c in all_chunks))
        total_tokens = sum(c.token_count for c in all_chunks)

        return RetrieveResult(
            query=query,
            total_chunks=len(all_chunks),
            chunks=all_chunks,
            documents_covered=docs_covered,
            sections_covered=sections_covered,
            total_tokens=total_tokens,
            retrieval_strategy="exhaustive" if exhaustive else "top_k",
        )

    def _exhaustive_retrieve(
        self,
        query: str,
        min_score: float,
        difficulty_level: Optional[str] = None,
        math_aware: bool = False,
    ) -> List[SourceChunk]:
        """Recuperación exhaustiva con múltiples estrategias."""
        seen_ids = set()
        all_chunks = []

        # Estrategia 1: Vector search amplio
        vector_results = self._retriever.search(
            query, top_k=100, vector_only=True,
            difficulty_level=difficulty_level,
            math_aware=math_aware
        ) if hasattr(self._retriever, 'search') else []

        for r in vector_results:
            if r.chunk_id not in seen_ids and r.score >= min_score:
                seen_ids.add(r.chunk_id)
                all_chunks.append(SourceChunk(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    doc_id=r.doc_id,
                    doc_title=r.doc_title,
                    section_path=r.header_path,
                    relevance_score=r.score,
                    token_count=self._count_tokens(r.content),
                ))

        # Estrategia 2: BM25 amplio
        bm25_results = self._retriever.search(
            query, top_k=100, bm25_only=True
        ) if hasattr(self._retriever, 'search') else []

        for r in bm25_results:
            if r.chunk_id not in seen_ids and r.score >= min_score:
                seen_ids.add(r.chunk_id)
                all_chunks.append(SourceChunk(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    doc_id=r.doc_id,
                    doc_title=r.doc_title,
                    section_path=r.header_path,
                    relevance_score=r.score,
                    token_count=self._count_tokens(r.content),
                ))

        # Estrategia 3: Búsqueda híbrida normal
        hybrid_results = self._retriever.search(query, top_k=100)

        for r in hybrid_results:
            if r.chunk_id not in seen_ids and r.score >= min_score:
                seen_ids.add(r.chunk_id)
                all_chunks.append(SourceChunk(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    doc_id=r.doc_id,
                    doc_title=r.doc_title,
                    section_path=r.header_path,
                    relevance_score=r.score,
                    token_count=self._count_tokens(r.content),
                ))

        # Ordenar por relevancia
        all_chunks.sort(key=lambda x: -x.relevance_score)

        return all_chunks

    def _expand_chunks_context(
        self,
        chunks: List[SourceChunk],
        query: str,
    ) -> List[SourceChunk]:
        """Expande chunks a contexto coherente usando ContextExpander."""
        try:
            from ..retrieval.context_expander import ContextExpander

            expander = ContextExpander(
                indices_dir=self.indices_dir,
                chunks_before=2,
                chunks_after=2,
                max_context_tokens=1500,
            )

            expanded = []
            for chunk in chunks:
                try:
                    ctx = expander.expand_chunk(chunk.chunk_id, query)
                    if ctx:
                        expanded.append(SourceChunk(
                            chunk_id=chunk.chunk_id,
                            content=ctx.expanded_content,
                            doc_id=chunk.doc_id,
                            doc_title=chunk.doc_title,
                            section_path=chunk.section_path,
                            relevance_score=chunk.relevance_score,
                            token_count=ctx.token_count,
                        ))
                    else:
                        expanded.append(chunk)
                except Exception:
                    expanded.append(chunk)

            return expanded
        except ImportError:
            return chunks

    # =========================================================================
    # QUERY: Responder preguntas con citas
    # =========================================================================

    def query(
        self,
        question: str,
        top_k: int = 10,
        require_citations: bool = True,
        min_confidence: float = 0.5,
        difficulty_level: Optional[str] = None,
        math_aware: bool = False,
    ) -> QueryResult:
        """
        Responde una pregunta con citas verificables.

        A diferencia de RETRIEVE, genera una respuesta sintetizada
        con afirmaciones citadas y score de confianza.

        Args:
            question: Pregunta a responder
            top_k: Número de fuentes a considerar
            require_citations: Exigir que cada afirmación tenga cita
            min_confidence: Confianza mínima para responder (abstención si no se alcanza)
            difficulty_level: Filtrar por nivel (introductory, intermediate, advanced, research)
            math_aware: Si True, expande query con términos matemáticos equivalentes

        Returns:
            QueryResult con respuesta citada y metadatos
        """
        self._init_retriever()
        self._init_synthesizer()

        # Recuperar fuentes
        results = self._retriever.search(
            question, top_k=top_k,
            difficulty_level=difficulty_level,
            math_aware=math_aware
        )

        if not results:
            return QueryResult(
                query=question,
                answer="",
                claims=[],
                sources_used=[],
                confidence_score=0.0,
                abstained=True,
                abstention_reason="No se encontraron fuentes relevantes",
            )

        # Convertir a SourceChunk
        sources = [
            SourceChunk(
                chunk_id=r.chunk_id,
                content=r.content,
                doc_id=r.doc_id,
                doc_title=r.doc_title,
                section_path=r.header_path,
                relevance_score=r.score,
                token_count=self._count_tokens(r.content),
            )
            for r in results
        ]

        # Generar respuesta con citas si grounding está habilitado
        if self.use_grounding and require_citations:
            try:
                return self._generate_grounded_response(question, sources)
            except Exception as e:
                logger.warning(f"Error en grounded generation: {e}")

        # Fallback: generación normal
        response = self._synthesizer.generate(
            query=question,
            results=results,
            stream=False,
        )

        # Calcular confianza basada en scores de retrieval
        max_score = max(r.score for r in results)
        confidence = min(max_score * 100, 1.0)  # Escalar RRF score

        if confidence < min_confidence:
            return QueryResult(
                query=question,
                answer=response.content,
                claims=[],
                sources_used=sources,
                confidence_score=confidence,
                abstained=True,
                abstention_reason=f"Confianza {confidence:.2f} < umbral {min_confidence}",
                model=response.model,
                tokens_used=response.tokens_input + response.tokens_output,
            )

        return QueryResult(
            query=question,
            answer=response.content,
            claims=[],  # Sin extracción de claims en modo fallback
            sources_used=sources,
            confidence_score=confidence,
            model=response.model,
            tokens_used=response.tokens_input + response.tokens_output,
        )

    def _generate_grounded_response(
        self,
        question: str,
        sources: List[SourceChunk],
    ) -> QueryResult:
        """Genera respuesta con citas verificadas usando CitationGrounding."""
        from ..generation.citation_grounding import CitationGroundingSystem

        grounding = CitationGroundingSystem(
            min_grounding_score=0.8,
            require_all_cited=True,
        )

        # Convertir sources a formato esperado
        from ..retrieval.fusion import RetrievalResult
        retrieval_results = [
            RetrievalResult(
                chunk_id=s.chunk_id,
                content=s.content,
                score=s.relevance_score,
                doc_id=s.doc_id,
                doc_title=s.doc_title,
                header_path=s.section_path,
                retriever_type="hybrid",
            )
            for s in sources
        ]

        # Generar con citas
        content = grounding.generate_with_citations(question, retrieval_results)
        final_content, grounding_result = grounding.enforce_grounding(
            content, retrieval_results
        )

        # Extraer claims con sus citas
        claims = []
        for claim in grounding_result.claims:
            claims.append(CitedClaim(
                claim=claim.claim,
                citations=claim.supporting_chunks,
                confidence=claim.confidence,
            ))

        return QueryResult(
            query=question,
            answer=final_content,
            claims=claims,
            sources_used=sources,
            confidence_score=grounding_result.grounding_score,
            model="grounded_synthesis",
        )

    # =========================================================================
    # VERIFY: Verificar afirmaciones
    # =========================================================================

    def verify(
        self,
        claim: str,
        max_sources: int = 20,
    ) -> VerifyResult:
        """
        Verifica una afirmación contra las fuentes.

        Busca evidencia que soporte, contradiga o no tenga información
        sobre la afirmación dada.

        Args:
            claim: Afirmación a verificar
            max_sources: Máximo de fuentes a revisar

        Returns:
            VerifyResult con estado de verificación y evidencia
        """
        self._init_retriever()

        # Buscar fuentes relevantes
        results = self._retriever.search(claim, top_k=max_sources)

        if not results:
            return VerifyResult(
                claim=claim,
                status=VerificationStatus.NOT_FOUND,
                confidence=0.0,
                evidence=[],
                explanation="No se encontraron fuentes relevantes para verificar esta afirmación.",
                sources_checked=0,
            )

        # Intentar usar NLI verification si está disponible
        try:
            return self._verify_with_nli(claim, results)
        except Exception as e:
            logger.warning(f"NLI verification no disponible: {e}")

        # Fallback: verificación heurística
        return self._verify_heuristic(claim, results)

    def _verify_with_nli(self, claim: str, results: List) -> VerifyResult:
        """Verificación usando Natural Language Inference."""
        # TODO: Implementar cuando se añada NLI verification
        raise NotImplementedError("NLI verification pendiente de implementación")

    def _verify_heuristic(self, claim: str, results: List) -> VerifyResult:
        """Verificación heurística basada en overlap de términos."""
        evidence = []
        supporting = 0

        claim_words = set(claim.lower().split())

        for r in results:
            content_words = set(r.content.lower().split())
            overlap = len(claim_words & content_words) / len(claim_words) if claim_words else 0

            if overlap > 0.3:
                relevance = "supports" if overlap > 0.5 else "related"
                if relevance == "supports":
                    supporting += 1

                evidence.append(VerificationEvidence(
                    chunk_id=r.chunk_id,
                    content_excerpt=r.content[:300] + "...",
                    relevance=relevance,
                ))

        # Determinar estado
        if supporting >= 2:
            status = VerificationStatus.SUPPORTED
            confidence = min(supporting / len(results), 0.8)
        elif supporting == 1:
            status = VerificationStatus.PARTIAL
            confidence = 0.5
        elif evidence:
            status = VerificationStatus.PARTIAL
            confidence = 0.3
        else:
            status = VerificationStatus.NOT_FOUND
            confidence = 0.0

        return VerifyResult(
            claim=claim,
            status=status,
            confidence=confidence,
            evidence=evidence[:5],
            explanation=f"Encontradas {len(evidence)} fuentes relacionadas, {supporting} soportan la afirmación.",
            sources_checked=len(results),
        )

    # =========================================================================
    # CITE: Generar citas formateadas
    # =========================================================================

    def cite(
        self,
        chunk_ids: List[str],
        style: str = "apa",
    ) -> CiteResult:
        """
        Genera citas formateadas para chunks específicos.

        Args:
            chunk_ids: IDs de los chunks a citar
            style: Estilo de citación (apa, ieee, chicago, markdown, inline)

        Returns:
            CiteResult con citas formateadas
        """
        self._load_chunks_store()

        citations = []

        for chunk_id in chunk_ids:
            chunk = self._chunks_store.get(chunk_id)
            if not chunk:
                continue

            # Extraer componentes
            doc_title = getattr(chunk, 'doc_title', 'Unknown')
            section = getattr(chunk, 'header_path', '')
            section_num = getattr(chunk, 'section_number', '')

            raw_components = {
                "doc_title": doc_title,
                "section": section,
                "section_number": section_num,
                "chunk_id": chunk_id,
            }

            # Formatear según estilo
            formatted = self._format_citation(raw_components, style)

            citations.append(FormattedCitation(
                chunk_id=chunk_id,
                formatted=formatted,
                style=style,
                raw_components=raw_components,
            ))

        # Generar bibliografía si hay múltiples citas
        bibliography = None
        if len(citations) > 1:
            bibliography = "\n".join(
                f"[{i+1}] {c.formatted}"
                for i, c in enumerate(citations)
            )

        return CiteResult(
            citations=citations,
            style=style,
            total_citations=len(citations),
            bibliography=bibliography,
        )

    def _format_citation(
        self,
        components: Dict[str, str],
        style: str,
    ) -> str:
        """Formatea una cita en el estilo especificado."""
        doc = components.get("doc_title", "")
        section = components.get("section", "")
        section_num = components.get("section_number", "")

        if style == "apa":
            parts = [doc]
            if section:
                parts.append(f"({section})")
            if section_num:
                parts.append(f"§{section_num}")
            return ", ".join(parts)

        elif style == "ieee":
            parts = [doc]
            if section_num:
                parts.append(f"sec. {section_num}")
            elif section:
                parts.append(section)
            return ", ".join(parts)

        elif style == "chicago":
            result = doc
            if section:
                result += f", \"{section}\""
            if section_num:
                result += f", §{section_num}"
            return result

        elif style == "markdown":
            link = f"[{doc}]"
            if section:
                link += f" > {section}"
            if section_num:
                link += f" (§{section_num})"
            return link

        else:  # inline
            parts = [doc]
            if section_num:
                parts.append(f"§{section_num}")
            elif section:
                parts.append(section)
            return " — ".join(parts)
