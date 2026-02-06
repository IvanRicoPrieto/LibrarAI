"""
Context Expander - Expansión LLM de contexto coherente.

Dado un chunk recuperado, expande hacia arriba y abajo para obtener
un fragmento coherente y autocontenido que tenga sentido para un humano.

Usa LLM para determinar los puntos de corte naturales.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExpandedContext:
    """Fragmento de contexto expandido y coherente."""
    chunk_id: str
    original_content: str
    expanded_content: str
    source_citation: str  # "Nielsen & Chuang, Cap. 5, Sec. 5.2"
    topic_summary: str  # "Explicación del período cuántico"
    relevance_to_query: str  # Explicación de por qué es relevante
    start_chunk_id: str  # Primer chunk incluido en la expansión
    end_chunk_id: str  # Último chunk incluido en la expansión
    token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "original_content": self.original_content,
            "expanded_content": self.expanded_content,
            "source_citation": self.source_citation,
            "topic_summary": self.topic_summary,
            "relevance_to_query": self.relevance_to_query,
            "start_chunk_id": self.start_chunk_id,
            "end_chunk_id": self.end_chunk_id,
            "token_count": self.token_count,
        }


class ContextExpander:
    """
    Expande chunks recuperados a fragmentos coherentes usando LLM.

    Para cada chunk:
    1. Recupera chunks adyacentes del mismo documento
    2. Concatena el contexto extendido
    3. Usa LLM para encontrar puntos de corte naturales
    4. Devuelve el fragmento coherente con metadatos
    """

    EXPANSION_PROMPT = """Eres un asistente que extrae fragmentos coherentes de textos académicos.

Te doy un texto extenso que contiene un fragmento central marcado con [CHUNK_CENTRAL]. Tu tarea es extraer la porción MÍNIMA del texto que:
1. Contenga completamente el [CHUNK_CENTRAL]
2. Sea autocontenida y tenga sentido por sí sola
3. No corte ideas a la mitad
4. Empiece y termine en puntos de corte naturales (inicio de párrafo, fin de sección, etc.)

TEXTO COMPLETO:
{extended_text}

QUERY DEL USUARIO (para contexto de relevancia):
{query}

Devuelve JSON con:
{{
  "extracted_text": "El fragmento coherente extraído (texto literal del original)",
  "topic_summary": "Resumen de 1 línea del tema principal del fragmento",
  "relevance_explanation": "Por qué este fragmento es relevante para la query (1-2 oraciones)"
}}

IMPORTANTE:
- extracted_text debe ser texto LITERAL del original, no parafrasear
- Incluye todo el [CHUNK_CENTRAL] más el contexto necesario
- Busca el equilibrio: ni demasiado corto (incompleto) ni demasiado largo (irrelevante)
- Prioriza coherencia sobre brevedad

Responde SOLO con el JSON."""

    def __init__(
        self,
        indices_dir: Path,
        chunks_before: int = 3,
        chunks_after: int = 3,
        max_context_tokens: int = 2000
    ):
        """
        Args:
            indices_dir: Directorio con índices (para cargar chunks.pkl)
            chunks_before: Chunks a incluir antes del central
            chunks_after: Chunks a incluir después del central
            max_context_tokens: Máximo de tokens para el contexto extendido
        """
        self.indices_dir = Path(indices_dir)
        self.chunks_before = chunks_before
        self.chunks_after = chunks_after
        self.max_context_tokens = max_context_tokens
        self._chunks_store: Optional[Dict] = None

    def _load_chunks_store(self):
        """Carga el almacén de chunks si no está cargado."""
        if self._chunks_store is not None:
            return

        chunks_path = self.indices_dir / "chunks.pkl"
        if chunks_path.exists():
            with open(chunks_path, 'rb') as f:
                self._chunks_store = pickle.load(f)
            logger.debug(f"Chunks store cargado: {len(self._chunks_store)} chunks")
        else:
            logger.warning(f"No se encontró chunks.pkl en {self.indices_dir}")
            self._chunks_store = {}

    def _get_adjacent_chunks(
        self,
        chunk_id: str,
        n_before: int,
        n_after: int
    ) -> Tuple[List[Any], Any, List[Any]]:
        """
        Obtiene chunks adyacentes del mismo documento.

        Returns:
            Tuple (chunks_before, central_chunk, chunks_after)
        """
        self._load_chunks_store()

        if chunk_id not in self._chunks_store:
            return [], None, []

        central = self._chunks_store[chunk_id]
        doc_id = central.doc_id

        # Obtener todos los chunks del mismo documento ordenados por chunk_id
        # (chunk_ids tienen formato doc_id_level_000001)
        doc_chunks = [
            c for c in self._chunks_store.values()
            if c.doc_id == doc_id and c.level.value == "micro"
        ]

        # Ordenar por chunk_id (asume orden lexicográfico = orden en documento)
        doc_chunks.sort(key=lambda x: x.chunk_id)

        # Encontrar índice del chunk central
        try:
            central_idx = next(
                i for i, c in enumerate(doc_chunks)
                if c.chunk_id == chunk_id
            )
        except StopIteration:
            return [], central, []

        # Obtener chunks antes y después
        start_idx = max(0, central_idx - n_before)
        end_idx = min(len(doc_chunks), central_idx + n_after + 1)

        before = doc_chunks[start_idx:central_idx]
        after = doc_chunks[central_idx + 1:end_idx]

        return before, central, after

    def _count_tokens(self, text: str) -> int:
        """Estima tokens en un texto."""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            return len(text) // 3

    def expand_chunk(
        self,
        chunk_id: str,
        query: str,
        source_citation: str = ""
    ) -> Optional[ExpandedContext]:
        """
        Expande un chunk a un fragmento coherente usando LLM.

        Args:
            chunk_id: ID del chunk a expandir
            query: Query del usuario (para contexto de relevancia)
            source_citation: Cita formateada (ej: "Nielsen, Cap. 5")

        Returns:
            ExpandedContext con el fragmento coherente
        """
        from src.llm_provider import complete as llm_complete

        before, central, after = self._get_adjacent_chunks(
            chunk_id, self.chunks_before, self.chunks_after
        )

        if central is None:
            logger.warning(f"Chunk no encontrado: {chunk_id}")
            return None

        # Construir texto extendido marcando el chunk central
        parts = []
        start_chunk_id = before[0].chunk_id if before else central.chunk_id
        end_chunk_id = after[-1].chunk_id if after else central.chunk_id

        for c in before:
            parts.append(c.content)

        parts.append(f"\n[CHUNK_CENTRAL]\n{central.content}\n[/CHUNK_CENTRAL]\n")

        for c in after:
            parts.append(c.content)

        extended_text = "\n\n".join(parts)

        # Truncar si es muy largo
        if self._count_tokens(extended_text) > self.max_context_tokens:
            # Truncar manteniendo el chunk central en el medio
            extended_text = extended_text[:self.max_context_tokens * 4]  # ~4 chars/token

        # Llamar al LLM
        prompt = self.EXPANSION_PROMPT.format(
            extended_text=extended_text,
            query=query
        )

        try:
            response = llm_complete(
                prompt=prompt,
                system="Eres un asistente que extrae fragmentos coherentes de textos académicos. Responde solo JSON válido.",
                temperature=0,
                max_tokens=self.max_context_tokens + 500,
                json_mode=True
            )

            data = json.loads(response.content)

            extracted = data.get("extracted_text", central.content)
            topic = data.get("topic_summary", central.topic_summary or "")
            relevance = data.get("relevance_explanation", "")

            # Si la cita no viene, construirla desde el chunk
            if not source_citation:
                source_citation = central.format_citation() if hasattr(central, 'format_citation') else f"{central.doc_title} — {central.header_path}"

            return ExpandedContext(
                chunk_id=chunk_id,
                original_content=central.content,
                expanded_content=extracted,
                source_citation=source_citation,
                topic_summary=topic,
                relevance_to_query=relevance,
                start_chunk_id=start_chunk_id,
                end_chunk_id=end_chunk_id,
                token_count=self._count_tokens(extracted)
            )

        except Exception as e:
            logger.warning(f"Error expandiendo chunk {chunk_id}: {e}")
            # Fallback: devolver el chunk central con contexto mínimo
            return ExpandedContext(
                chunk_id=chunk_id,
                original_content=central.content,
                expanded_content=central.content,
                source_citation=source_citation or f"{central.doc_title}",
                topic_summary=central.topic_summary if hasattr(central, 'topic_summary') else "",
                relevance_to_query="",
                start_chunk_id=chunk_id,
                end_chunk_id=chunk_id,
                token_count=self._count_tokens(central.content)
            )

    def expand_chunks(
        self,
        chunk_ids: List[str],
        query: str,
        source_citations: Optional[Dict[str, str]] = None
    ) -> List[ExpandedContext]:
        """
        Expande múltiples chunks.

        Args:
            chunk_ids: Lista de chunk_ids a expandir
            query: Query del usuario
            source_citations: Opcional dict chunk_id -> citation

        Returns:
            Lista de ExpandedContext
        """
        results = []
        source_citations = source_citations or {}

        for chunk_id in chunk_ids:
            citation = source_citations.get(chunk_id, "")
            expanded = self.expand_chunk(chunk_id, query, citation)
            if expanded:
                results.append(expanded)

        return results

    def expand_retrieval_results(
        self,
        retrieval_results: List[Any],
        query: str
    ) -> List[ExpandedContext]:
        """
        Expande una lista de RetrievalResult.

        Args:
            retrieval_results: Lista de RetrievalResult
            query: Query del usuario

        Returns:
            Lista de ExpandedContext
        """
        results = []

        for result in retrieval_results:
            # Construir citation desde los metadatos del resultado
            citation_parts = [result.doc_title]
            if hasattr(result, 'metadata') and result.metadata:
                hierarchy = result.metadata.get('section_hierarchy', [])
                if hierarchy:
                    citation_parts.append(" > ".join(hierarchy[:2]))
                section_num = result.metadata.get('section_number', '')
                if section_num:
                    citation_parts.append(f"§{section_num}")
            elif result.header_path:
                citation_parts.append(result.header_path)

            citation = " — ".join(citation_parts)

            expanded = self.expand_chunk(result.chunk_id, query, citation)
            if expanded:
                results.append(expanded)

        return results
