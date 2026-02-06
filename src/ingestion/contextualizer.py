"""
Contextual Retrieval — Añade contexto documental a chunks antes de embedding.

La técnica consiste en prepender a cada chunk un breve prefijo de contexto
generado por LLM (título, sección, resumen) antes de calcular su embedding.
Esto mejora la calidad del retrieval especialmente para chunks que usan
referencias implícitas o pronombres.

El contexto se aplica SOLO al texto usado para embedding.
El contenido original del chunk no cambia para display/BM25.

Referencia: Anthropic — "Contextual Retrieval" (2024)
"""

import json
import hashlib
import logging
from dataclasses import dataclass
from typing import List, Dict

from .chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class ChunkContext:
    """Contexto generado para un chunk."""
    chunk_id: str
    context_prefix: str
    embedding_text: str  # context_prefix + chunk.content
    token_count: int


class ChunkContextualizer:
    """
    Genera prefijos de contexto para chunks usando LLM.

    Flujo:
    1. Generar un resumen del documento (una vez por documento, cacheado).
    2. Para cada batch de chunks, pedir al LLM un prefijo de contexto
       de 1 frase que sitúe el chunk dentro del documento.
    3. Concatenar prefijo + contenido original → texto para embedding.
    """

    def __init__(
        self,
        max_context_tokens: int = 100,
        batch_size: int = 20,
    ):
        """
        Args:
            max_context_tokens: Máximo de tokens para el prefijo de contexto.
            batch_size: Chunks por llamada LLM en contextualize_batch.
        """
        self.max_context_tokens = max_context_tokens
        self.batch_size = batch_size
        self._doc_summary_cache: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Resumen de documento (1 llamada por documento, cacheado)
    # ------------------------------------------------------------------

    def generate_document_summary(
        self,
        doc_title: str,
        sections_text: str,
    ) -> str:
        """Genera un resumen breve del documento. Cachea por contenido."""
        cache_key = hashlib.sha256(
            f"{doc_title}:{sections_text[:500]}".encode()
        ).hexdigest()[:16]

        if cache_key in self._doc_summary_cache:
            return self._doc_summary_cache[cache_key]

        try:
            from src.llm_provider import complete as llm_complete

            response = llm_complete(
                prompt=(
                    f'Documento: "{doc_title}"\n\n'
                    f"Secciones (extracto):\n{sections_text[:2000]}\n\n"
                    "Genera un resumen de 2-3 frases que describa de qué trata "
                    "este documento, su enfoque y los temas principales que cubre."
                ),
                system=(
                    "Eres un asistente que genera resúmenes concisos de documentos "
                    "académicos. Responde SOLO con el resumen, sin preámbulos."
                ),
                temperature=0,
                max_tokens=200,
            )
            summary = response.content.strip()

        except Exception as e:
            logger.warning(f"Error generando resumen para '{doc_title}': {e}")
            summary = f"Documento: {doc_title}"

        self._doc_summary_cache[cache_key] = summary
        logger.info(f"Resumen generado para '{doc_title}' ({len(summary)} chars)")
        return summary

    # ------------------------------------------------------------------
    # Contextualización por batch
    # ------------------------------------------------------------------

    def contextualize_batch(
        self,
        chunks: List[Chunk],
        doc_summary: str,
    ) -> List[ChunkContext]:
        """Genera contextos para una lista de chunks (sub-batcheado)."""
        if not chunks:
            return []

        results: List[ChunkContext] = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batch_contexts = self._contextualize_single_batch(batch, doc_summary)
            results.extend(batch_contexts)
        return results

    def _contextualize_single_batch(
        self,
        chunks: List[Chunk],
        doc_summary: str,
    ) -> List[ChunkContext]:
        """Procesa un sub-batch con una sola llamada LLM."""
        # Construir prompt con los chunks
        chunks_text = ""
        for idx, chunk in enumerate(chunks):
            chunks_text += (
                f"\n--- Chunk {idx} ---\n"
                f"Sección: {chunk.header_path}\n"
                f"Contenido: {chunk.content[:500]}\n"
            )

        try:
            from src.llm_provider import complete as llm_complete

            response = llm_complete(
                prompt=(
                    f"Documento: {chunks[0].doc_title}\n"
                    f"Resumen del documento: {doc_summary}\n\n"
                    f"Chunks a contextualizar:{chunks_text}\n\n"
                    f"Para cada chunk, genera un prefijo de contexto de 1 frase "
                    f"que sitúe el fragmento en el documento. El prefijo debe "
                    f"mencionar el documento, la sección y el tema específico.\n\n"
                    f"Responde con un JSON array de strings, uno por chunk, "
                    f"en el mismo orden. Ejemplo: "
                    f'["Este fragmento del libro X, capítulo Y, trata sobre Z.", ...]'
                ),
                system=(
                    "Generas prefijos de contexto concisos para fragmentos de "
                    "documentos académicos. Cada prefijo debe ser 1 frase que "
                    "sitúe el fragmento en su contexto documental. "
                    "Responde SOLO con un JSON array de strings."
                ),
                temperature=0,
                max_tokens=100 * len(chunks),
                json_mode=True,
            )

            prefixes = self._parse_prefixes(response.content, len(chunks))

        except Exception as e:
            logger.warning(f"Error contextualizando batch: {e}. Usando fallback.")
            return self._fallback_contexts(chunks)

        # Asegurar misma longitud
        default_prefix = (
            f"Este fragmento pertenece al documento '{chunks[0].doc_title}'."
        )
        while len(prefixes) < len(chunks):
            prefixes.append(default_prefix)

        results: List[ChunkContext] = []
        for chunk, prefix in zip(chunks, prefixes):
            prefix_str = str(prefix).strip()
            embedding_text = self.build_embedding_text(chunk, prefix_str)
            results.append(
                ChunkContext(
                    chunk_id=chunk.chunk_id,
                    context_prefix=prefix_str,
                    embedding_text=embedding_text,
                    token_count=len(embedding_text) // 3,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_prefixes(raw: str, expected: int) -> List[str]:
        """Parsea la respuesta LLM (JSON array de strings)."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Intentar extraer array del texto
            import re

            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return []
            else:
                return []

        if isinstance(parsed, list):
            return [str(p) for p in parsed]

        # Si el LLM devuelve un dict con una clave, extraer el array
        if isinstance(parsed, dict):
            for val in parsed.values():
                if isinstance(val, list):
                    return [str(p) for p in val]

        return []

    def _fallback_contexts(self, chunks: List[Chunk]) -> List[ChunkContext]:
        """Genera contextos determinísticos sin LLM (fallback)."""
        results: List[ChunkContext] = []
        for chunk in chunks:
            prefix = (
                f"Este fragmento pertenece al documento '{chunk.doc_title}', "
                f"sección '{chunk.header_path}'."
            )
            embedding_text = self.build_embedding_text(chunk, prefix)
            results.append(
                ChunkContext(
                    chunk_id=chunk.chunk_id,
                    context_prefix=prefix,
                    embedding_text=embedding_text,
                    token_count=len(embedding_text) // 3,
                )
            )
        return results

    @staticmethod
    def build_embedding_text(chunk: Chunk, context_prefix: str) -> str:
        """Concatena prefijo de contexto + contenido original para embedding."""
        if not context_prefix:
            return chunk.content
        return f"{context_prefix}\n\n{chunk.content}"
