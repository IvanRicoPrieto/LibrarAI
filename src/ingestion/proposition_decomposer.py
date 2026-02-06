"""
Proposition-based Indexing — Descompone chunks en proposiciones atómicas.

Cada chunk se descompone en claims/hechos individuales auto-contenidos.
Cada proposición se embebe por separado en una colección Qdrant dedicada,
manteniendo referencia al chunk padre para poder expandir contexto.

Una query como "quién inventó el algoritmo de Shor" matcheará directamente
con la proposición "Peter Shor propuso el algoritmo de factorización
cuántica en 1994" con un score muy alto, en vez de matchear parcialmente
con un chunk largo que menciona muchas cosas.
"""

import json
import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class Proposition:
    """Proposición atómica auto-contenida extraída de un chunk."""
    proposition_id: str
    content: str
    parent_chunk_id: str
    doc_id: str
    doc_title: str
    header_path: str
    token_count: int
    content_hash: str = field(default="")

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.content.encode()
            ).hexdigest()[:16]


class PropositionDecomposer:
    """
    Descompone chunks en proposiciones atómicas usando LLM.

    Cada proposición es un hecho individual, auto-contenido (no usa
    pronombres ni referencias implícitas), y verificable.
    """

    def __init__(
        self,
        min_proposition_tokens: int = 10,
        max_propositions_per_chunk: int = 15,
        batch_size: int = 5,
    ):
        """
        Args:
            min_proposition_tokens: Mínimo de tokens para considerar válida.
            max_propositions_per_chunk: Máximo de proposiciones por chunk.
            batch_size: Chunks por llamada LLM.
        """
        self.min_proposition_tokens = min_proposition_tokens
        self.max_propositions_per_chunk = max_propositions_per_chunk
        self.batch_size = batch_size
        self._seen_hashes: set = set()

    def reset_dedup_cache(self):
        """Limpia la caché de deduplicación."""
        self._seen_hashes.clear()

    # ------------------------------------------------------------------
    # Descomposición de un chunk individual
    # ------------------------------------------------------------------

    def decompose_chunk(self, chunk: Chunk) -> List[Proposition]:
        """Descompone un chunk en proposiciones atómicas."""
        try:
            from src.llm_provider import complete as llm_complete

            response = llm_complete(
                prompt=(
                    f"Documento: {chunk.doc_title}\n"
                    f"Sección: {chunk.header_path}\n\n"
                    f"Texto:\n{chunk.content}\n\n"
                    "Descompón este texto en proposiciones atómicas. Cada proposición debe:\n"
                    "- Ser un hecho individual y auto-contenido\n"
                    "- No usar pronombres ni referencias implícitas\n"
                    "- Incluir el contexto necesario para entenderse sin el texto original\n"
                    "- Ser verificable\n\n"
                    "Responde con un JSON array de strings."
                ),
                system=(
                    "Eres un experto en descomponer textos académicos en proposiciones "
                    "atómicas. Cada proposición debe ser un hecho auto-contenido que no "
                    "requiere contexto externo para entenderse. Responde SOLO con un "
                    "JSON array de strings."
                ),
                temperature=0,
                max_tokens=1500,
                json_mode=True,
            )

            raw_propositions = self._parse_propositions(response.content)
            return self._post_process(raw_propositions, chunk)

        except Exception as e:
            logger.warning(
                f"Error descomponiendo chunk {chunk.chunk_id}: {e}. "
                f"Usando chunk completo como proposición."
            )
            return self._fallback_decompose(chunk)

    # ------------------------------------------------------------------
    # Descomposición por batch
    # ------------------------------------------------------------------

    def decompose_batch(
        self, chunks: List[Chunk]
    ) -> Dict[str, List[Proposition]]:
        """
        Descompone múltiples chunks. Retorna dict chunk_id -> proposiciones.
        """
        results: Dict[str, List[Proposition]] = {}

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]

            if len(batch) == 1:
                props = self.decompose_chunk(batch[0])
                results[batch[0].chunk_id] = props
            else:
                batch_results = self._decompose_multi_batch(batch)
                results.update(batch_results)

        return results

    def _decompose_multi_batch(
        self, chunks: List[Chunk]
    ) -> Dict[str, List[Proposition]]:
        """Procesa varios chunks en una sola llamada LLM."""
        chunks_text = ""
        for idx, chunk in enumerate(chunks):
            chunks_text += (
                f'\n--- Chunk {idx} (id: {chunk.chunk_id}) ---\n'
                f"Sección: {chunk.header_path}\n"
                f"Texto: {chunk.content[:800]}\n"
            )

        try:
            from src.llm_provider import complete as llm_complete

            response = llm_complete(
                prompt=(
                    f"Documento: {chunks[0].doc_title}\n\n"
                    f"Chunks:{chunks_text}\n\n"
                    "Para CADA chunk, descompón el texto en proposiciones atómicas "
                    "auto-contenidas. Responde con un JSON object donde las claves "
                    "son los índices (0, 1, 2...) y los valores son arrays de strings.\n"
                    'Ejemplo: {"0": ["prop1", "prop2"], "1": ["prop3"]}'
                ),
                system=(
                    "Descompones textos académicos en proposiciones atómicas. "
                    "Cada proposición es un hecho auto-contenido, sin pronombres. "
                    "Responde SOLO con JSON."
                ),
                temperature=0,
                max_tokens=1500 * len(chunks),
                json_mode=True,
            )

            parsed = json.loads(response.content)
            results: Dict[str, List[Proposition]] = {}

            for idx, chunk in enumerate(chunks):
                raw = parsed.get(str(idx), parsed.get(idx, []))
                if isinstance(raw, list):
                    props = self._post_process(
                        [str(p) for p in raw], chunk
                    )
                else:
                    props = self._fallback_decompose(chunk)
                results[chunk.chunk_id] = props

            return results

        except Exception as e:
            logger.warning(f"Error en batch de proposiciones: {e}")
            return {
                c.chunk_id: self._fallback_decompose(c) for c in chunks
            }

    # ------------------------------------------------------------------
    # Post-procesado
    # ------------------------------------------------------------------

    def _post_process(
        self, raw_propositions: List[str], chunk: Chunk
    ) -> List[Proposition]:
        """Filtra, dedup y genera IDs para proposiciones."""
        propositions: List[Proposition] = []
        counter = 0

        for raw in raw_propositions:
            raw = raw.strip()
            if not raw:
                continue

            # Filtrar muy cortas
            token_est = len(raw) // 3
            if token_est < self.min_proposition_tokens:
                continue

            # Deduplicar por hash
            content_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
            if content_hash in self._seen_hashes:
                continue
            self._seen_hashes.add(content_hash)

            counter += 1
            prop_id = f"{chunk.chunk_id}_prop_{counter:03d}"

            propositions.append(
                Proposition(
                    proposition_id=prop_id,
                    content=raw,
                    parent_chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    header_path=chunk.header_path,
                    token_count=token_est,
                    content_hash=content_hash,
                )
            )

            if len(propositions) >= self.max_propositions_per_chunk:
                break

        return propositions

    def _fallback_decompose(self, chunk: Chunk) -> List[Proposition]:
        """Fallback: usa el chunk completo como una sola proposición."""
        content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()[:16]
        if content_hash in self._seen_hashes:
            return []
        self._seen_hashes.add(content_hash)

        return [
            Proposition(
                proposition_id=f"{chunk.chunk_id}_prop_001",
                content=chunk.content,
                parent_chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                header_path=chunk.header_path,
                token_count=chunk.token_count,
                content_hash=content_hash,
            )
        ]

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_propositions(raw: str) -> List[str]:
        """Parsea la respuesta LLM como JSON array de strings."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
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
        if isinstance(parsed, dict):
            for val in parsed.values():
                if isinstance(val, list):
                    return [str(p) for p in val]
        return []
