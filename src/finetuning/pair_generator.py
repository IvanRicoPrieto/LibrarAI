"""
Synthetic Pair Generator — Genera pares de entrenamiento para fine-tuning.

A partir de los chunks indexados, usa un LLM para generar preguntas
sintéticas que serían respondidas por cada chunk. Genera también
hard negatives usando BM25 para crear ejemplos de entrenamiento
más discriminativos.
"""

import json
import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingPair:
    """Par de entrenamiento (query, passage_relevante)."""
    query: str
    positive_passage: str
    chunk_id: str
    doc_id: str
    doc_title: str
    difficulty: str  # "easy", "medium", "hard"
    query_type: str  # "factual", "conceptual", "comparative"
    query_hash: str = field(default="")

    def __post_init__(self):
        if not self.query_hash:
            self.query_hash = hashlib.sha256(
                self.query.lower().strip().encode()
            ).hexdigest()[:12]


class SyntheticPairGenerator:
    """
    Genera pares sintéticos de entrenamiento usando LLM.

    Para cada chunk genera 3 tipos de preguntas:
    - Factual: pregunta directa sobre un hecho del chunk
    - Conceptual: pregunta que requiere comprensión
    - Variada: pregunta desde otra perspectiva o con sinónimos
    """

    def __init__(
        self,
        queries_per_chunk: int = 3,
        batch_size: int = 5,
        min_query_length: int = 15,
    ):
        """
        Args:
            queries_per_chunk: Preguntas a generar por chunk.
            batch_size: Chunks por llamada LLM.
            min_query_length: Largo mínimo de query válida.
        """
        self.queries_per_chunk = queries_per_chunk
        self.batch_size = batch_size
        self.min_query_length = min_query_length
        self._seen_hashes: set = set()

    # ------------------------------------------------------------------
    # Generación principal
    # ------------------------------------------------------------------

    def generate_from_chunks(self, chunks: list) -> List[TrainingPair]:
        """
        Genera pares de entrenamiento a partir de chunks.

        Args:
            chunks: Lista de Chunk objects.

        Returns:
            Lista de TrainingPair.
        """
        all_pairs: List[TrainingPair] = []

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batch_pairs = self._generate_batch(batch)
            all_pairs.extend(batch_pairs)

        logger.info(f"Generados {len(all_pairs)} pares de entrenamiento")
        return all_pairs

    def _generate_batch(self, chunks: list) -> List[TrainingPair]:
        """Genera pares para un batch de chunks."""
        chunks_text = ""
        for idx, chunk in enumerate(chunks):
            title = getattr(chunk, "doc_title", "Documento")
            section = getattr(chunk, "header_path", "")
            content = getattr(chunk, "content", str(chunk))[:600]
            chunks_text += (
                f"\n--- Chunk {idx} ---\n"
                f"Documento: {title}\n"
                f"Sección: {section}\n"
                f"Contenido: {content}\n"
            )

        try:
            from src.llm_provider import complete as llm_complete

            response = llm_complete(
                prompt=(
                    f"Chunks:{chunks_text}\n\n"
                    f"Para CADA chunk, genera {self.queries_per_chunk} preguntas "
                    f"que serían respondidas por ese chunk:\n"
                    f"1. Una pregunta factual directa\n"
                    f"2. Una pregunta conceptual que requiera comprensión\n"
                    f"3. Una pregunta con sinónimos o perspectiva diferente\n\n"
                    f"Responde con JSON:\n"
                    '{"chunks": [{"index": 0, "queries": ['
                    '{"query": "...", "type": "factual", "difficulty": "easy"}'
                    "]}]}"
                ),
                system=(
                    "Generas preguntas de búsqueda realistas para entrenar "
                    "un modelo de embeddings. Las preguntas deben ser naturales, "
                    "variadas y relevantes para el contenido. "
                    "Responde SOLO con JSON."
                ),
                temperature=0.7,
                max_tokens=300 * len(chunks),
                json_mode=True,
            )

            return self._parse_pairs(response.content, chunks)

        except Exception as e:
            logger.warning(f"Error generando pares: {e}")
            return []

    # ------------------------------------------------------------------
    # Hard negatives
    # ------------------------------------------------------------------

    def generate_hard_negatives(
        self,
        pairs: List[TrainingPair],
        all_chunks: list,
        negatives_per_pair: int = 3,
    ) -> List[Dict]:
        """
        Genera hard negatives para cada par usando BM25.

        Un hard negative es un chunk que es temáticamente similar a la query
        pero NO contiene la respuesta correcta.

        Returns:
            Lista de dicts con query, positive, negatives.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 no disponible. Sin hard negatives.")
            return [
                {"query": p.query, "positive": p.positive_passage, "negatives": []}
                for p in pairs
            ]

        # Construir índice BM25 temporal
        from ..utils.text_processing import tokenize_for_bm25

        contents = [getattr(c, "content", str(c)) for c in all_chunks]
        chunk_ids = [getattr(c, "chunk_id", str(i)) for i, c in enumerate(all_chunks)]
        tokenized = [tokenize_for_bm25(c) for c in contents]
        bm25 = BM25Okapi(tokenized)

        results = []
        for pair in pairs:
            query_tokens = tokenize_for_bm25(pair.query)
            scores = bm25.get_scores(query_tokens)

            # Ordenar por score descendente
            ranked = sorted(
                enumerate(scores), key=lambda x: x[1], reverse=True
            )

            negatives = []
            for idx, score in ranked:
                if chunk_ids[idx] == pair.chunk_id:
                    continue  # Excluir positivo
                negatives.append(contents[idx])
                if len(negatives) >= negatives_per_pair:
                    break

            results.append({
                "query": pair.query,
                "positive": pair.positive_passage,
                "negatives": negatives,
            })

        logger.info(f"Generados hard negatives para {len(results)} pares")
        return results

    # ------------------------------------------------------------------
    # Deduplicación
    # ------------------------------------------------------------------

    def deduplicate_queries(
        self, pairs: List[TrainingPair]
    ) -> List[TrainingPair]:
        """Elimina queries duplicadas o casi-duplicadas."""
        unique = []
        for pair in pairs:
            if pair.query_hash not in self._seen_hashes:
                self._seen_hashes.add(pair.query_hash)
                unique.append(pair)
        logger.info(
            f"Deduplicación: {len(pairs)} → {len(unique)} pares únicos"
        )
        return unique

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_pairs(
        self, raw: str, chunks: list
    ) -> List[TrainingPair]:
        """Parsea respuesta JSON del LLM."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []

        chunk_data = parsed.get("chunks", [])
        if not isinstance(chunk_data, list):
            # Intentar otras estructuras
            if isinstance(parsed, list):
                chunk_data = parsed
            elif isinstance(parsed, dict):
                for val in parsed.values():
                    if isinstance(val, list):
                        chunk_data = val
                        break

        pairs = []
        for item in chunk_data:
            if not isinstance(item, dict):
                continue

            idx = item.get("index", len(pairs) // self.queries_per_chunk)
            if idx >= len(chunks):
                continue

            chunk = chunks[idx]
            queries = item.get("queries", [])

            for q_item in queries:
                if isinstance(q_item, str):
                    query_text = q_item
                    q_type = "general"
                    difficulty = "medium"
                elif isinstance(q_item, dict):
                    query_text = q_item.get("query", "")
                    q_type = q_item.get("type", "general")
                    difficulty = q_item.get("difficulty", "medium")
                else:
                    continue

                if len(query_text) < self.min_query_length:
                    continue

                pair = TrainingPair(
                    query=query_text,
                    positive_passage=getattr(chunk, "content", str(chunk)),
                    chunk_id=getattr(chunk, "chunk_id", f"chunk_{idx}"),
                    doc_id=getattr(chunk, "doc_id", ""),
                    doc_title=getattr(chunk, "doc_title", ""),
                    difficulty=difficulty,
                    query_type=q_type,
                )
                pairs.append(pair)

        return self.deduplicate_queries(pairs)
