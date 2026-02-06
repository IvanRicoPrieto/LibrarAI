"""Tests para PropositionDecomposer (Proposition-based Indexing)."""

import sys
import json
from pathlib import Path
from unittest.mock import patch
from dataclasses import dataclass, field
from typing import List, Optional

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class MockChunk:
    chunk_id: str = "doc1_micro_000001"
    content: str = (
        "El algoritmo de Shor fue propuesto por Peter Shor en 1994. "
        "Permite factorizar números enteros en tiempo polinómico usando "
        "un ordenador cuántico. La clave es el uso de la transformada "
        "cuántica de Fourier para encontrar el período de una función."
    )
    level: str = "micro"
    doc_id: str = "doc1"
    doc_title: str = "Computación Cuántica"
    header_path: str = "Cap 5 > Shor"
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    token_count: int = 50
    content_hash: str = "abc123"


@dataclass
class MockLLMResponse:
    content: str = ""
    ok: bool = True
    model: str = "mock"
    provider: str = "mock"
    tokens_input: int = 100
    tokens_output: int = 50


class TestPropositionDecomposer:

    @patch("src.llm_provider.complete")
    def test_decompose_chunk(self, mock_complete):
        """Test descomposición de un chunk en proposiciones."""
        propositions = [
            "El algoritmo de Shor fue propuesto por Peter Shor en 1994.",
            "El algoritmo de Shor factoriza números enteros en tiempo polinómico.",
            "El algoritmo de Shor requiere un ordenador cuántico.",
            "El algoritmo de Shor utiliza la transformada cuántica de Fourier.",
        ]
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps(propositions)
        )

        from src.ingestion.proposition_decomposer import PropositionDecomposer

        decomposer = PropositionDecomposer()
        chunk = MockChunk()
        result = decomposer.decompose_chunk(chunk)

        assert len(result) == 4
        for prop in result:
            assert prop.parent_chunk_id == chunk.chunk_id
            assert prop.doc_id == chunk.doc_id
            assert prop.doc_title == chunk.doc_title
            assert prop.proposition_id.startswith(chunk.chunk_id)

    @patch("src.llm_provider.complete")
    def test_decompose_chunk_filters_short(self, mock_complete):
        """Test que proposiciones cortas se filtran."""
        propositions = [
            "El algoritmo de Shor fue propuesto por Peter Shor en 1994.",
            "OK",  # Muy corta
            "Sí",  # Muy corta
        ]
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps(propositions)
        )

        from src.ingestion.proposition_decomposer import PropositionDecomposer

        decomposer = PropositionDecomposer(min_proposition_tokens=10)
        result = decomposer.decompose_chunk(MockChunk())

        assert len(result) == 1  # Solo la primera es suficientemente larga

    @patch("src.llm_provider.complete")
    def test_decompose_chunk_dedup(self, mock_complete):
        """Test deduplicación de proposiciones."""
        propositions = [
            "El algoritmo de Shor factoriza números.",
            "El algoritmo de Shor factoriza números.",  # Duplicado
        ]
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps(propositions)
        )

        from src.ingestion.proposition_decomposer import PropositionDecomposer

        decomposer = PropositionDecomposer()
        result = decomposer.decompose_chunk(MockChunk())

        assert len(result) == 1

    @patch("src.llm_provider.complete")
    def test_decompose_chunk_max_limit(self, mock_complete):
        """Test que se respeta max_propositions_per_chunk."""
        propositions = [f"Proposición {i} con contenido suficiente." for i in range(20)]
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps(propositions)
        )

        from src.ingestion.proposition_decomposer import PropositionDecomposer

        decomposer = PropositionDecomposer(max_propositions_per_chunk=5)
        result = decomposer.decompose_chunk(MockChunk())

        assert len(result) <= 5

    @patch("src.llm_provider.complete")
    def test_decompose_chunk_llm_error(self, mock_complete):
        """Test fallback cuando LLM falla."""
        mock_complete.side_effect = Exception("API error")

        from src.ingestion.proposition_decomposer import PropositionDecomposer

        decomposer = PropositionDecomposer()
        chunk = MockChunk()
        result = decomposer.decompose_chunk(chunk)

        # Fallback: chunk completo como proposición
        assert len(result) == 1
        assert result[0].content == chunk.content

    @patch("src.llm_provider.complete")
    def test_decompose_batch(self, mock_complete):
        """Test descomposición por batch."""
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps({
                "0": ["Prop A1 con contenido suficiente.", "Prop A2 con contenido suficiente."],
                "1": ["Prop B1 con contenido suficiente."],
            })
        )

        from src.ingestion.proposition_decomposer import PropositionDecomposer

        decomposer = PropositionDecomposer(batch_size=5)
        chunks = [
            MockChunk(chunk_id="chunk_a", content="Contenido A"),
            MockChunk(chunk_id="chunk_b", content="Contenido B"),
        ]

        result = decomposer.decompose_batch(chunks)

        assert "chunk_a" in result
        assert "chunk_b" in result
        assert len(result["chunk_a"]) == 2
        assert len(result["chunk_b"]) == 1

    def test_proposition_ids_unique(self):
        """Test que los IDs de proposición son únicos."""
        from src.ingestion.proposition_decomposer import Proposition

        props = [
            Proposition(
                proposition_id=f"chunk_micro_001_prop_{i:03d}",
                content=f"Proposición {i}",
                parent_chunk_id="chunk_micro_001",
                doc_id="doc1",
                doc_title="Test",
                header_path="Cap 1",
                token_count=10,
            )
            for i in range(5)
        ]

        ids = [p.proposition_id for p in props]
        assert len(ids) == len(set(ids))

    def test_proposition_content_hash(self):
        """Test que el hash se genera automáticamente."""
        from src.ingestion.proposition_decomposer import Proposition

        p = Proposition(
            proposition_id="test",
            content="El algoritmo de Shor factoriza.",
            parent_chunk_id="chunk_001",
            doc_id="doc1",
            doc_title="Test",
            header_path="Cap 1",
            token_count=10,
        )

        assert p.content_hash != ""
        assert len(p.content_hash) == 16

    def test_reset_dedup_cache(self):
        """Test que reset limpia la caché de dedup."""
        from src.ingestion.proposition_decomposer import PropositionDecomposer

        decomposer = PropositionDecomposer()
        decomposer._seen_hashes.add("test_hash")
        assert len(decomposer._seen_hashes) == 1

        decomposer.reset_dedup_cache()
        assert len(decomposer._seen_hashes) == 0
