"""Tests para ContextExpander (expansión LLM de contexto coherente)."""

import sys
from pathlib import Path
from unittest.mock import patch
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
import pickle
import tempfile
from enum import Enum

import pytest

# Configurar path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Mock de ChunkLevel que se puede serializar con pickle
class MockChunkLevel(Enum):
    """Mock de ChunkLevel que es pickleable."""
    MICRO = "micro"
    MESO = "meso"
    MACRO = "macro"


@dataclass
class MockChunk:
    """Mock de Chunk para tests (pickleable)."""
    chunk_id: str = "test_doc_micro_000001"
    content: str = "El algoritmo de Shor factoriza números en tiempo polinómico."
    level: MockChunkLevel = MockChunkLevel.MICRO
    doc_id: str = "test_doc"
    doc_title: str = "Nielsen & Chuang"
    header_path: str = "Cap 5 > Shor"
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    token_count: int = 30
    content_hash: str = "abc123"
    section_hierarchy: List[str] = field(default_factory=lambda: ["Chapter 5", "5.1 Shor"])
    section_number: str = "5.1"
    topic_summary: str = "Algoritmo de factorización"

    def format_citation(self) -> str:
        parts = [self.doc_title]
        if self.section_hierarchy:
            parts.append(" > ".join(self.section_hierarchy[:2]))
        if self.section_number:
            parts.append(f"§{self.section_number}")
        return " — ".join(parts)


@dataclass
class MockRetrievalResult:
    """Mock de RetrievalResult para tests."""
    chunk_id: str = "test_doc_micro_000001"
    content: str = "Contenido del chunk."
    score: float = 0.85
    doc_id: str = "test_doc"
    doc_title: str = "Nielsen & Chuang"
    header_path: str = "Cap 5 > Shor"
    retriever_type: str = "vector"
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "section_hierarchy": ["Chapter 5", "5.1 Shor"],
        "section_number": "5.1"
    })


@dataclass
class MockLLMResponse:
    """Mock de LLMResponse."""
    content: str = ""
    ok: bool = True
    model: str = "mock"
    provider: str = "mock"
    tokens_input: int = 100
    tokens_output: int = 50


class TestExpandedContext:
    """Tests para ExpandedContext dataclass."""

    def test_expanded_context_dataclass(self):
        """Test creación de ExpandedContext."""
        from src.retrieval.context_expander import ExpandedContext

        ctx = ExpandedContext(
            chunk_id="test_001",
            original_content="Contenido original.",
            expanded_content="Contenido expandido con más contexto.",
            source_citation="Nielsen & Chuang — Chapter 5 — §5.1",
            topic_summary="Algoritmo de Shor",
            relevance_to_query="Explica la factorización cuántica",
            start_chunk_id="test_000",
            end_chunk_id="test_002",
            token_count=100
        )

        assert ctx.chunk_id == "test_001"
        assert "expandido" in ctx.expanded_content
        assert ctx.token_count == 100

    def test_expanded_context_to_dict(self):
        """Test serialización a dict."""
        from src.retrieval.context_expander import ExpandedContext

        ctx = ExpandedContext(
            chunk_id="test_001",
            original_content="Original.",
            expanded_content="Expanded.",
            source_citation="Test Citation",
            topic_summary="Test Topic",
            relevance_to_query="Relevance explanation",
            start_chunk_id="test_000",
            end_chunk_id="test_002",
            token_count=50
        )

        d = ctx.to_dict()
        assert d["chunk_id"] == "test_001"
        assert d["expanded_content"] == "Expanded."
        assert d["source_citation"] == "Test Citation"
        assert d["token_count"] == 50


class TestContextExpander:
    """Tests para ContextExpander."""

    def _create_temp_chunks_store(self, chunks: list) -> Path:
        """Crea un chunks.pkl temporal para tests."""
        temp_dir = Path(tempfile.mkdtemp())
        chunks_dict = {c.chunk_id: c for c in chunks}
        with open(temp_dir / "chunks.pkl", "wb") as f:
            pickle.dump(chunks_dict, f)
        return temp_dir

    def _make_chunks(self, n: int = 5, doc_id: str = "test_doc") -> list:
        """Genera n chunks consecutivos de prueba."""
        chunks = []
        for i in range(n):
            chunks.append(MockChunk(
                chunk_id=f"{doc_id}_micro_{i:06d}",
                content=f"Párrafo {i}: Contenido sobre el tema {i}. " * 3,
                level=MockChunkLevel.MICRO,
                doc_id=doc_id,
                header_path=f"Cap 5 > Sección 5.{i + 1}",
                section_number=f"5.{i + 1}",
            ))
        return chunks

    def test_count_tokens(self):
        """Test estimación de tokens."""
        from src.retrieval.context_expander import ContextExpander

        temp_dir = Path(tempfile.mkdtemp())
        expander = ContextExpander(indices_dir=temp_dir)

        # Tiktoken es eficiente con caracteres repetidos
        # 300 'a' puede resultar en ~38 tokens con tiktoken
        text = "a" * 300
        tokens = expander._count_tokens(text)
        assert 20 <= tokens <= 150  # Rango amplio para diferentes tokenizers

    def test_get_adjacent_chunks(self):
        """Test obtención de chunks adyacentes."""
        chunks = self._make_chunks(7)
        temp_dir = self._create_temp_chunks_store(chunks)

        from src.retrieval.context_expander import ContextExpander

        expander = ContextExpander(
            indices_dir=temp_dir,
            chunks_before=2,
            chunks_after=2
        )

        central_id = "test_doc_micro_000003"  # Chunk en el medio
        before, central, after = expander._get_adjacent_chunks(central_id, 2, 2)

        assert central.chunk_id == central_id
        assert len(before) == 2
        assert len(after) == 2
        assert before[0].chunk_id == "test_doc_micro_000001"
        assert after[-1].chunk_id == "test_doc_micro_000005"

    def test_get_adjacent_chunks_at_start(self):
        """Test chunks adyacentes cuando el central está al principio."""
        chunks = self._make_chunks(5)
        temp_dir = self._create_temp_chunks_store(chunks)

        from src.retrieval.context_expander import ContextExpander

        expander = ContextExpander(indices_dir=temp_dir)

        before, central, after = expander._get_adjacent_chunks(
            "test_doc_micro_000000", 3, 3
        )

        assert len(before) == 0  # No hay chunks antes del primero
        assert len(after) == 3

    def test_get_adjacent_chunks_at_end(self):
        """Test chunks adyacentes cuando el central está al final."""
        chunks = self._make_chunks(5)
        temp_dir = self._create_temp_chunks_store(chunks)

        from src.retrieval.context_expander import ContextExpander

        expander = ContextExpander(indices_dir=temp_dir)

        before, central, after = expander._get_adjacent_chunks(
            "test_doc_micro_000004", 3, 3
        )

        assert len(before) == 3
        assert len(after) == 0  # No hay chunks después del último

    @patch("src.llm_provider.complete")
    def test_expand_chunk(self, mock_complete):
        """Test expansión de un chunk individual."""
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps({
                "extracted_text": "Este es el texto expandido coherente que incluye el contexto necesario.",
                "topic_summary": "Factorización cuántica usando Shor",
                "relevance_explanation": "Explica cómo funciona el período cuántico."
            })
        )

        chunks = self._make_chunks(5)
        temp_dir = self._create_temp_chunks_store(chunks)

        from src.retrieval.context_expander import ContextExpander

        expander = ContextExpander(indices_dir=temp_dir)
        result = expander.expand_chunk(
            "test_doc_micro_000002",
            "¿Cómo funciona el algoritmo de Shor?"
        )

        assert result is not None
        assert result.chunk_id == "test_doc_micro_000002"
        assert "expandido" in result.expanded_content or "texto" in result.expanded_content.lower()
        assert result.topic_summary != ""

    @patch("src.llm_provider.complete")
    def test_expand_chunk_fallback(self, mock_complete):
        """Test fallback cuando LLM falla."""
        mock_complete.side_effect = Exception("API Error")

        chunks = self._make_chunks(3)
        temp_dir = self._create_temp_chunks_store(chunks)

        from src.retrieval.context_expander import ContextExpander

        expander = ContextExpander(indices_dir=temp_dir)
        result = expander.expand_chunk(
            "test_doc_micro_000001",
            "query test"
        )

        # Debe usar fallback con contenido original
        assert result is not None
        assert result.expanded_content == chunks[1].content

    def test_expand_chunk_not_found(self):
        """Test cuando el chunk no existe."""
        temp_dir = self._create_temp_chunks_store([])

        from src.retrieval.context_expander import ContextExpander

        expander = ContextExpander(indices_dir=temp_dir)
        result = expander.expand_chunk("nonexistent_chunk", "query")

        assert result is None

    @patch("src.llm_provider.complete")
    def test_expand_chunks(self, mock_complete):
        """Test expansión de múltiples chunks."""
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps({
                "extracted_text": "Texto expandido.",
                "topic_summary": "Topic",
                "relevance_explanation": "Relevancia"
            })
        )

        chunks = self._make_chunks(5)
        temp_dir = self._create_temp_chunks_store(chunks)

        from src.retrieval.context_expander import ContextExpander

        expander = ContextExpander(indices_dir=temp_dir)
        results = expander.expand_chunks(
            ["test_doc_micro_000001", "test_doc_micro_000003"],
            "query test"
        )

        assert len(results) == 2

    @patch("src.llm_provider.complete")
    def test_expand_retrieval_results(self, mock_complete):
        """Test expansión de RetrievalResults."""
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps({
                "extracted_text": "Texto expandido desde retrieval result.",
                "topic_summary": "Topic del resultado",
                "relevance_explanation": "Es relevante porque..."
            })
        )

        chunks = self._make_chunks(5)
        temp_dir = self._create_temp_chunks_store(chunks)

        from src.retrieval.context_expander import ContextExpander

        retrieval_results = [
            MockRetrievalResult(chunk_id="test_doc_micro_000001"),
            MockRetrievalResult(chunk_id="test_doc_micro_000002"),
        ]

        expander = ContextExpander(indices_dir=temp_dir)
        results = expander.expand_retrieval_results(
            retrieval_results,
            "¿Qué es la computación cuántica?"
        )

        assert len(results) == 2
        for r in results:
            assert r.source_citation != ""


class TestExpansionPrompt:
    """Tests para el prompt de expansión."""

    def test_prompt_contains_required_fields(self):
        """Test que el prompt incluye instrucciones necesarias."""
        from src.retrieval.context_expander import ContextExpander

        prompt = ContextExpander.EXPANSION_PROMPT

        assert "CHUNK_CENTRAL" in prompt
        assert "extracted_text" in prompt
        assert "topic_summary" in prompt
        assert "relevance_explanation" in prompt
        assert "JSON" in prompt


class TestCitationFormatting:
    """Tests para formateo de citas en contextos expandidos."""

    @patch("src.llm_provider.complete")
    def test_citation_includes_section_hierarchy(self, mock_complete):
        """Test que la cita incluye jerarquía de secciones si está disponible."""
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps({
                "extracted_text": "Texto.",
                "topic_summary": "Topic",
                "relevance_explanation": "Relevancia"
            })
        )

        chunk = MockChunk(
            chunk_id="test_micro_000001",
            level=MockChunkLevel.MICRO,
            section_hierarchy=["Chapter 5", "5.2 QFT"],
            section_number="5.2.1"
        )
        chunks = [chunk]

        temp_dir = Path(tempfile.mkdtemp())
        with open(temp_dir / "chunks.pkl", "wb") as f:
            pickle.dump({c.chunk_id: c for c in chunks}, f)

        from src.retrieval.context_expander import ContextExpander

        expander = ContextExpander(indices_dir=temp_dir)
        result = expander.expand_chunk("test_micro_000001", "query")

        # La cita debe contener información de la jerarquía
        assert result is not None
        assert "5.2" in result.source_citation or "Chapter" in result.source_citation
