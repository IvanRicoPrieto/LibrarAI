"""Tests para ChunkContextualizer (Contextual Retrieval)."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
from typing import List, Optional

import pytest

# Configurar path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class MockChunk:
    """Mock de Chunk para tests."""
    chunk_id: str = "test_doc_micro_000001"
    content: str = "El algoritmo de Shor factoriza números en tiempo polinómico."
    level: str = "micro"
    doc_id: str = "test_doc"
    doc_title: str = "Computación Cuántica - Nielsen & Chuang"
    header_path: str = "Cap 5 > Algoritmos > Shor"
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    token_count: int = 30
    content_hash: str = "abc123"


@dataclass
class MockLLMResponse:
    """Mock de LLMResponse."""
    content: str = ""
    ok: bool = True
    model: str = "mock"
    provider: str = "mock"
    tokens_input: int = 100
    tokens_output: int = 50


class TestChunkContextualizer:
    """Tests para ChunkContextualizer."""

    def _make_chunks(self, n: int = 3) -> list:
        """Genera n chunks de prueba."""
        chunks = []
        for i in range(n):
            chunks.append(MockChunk(
                chunk_id=f"test_doc_micro_{i:06d}",
                content=f"Contenido del chunk {i} sobre computación cuántica.",
                header_path=f"Cap {i + 1} > Sección {i + 1}",
            ))
        return chunks

    @patch("src.llm_provider.complete")
    def test_generate_document_summary(self, mock_complete):
        """Test generación de resumen de documento."""
        mock_complete.return_value = MockLLMResponse(
            content="Este documento cubre los fundamentos de la computación cuántica."
        )

        from src.ingestion.contextualizer import ChunkContextualizer

        ctx = ChunkContextualizer()
        summary = ctx.generate_document_summary(
            "Nielsen & Chuang",
            "Cap 1: Fundamentos\nCap 2: Qubits\nCap 3: Circuitos"
        )

        assert "computación cuántica" in summary.lower()
        assert mock_complete.called

    @patch("src.llm_provider.complete")
    def test_summary_caching(self, mock_complete):
        """Test que el resumen se cachea por documento."""
        mock_complete.return_value = MockLLMResponse(
            content="Resumen del documento."
        )

        from src.ingestion.contextualizer import ChunkContextualizer

        ctx = ChunkContextualizer()
        s1 = ctx.generate_document_summary("Doc A", "Contenido A")
        s2 = ctx.generate_document_summary("Doc A", "Contenido A")

        assert s1 == s2
        assert mock_complete.call_count == 1  # Solo una llamada

    @patch("src.llm_provider.complete")
    def test_contextualize_batch(self, mock_complete):
        """Test contextualización de batch de chunks."""
        prefixes = [
            "Este fragmento del libro de Nielsen, cap 1, trata sobre qubits.",
            "Este fragmento del libro de Nielsen, cap 2, trata sobre gates.",
            "Este fragmento del libro de Nielsen, cap 3, trata sobre circuitos.",
        ]
        mock_complete.return_value = MockLLMResponse(
            content=f'{prefixes}'
        )

        from src.ingestion.contextualizer import ChunkContextualizer

        ctx = ChunkContextualizer(batch_size=10)
        chunks = self._make_chunks(3)

        import json
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps(prefixes)
        )

        results = ctx.contextualize_batch(chunks, "Resumen del doc")

        assert len(results) == 3
        for r in results:
            assert r.context_prefix != ""
            assert r.embedding_text.startswith(r.context_prefix)
            assert chunks[0].content in results[0].embedding_text or \
                   chunks[0].content[:50] in results[0].embedding_text

    @patch("src.llm_provider.complete")
    def test_contextualize_batch_llm_error(self, mock_complete):
        """Test fallback cuando LLM falla."""
        mock_complete.side_effect = Exception("API error")

        from src.ingestion.contextualizer import ChunkContextualizer

        ctx = ChunkContextualizer()
        chunks = self._make_chunks(2)

        results = ctx.contextualize_batch(chunks, "Resumen")

        assert len(results) == 2
        # Debe usar fallback con doc_title y header_path
        for r in results:
            assert r.context_prefix != ""
            assert "Nielsen" in r.context_prefix

    def test_build_embedding_text(self):
        """Test concatenación de prefijo + contenido."""
        from src.ingestion.contextualizer import ChunkContextualizer

        ctx = ChunkContextualizer()
        chunk = MockChunk()

        text = ctx.build_embedding_text(chunk, "Prefijo de contexto.")
        assert text.startswith("Prefijo de contexto.")
        assert chunk.content in text

    def test_build_embedding_text_empty_prefix(self):
        """Test con prefijo vacío retorna contenido original."""
        from src.ingestion.contextualizer import ChunkContextualizer

        ctx = ChunkContextualizer()
        chunk = MockChunk()

        text = ctx.build_embedding_text(chunk, "")
        assert text == chunk.content

    @patch("src.llm_provider.complete")
    def test_sub_batching(self, mock_complete):
        """Test que se subdivide correctamente en sub-batches."""
        import json

        from src.ingestion.contextualizer import ChunkContextualizer

        ctx = ChunkContextualizer(batch_size=2)
        chunks = self._make_chunks(5)

        # Configurar mock para retornar prefijos correctos cada vez
        def mock_response(*args, **kwargs):
            # Contar chunks en el prompt
            prompt = kwargs.get("prompt", args[0] if args else "")
            n = prompt.count("--- Chunk")
            return MockLLMResponse(
                content=json.dumps([f"Prefijo {i}" for i in range(n)])
            )

        mock_complete.side_effect = mock_response

        results = ctx.contextualize_batch(chunks, "Resumen")

        assert len(results) == 5
        # Debe haber hecho 3 llamadas (2+2+1)
        assert mock_complete.call_count == 3

    def test_parse_prefixes_valid_json(self):
        """Test parsing de JSON válido."""
        from src.ingestion.contextualizer import ChunkContextualizer

        result = ChunkContextualizer._parse_prefixes(
            '["prefijo 1", "prefijo 2"]', 2
        )
        assert result == ["prefijo 1", "prefijo 2"]

    def test_parse_prefixes_wrapped_in_object(self):
        """Test parsing cuando LLM devuelve un objeto con array."""
        from src.ingestion.contextualizer import ChunkContextualizer

        result = ChunkContextualizer._parse_prefixes(
            '{"prefixes": ["p1", "p2"]}', 2
        )
        assert result == ["p1", "p2"]

    def test_parse_prefixes_invalid_json(self):
        """Test fallback con JSON inválido."""
        from src.ingestion.contextualizer import ChunkContextualizer

        result = ChunkContextualizer._parse_prefixes("not json", 2)
        assert result == []

    def test_chunk_context_dataclass(self):
        """Test del dataclass ChunkContext."""
        from src.ingestion.contextualizer import ChunkContext

        ctx = ChunkContext(
            chunk_id="test_001",
            context_prefix="Prefijo.",
            embedding_text="Prefijo.\n\nContenido del chunk.",
            token_count=20,
        )
        assert ctx.chunk_id == "test_001"
        assert ctx.token_count == 20
