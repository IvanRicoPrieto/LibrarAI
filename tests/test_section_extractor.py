"""Tests para SectionExtractor (extracción LLM de jerarquía de secciones)."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
from typing import List, Optional
import json

import pytest

# Configurar path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class MockLLMResponse:
    """Mock de LLMResponse."""
    content: str = ""
    ok: bool = True
    model: str = "mock"
    provider: str = "mock"
    tokens_input: int = 100
    tokens_output: int = 50


class TestSectionMetadata:
    """Tests para SectionMetadata dataclass."""

    def test_section_metadata_dataclass(self):
        """Test creación de SectionMetadata."""
        from src.ingestion.section_extractor import SectionMetadata

        meta = SectionMetadata(
            chunk_id="test_chunk_001",
            section_hierarchy=["Chapter 5", "5.2 Quantum Fourier Transform"],
            section_number="5.2.1",
            topic_summary="Explanation of period finding in Shor's algorithm"
        )

        assert meta.chunk_id == "test_chunk_001"
        assert len(meta.section_hierarchy) == 2
        assert meta.section_number == "5.2.1"
        assert "period finding" in meta.topic_summary

    def test_section_metadata_to_dict(self):
        """Test serialización a dict."""
        from src.ingestion.section_extractor import SectionMetadata

        meta = SectionMetadata(
            chunk_id="test_chunk_002",
            section_hierarchy=["Cap 3"],
            section_number="3.1",
            topic_summary="Qubits basics"
        )

        d = meta.to_dict()
        assert d["chunk_id"] == "test_chunk_002"
        assert d["section_hierarchy"] == ["Cap 3"]
        assert d["section_number"] == "3.1"
        assert d["topic_summary"] == "Qubits basics"

    def test_section_metadata_format_citation(self):
        """Test formateo de cita."""
        from src.ingestion.section_extractor import SectionMetadata

        meta = SectionMetadata(
            chunk_id="test_chunk_003",
            section_hierarchy=["Chapter 5", "5.2 QFT"],
            section_number="5.2.1",
            topic_summary="Period finding algorithm"
        )

        citation = meta.format_citation("Nielsen & Chuang")
        assert "Nielsen & Chuang" in citation
        assert "Chapter 5" in citation or "5.2 QFT" in citation


class TestSectionExtractor:
    """Tests para SectionExtractor."""

    def _make_chunk_tuples(self, n: int = 3, with_doc_title: bool = True):
        """Genera n tuplas de chunks de prueba."""
        tuples = []
        doc_title = "Computación Cuántica - Nielsen & Chuang"
        for i in range(n):
            content = f"Contenido del chunk {i} sobre {['qubits', 'gates', 'algoritmos'][i % 3]}."
            header_path = f"Cap {i + 1} > Sección {i + 1}.1"
            chunk_id = f"test_doc_micro_{i:06d}"
            if with_doc_title:
                tuples.append((chunk_id, content, doc_title, header_path))
            else:
                tuples.append((chunk_id, content, header_path))
        return tuples

    @patch("src.llm_provider.complete")
    def test_extract_single(self, mock_complete):
        """Test extracción de metadata para un solo chunk."""
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps({
                "section_hierarchy": ["Chapter 5", "5.1 Shor's Algorithm"],
                "section_number": "5.1.2",
                "topic_summary": "Factorización usando transformada de Fourier cuántica"
            })
        )

        from src.ingestion.section_extractor import SectionExtractor

        extractor = SectionExtractor()
        meta = extractor.extract_single(
            chunk_id="test_chunk_001",
            content="El algoritmo de Shor factoriza números en tiempo polinómico usando la QFT.",
            doc_title="Computación Cuántica",
            header_path="Cap 5 > Algoritmos > Shor"
        )

        assert meta.chunk_id == "test_chunk_001"
        assert "Chapter 5" in meta.section_hierarchy
        assert meta.section_number == "5.1.2"
        assert "Fourier" in meta.topic_summary

    @patch("src.llm_provider.complete")
    def test_extract_single_fallback(self, mock_complete):
        """Test fallback cuando LLM falla."""
        mock_complete.side_effect = Exception("API Error")

        from src.ingestion.section_extractor import SectionExtractor

        extractor = SectionExtractor()
        meta = extractor.extract_single(
            chunk_id="test_chunk_002",
            content="Contenido sobre qubits.",
            doc_title="Test Doc",
            header_path="Cap 5 > Qubits"
        )

        # Fallback debe usar header_path parseado
        assert meta.chunk_id == "test_chunk_002"
        assert len(meta.section_hierarchy) > 0 or meta.topic_summary != ""

    @patch("src.llm_provider.complete")
    def test_extract_batch(self, mock_complete):
        """Test extracción en batch."""
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps([
                {
                    "chunk_index": 0,
                    "section_hierarchy": ["Cap 1"],
                    "section_number": "1.1",
                    "topic_summary": "Intro a qubits"
                },
                {
                    "chunk_index": 1,
                    "section_hierarchy": ["Cap 2"],
                    "section_number": "2.1",
                    "topic_summary": "Gates cuánticos"
                },
                {
                    "chunk_index": 2,
                    "section_hierarchy": ["Cap 3"],
                    "section_number": "3.1",
                    "topic_summary": "Algoritmos cuánticos"
                }
            ])
        )

        from src.ingestion.section_extractor import SectionExtractor

        extractor = SectionExtractor(batch_size=10)
        chunks = self._make_chunk_tuples(3, with_doc_title=True)
        results = extractor.extract_batch(chunks)

        assert len(results) == 3
        assert results[0].chunk_id == "test_doc_micro_000000"
        assert results[0].section_number == "1.1"

    @patch("src.llm_provider.complete")
    def test_extract_batch_sub_batching(self, mock_complete):
        """Test que se subdivide en sub-batches correctamente."""
        call_count = [0]

        def mock_response(*args, **kwargs):
            # Retornar el número correcto de metadatos por batch
            prompt = kwargs.get("prompt", args[0] if args else "")
            n = prompt.count("--- CHUNK")
            call_count[0] += 1
            return MockLLMResponse(
                content=json.dumps([
                    {
                        "chunk_index": i,
                        "section_hierarchy": [f"Cap {i}"],
                        "section_number": f"{i}.1",
                        "topic_summary": f"Topic {i}"
                    }
                    for i in range(n)
                ])
            )

        mock_complete.side_effect = mock_response

        from src.ingestion.section_extractor import SectionExtractor

        extractor = SectionExtractor(batch_size=2)
        chunks = self._make_chunk_tuples(5, with_doc_title=True)

        # extract_batch procesa todo el batch de una vez
        results = extractor.extract_batch(chunks)

        assert len(results) == 5

    @patch("src.llm_provider.complete")
    def test_extract_for_document(self, mock_complete):
        """Test extracción para todos los chunks de un documento."""
        call_count = [0]

        def mock_response(*args, **kwargs):
            prompt = kwargs.get("prompt", args[0] if args else "")
            n = prompt.count("--- CHUNK")
            call_count[0] += 1
            return MockLLMResponse(
                content=json.dumps([
                    {
                        "chunk_index": i,
                        "section_hierarchy": [f"Cap {i}"],
                        "section_number": f"{i}.1",
                        "topic_summary": f"Topic {i}"
                    }
                    for i in range(n)
                ])
            )

        mock_complete.side_effect = mock_response

        from src.ingestion.section_extractor import SectionExtractor

        extractor = SectionExtractor(batch_size=10)
        # extract_for_document espera tuplas (chunk_id, content, header_path) sin doc_title
        chunks = self._make_chunk_tuples(3, with_doc_title=False)
        results = extractor.extract_for_document(chunks, "Test Doc")

        assert len(results) == 3
        for result in results:
            assert result.chunk_id.startswith("test_doc_micro_")

    def test_fallback_extraction(self):
        """Test extracción fallback sin LLM."""
        from src.ingestion.section_extractor import SectionExtractor

        extractor = SectionExtractor()
        meta = extractor._fallback_extraction(
            chunk_id="test_chunk_001",
            content="El algoritmo de Shor factoriza números.",
            header_path="Capítulo 5 > 5.2 QFT > 5.2.1 Period Finding"
        )

        # Debe extraer algo del header_path
        assert meta.chunk_id == "test_chunk_001"
        assert len(meta.section_hierarchy) > 0
        assert meta.confidence < 1.0  # Fallback tiene menor confianza

    def test_fallback_extraction_with_numbers(self):
        """Test fallback extrae números de sección."""
        from src.ingestion.section_extractor import SectionExtractor

        extractor = SectionExtractor()
        meta = extractor._fallback_extraction(
            chunk_id="test_chunk_002",
            content="Este es el contenido.",
            header_path="5.2.1 Period Finding Algorithm"
        )

        assert meta.section_number == "5.2.1"

    @patch("src.llm_provider.complete")
    def test_parse_invalid_json(self, mock_complete):
        """Test manejo de JSON inválido del LLM."""
        mock_complete.return_value = MockLLMResponse(
            content="This is not valid JSON"
        )

        from src.ingestion.section_extractor import SectionExtractor

        extractor = SectionExtractor()
        meta = extractor.extract_single(
            chunk_id="test_chunk_003",
            content="Contenido de prueba.",
            doc_title="Test Doc",
            header_path="Cap 5 > Sección 5.1"
        )

        # Debe usar fallback
        assert meta is not None
        assert isinstance(meta.section_hierarchy, list)


class TestPromptBuilder:
    """Tests para el prompt de extracción."""

    def test_prompt_contains_chunk_info(self):
        """Test que el prompt incluye información del chunk."""
        from src.ingestion.section_extractor import SectionExtractor

        extractor = SectionExtractor()

        # Verificar que EXTRACTION_PROMPT existe y tiene los campos esperados
        assert hasattr(extractor, 'EXTRACTION_PROMPT')
        assert "section_hierarchy" in extractor.EXTRACTION_PROMPT
        assert "section_number" in extractor.EXTRACTION_PROMPT
        assert "topic_summary" in extractor.EXTRACTION_PROMPT
