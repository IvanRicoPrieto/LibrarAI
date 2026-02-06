"""Tests para Agent Interface (API optimizada para agentes)."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
import tempfile
import pickle

import pytest

# Configurar path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Mocks
# =============================================================================

@dataclass
class MockRetrievalResult:
    """Mock de RetrievalResult."""
    chunk_id: str = "test_doc_micro_000001"
    content: str = "Contenido del chunk sobre algoritmos cuánticos."
    score: float = 0.85
    doc_id: str = "test_doc"
    doc_title: str = "Nielsen & Chuang"
    header_path: str = "Cap 5 > Shor"
    retriever_type: str = "hybrid"


@dataclass
class MockChunk:
    """Mock de Chunk para tests."""
    chunk_id: str = "test_doc_micro_000001"
    content: str = "El algoritmo de Shor factoriza números en tiempo polinómico."
    doc_id: str = "test_doc"
    doc_title: str = "Nielsen & Chuang"
    header_path: str = "Cap 5 > Shor"
    section_number: str = "5.1"


@dataclass
class MockGeneratedResponse:
    """Mock de GeneratedResponse."""
    content: str = "Respuesta generada sobre algoritmos cuánticos."
    query: str = "test query"
    query_type: str = "test"
    sources_used: List[str] = field(default_factory=list)
    model: str = "test_model"
    tokens_input: int = 100
    tokens_output: int = 50
    latency_ms: float = 100.0


# =============================================================================
# Tests para Dataclasses
# =============================================================================

class TestContentNode:
    """Tests para ContentNode dataclass."""

    def test_creation(self):
        """Test creación básica."""
        from src.api import ContentNode

        node = ContentNode(
            id="doc_001",
            title="Documento de Prueba",
            level="document",
            path="Documento de Prueba",
            summary="Resumen del documento",
            token_count=1000,
            children_count=5,
            relevance_score=0.9,
        )

        assert node.id == "doc_001"
        assert node.level == "document"
        assert node.token_count == 1000

    def test_to_dict(self):
        """Test serialización a dict."""
        from src.api import ContentNode

        node = ContentNode(
            id="sec_001",
            title="Sección 1",
            level="section",
            path="Doc > Sección 1",
        )

        d = node.to_dict()
        assert d["id"] == "sec_001"
        assert d["level"] == "section"
        assert "path" in d


class TestExploreResult:
    """Tests para ExploreResult dataclass."""

    def test_creation_and_serialization(self):
        """Test creación y serialización."""
        from src.api import ExploreResult, ContentNode

        result = ExploreResult(
            query="algoritmo de Shor",
            total_documents=3,
            total_relevant_chunks=25,
            content_tree=[
                ContentNode(
                    id="nc",
                    title="Nielsen & Chuang",
                    level="document",
                    path="Nielsen & Chuang",
                    relevance_score=0.9,
                )
            ],
            topic_clusters=[
                {"entity": "QFT", "relation": "used_in", "score": 0.8}
            ],
            suggested_queries=[
                "¿Qué es el algoritmo de Shor?",
                "Aplicaciones de Shor",
            ],
            coverage_summary="Encontrados 3 documentos con 25 fragmentos.",
        )

        assert result.total_documents == 3
        assert len(result.content_tree) == 1
        assert len(result.suggested_queries) == 2

        # Test JSON
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["query"] == "algoritmo de Shor"
        assert len(parsed["content_tree"]) == 1


class TestRetrieveResult:
    """Tests para RetrieveResult dataclass."""

    def test_creation(self):
        """Test creación."""
        from src.api.agent_interface import RetrieveResult, SourceChunk

        result = RetrieveResult(
            query="QFT",
            total_chunks=10,
            chunks=[
                SourceChunk(
                    chunk_id="c1",
                    content="Contenido 1",
                    doc_id="d1",
                    doc_title="Doc 1",
                    section_path="Sec 1",
                    relevance_score=0.9,
                    token_count=100,
                )
            ],
            documents_covered=["d1"],
            sections_covered=["Sec 1"],
            total_tokens=100,
            retrieval_strategy="exhaustive",
        )

        assert result.total_chunks == 10
        assert result.retrieval_strategy == "exhaustive"


class TestQueryResult:
    """Tests para QueryResult dataclass."""

    def test_with_citations(self):
        """Test con citas."""
        from src.api.agent_interface import QueryResult, CitedClaim, SourceChunk

        result = QueryResult(
            query="¿Qué es el entrelazamiento?",
            answer="El entrelazamiento es una correlación cuántica [1].",
            claims=[
                CitedClaim(
                    claim="El entrelazamiento es una correlación cuántica",
                    citations=["nc_3.1"],
                    confidence=0.95,
                )
            ],
            sources_used=[
                SourceChunk(
                    chunk_id="nc_3.1",
                    content="El entrelazamiento cuántico...",
                    doc_id="nc",
                    doc_title="Nielsen & Chuang",
                    section_path="Cap 3",
                    relevance_score=0.9,
                    token_count=200,
                )
            ],
            confidence_score=0.9,
            model="claude-sonnet",
            tokens_used=500,
        )

        assert result.confidence_score == 0.9
        assert len(result.claims) == 1
        assert not result.abstained

    def test_abstention(self):
        """Test abstención cuando confianza es baja."""
        from src.api.agent_interface import QueryResult

        result = QueryResult(
            query="¿Cuál es la masa de un fotón?",
            answer="",
            claims=[],
            sources_used=[],
            confidence_score=0.2,
            abstained=True,
            abstention_reason="Confianza insuficiente",
        )

        assert result.abstained
        assert result.abstention_reason is not None


class TestVerifyResult:
    """Tests para VerifyResult dataclass."""

    def test_supported(self):
        """Test verificación soportada."""
        from src.api import VerifyResult, VerificationStatus
        from src.api.agent_interface import VerificationEvidence

        result = VerifyResult(
            claim="El algoritmo de Shor factoriza en tiempo polinómico",
            status=VerificationStatus.SUPPORTED,
            confidence=0.9,
            evidence=[
                VerificationEvidence(
                    chunk_id="nc_5.1",
                    content_excerpt="Shor demostró que la factorización...",
                    relevance="supports",
                    quote="tiempo polinómico O(log³n)",
                )
            ],
            explanation="Encontradas 3 fuentes que confirman la afirmación.",
            sources_checked=5,
        )

        assert result.status == VerificationStatus.SUPPORTED
        assert result.confidence == 0.9
        assert len(result.evidence) == 1

    def test_not_found(self):
        """Test afirmación no encontrada."""
        from src.api import VerifyResult, VerificationStatus

        result = VerifyResult(
            claim="Los unicornios usan qubits",
            status=VerificationStatus.NOT_FOUND,
            confidence=0.0,
            evidence=[],
            explanation="No se encontraron fuentes relevantes.",
            sources_checked=10,
        )

        assert result.status == VerificationStatus.NOT_FOUND


class TestCiteResult:
    """Tests para CiteResult dataclass."""

    def test_apa_style(self):
        """Test estilo APA."""
        from src.api.agent_interface import CiteResult, FormattedCitation

        result = CiteResult(
            citations=[
                FormattedCitation(
                    chunk_id="nc_5.1",
                    formatted="Nielsen & Chuang, (Chapter 5), §5.1",
                    style="apa",
                    raw_components={
                        "doc_title": "Nielsen & Chuang",
                        "section": "Chapter 5",
                        "section_number": "5.1",
                    },
                )
            ],
            style="apa",
            total_citations=1,
        )

        assert result.style == "apa"
        assert "Nielsen" in result.citations[0].formatted


# =============================================================================
# Tests para AgentAPI
# =============================================================================

class TestAgentAPI:
    """Tests para AgentAPI."""

    def _create_temp_indices(self) -> Path:
        """Crea directorio temporal con chunks mock."""
        temp_dir = Path(tempfile.mkdtemp())

        # Crear chunks.pkl
        chunks = {
            "test_doc_micro_000001": MockChunk(
                chunk_id="test_doc_micro_000001",
                content="El algoritmo de Shor factoriza números.",
                doc_id="test_doc",
                doc_title="Nielsen & Chuang",
                header_path="Cap 5 > Shor",
            ),
            "test_doc_micro_000002": MockChunk(
                chunk_id="test_doc_micro_000002",
                content="La QFT es fundamental para Shor.",
                doc_id="test_doc",
                doc_title="Nielsen & Chuang",
                header_path="Cap 5 > QFT",
            ),
        }
        with open(temp_dir / "chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)

        return temp_dir

    @patch("src.retrieval.fusion.UnifiedRetriever")
    def test_explore(self, mock_retriever_class):
        """Test modo EXPLORE."""
        # Setup mock
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            MockRetrievalResult(
                chunk_id="nc_5.1",
                doc_id="nc",
                doc_title="Nielsen & Chuang",
                header_path="Cap 5 > Shor",
                score=0.9,
            ),
            MockRetrievalResult(
                chunk_id="nc_5.2",
                doc_id="nc",
                doc_title="Nielsen & Chuang",
                header_path="Cap 5 > QFT",
                score=0.8,
            ),
        ]
        mock_retriever_class.return_value = mock_retriever

        temp_dir = self._create_temp_indices()

        from src.api import AgentAPI

        api = AgentAPI(indices_dir=temp_dir)
        api._retriever = mock_retriever

        result = api.explore("algoritmo de Shor")

        assert result.query == "algoritmo de Shor"
        assert result.total_documents >= 1
        assert len(result.content_tree) >= 1
        assert len(result.suggested_queries) > 0

    @patch("src.retrieval.fusion.UnifiedRetriever")
    def test_retrieve_normal(self, mock_retriever_class):
        """Test modo RETRIEVE normal (top-k)."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            MockRetrievalResult(chunk_id="c1", score=0.9),
            MockRetrievalResult(chunk_id="c2", score=0.8),
        ]
        mock_retriever_class.return_value = mock_retriever

        temp_dir = self._create_temp_indices()

        from src.api import AgentAPI

        api = AgentAPI(indices_dir=temp_dir)
        api._retriever = mock_retriever

        result = api.retrieve("QFT", max_chunks=10, expand_context=False)

        assert result.query == "QFT"
        assert result.retrieval_strategy == "top_k"
        assert len(result.chunks) == 2

    @patch("src.retrieval.fusion.UnifiedRetriever")
    def test_retrieve_exhaustive(self, mock_retriever_class):
        """Test modo RETRIEVE exhaustivo."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            MockRetrievalResult(chunk_id=f"c{i}", score=0.9-i*0.1)
            for i in range(5)
        ]
        mock_retriever_class.return_value = mock_retriever

        temp_dir = self._create_temp_indices()

        from src.api import AgentAPI

        api = AgentAPI(indices_dir=temp_dir)
        api._retriever = mock_retriever

        result = api.retrieve("QFT", exhaustive=True, expand_context=False)

        assert result.retrieval_strategy == "exhaustive"
        # En modo exhaustivo debería haber más resultados (de múltiples pasadas)
        assert len(result.chunks) >= 5

    @patch("src.retrieval.fusion.UnifiedRetriever")
    @patch("src.generation.synthesizer.ResponseSynthesizer")
    def test_query_with_response(self, mock_synth_class, mock_retriever_class):
        """Test modo QUERY genera respuesta."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            MockRetrievalResult(chunk_id="c1", score=0.9),
        ]
        mock_retriever_class.return_value = mock_retriever

        mock_synth = MagicMock()
        mock_synth.generate.return_value = MockGeneratedResponse(
            content="El entrelazamiento es una correlación cuántica.",
        )
        mock_synth_class.return_value = mock_synth

        temp_dir = self._create_temp_indices()

        from src.api import AgentAPI

        api = AgentAPI(indices_dir=temp_dir, use_grounding=False)
        api._retriever = mock_retriever
        api._synthesizer = mock_synth

        result = api.query("¿Qué es el entrelazamiento?")

        assert result.query == "¿Qué es el entrelazamiento?"
        assert "entrelazamiento" in result.answer.lower()
        assert not result.abstained

    @patch("src.retrieval.fusion.UnifiedRetriever")
    def test_query_abstention(self, mock_retriever_class):
        """Test QUERY abstención cuando no hay fuentes."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []  # Sin resultados
        mock_retriever_class.return_value = mock_retriever

        temp_dir = self._create_temp_indices()

        from src.api import AgentAPI

        api = AgentAPI(indices_dir=temp_dir)
        api._retriever = mock_retriever

        result = api.query("Pregunta sin respuesta en la biblioteca")

        assert result.abstained
        assert result.abstention_reason is not None

    @patch("src.retrieval.fusion.UnifiedRetriever")
    def test_verify_supported(self, mock_retriever_class):
        """Test VERIFY afirmación soportada."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            MockRetrievalResult(
                chunk_id="c1",
                content="El algoritmo de Shor factoriza números en tiempo polinómico.",
                score=0.9,
            ),
            MockRetrievalResult(
                chunk_id="c2",
                content="Shor demostró la factorización polinómica cuántica.",
                score=0.85,
            ),
        ]
        mock_retriever_class.return_value = mock_retriever

        temp_dir = self._create_temp_indices()

        from src.api import AgentAPI, VerificationStatus

        api = AgentAPI(indices_dir=temp_dir)
        api._retriever = mock_retriever

        result = api.verify("Shor factoriza en tiempo polinómico")

        # Con contenido que coincide, debería ser soportado o parcial
        assert result.status in [
            VerificationStatus.SUPPORTED,
            VerificationStatus.PARTIAL
        ]
        assert len(result.evidence) > 0

    @patch("src.retrieval.fusion.UnifiedRetriever")
    def test_verify_not_found(self, mock_retriever_class):
        """Test VERIFY afirmación no encontrada."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        mock_retriever_class.return_value = mock_retriever

        temp_dir = self._create_temp_indices()

        from src.api import AgentAPI, VerificationStatus

        api = AgentAPI(indices_dir=temp_dir)
        api._retriever = mock_retriever

        result = api.verify("Afirmación que no existe en ninguna fuente")

        assert result.status == VerificationStatus.NOT_FOUND
        assert result.sources_checked == 0

    def test_cite_apa(self):
        """Test CITE genera citas APA."""
        temp_dir = self._create_temp_indices()

        from src.api import AgentAPI

        api = AgentAPI(indices_dir=temp_dir)

        result = api.cite(
            chunk_ids=["test_doc_micro_000001"],
            style="apa",
        )

        assert result.style == "apa"
        assert result.total_citations == 1
        assert "Nielsen" in result.citations[0].formatted

    def test_cite_multiple_styles(self):
        """Test CITE con múltiples estilos."""
        temp_dir = self._create_temp_indices()

        from src.api import AgentAPI

        api = AgentAPI(indices_dir=temp_dir)

        for style in ["apa", "ieee", "chicago", "markdown", "inline"]:
            result = api.cite(
                chunk_ids=["test_doc_micro_000001"],
                style=style,
            )
            assert result.style == style
            assert len(result.citations) == 1

    def test_cite_bibliography(self):
        """Test CITE genera bibliografía para múltiples citas."""
        temp_dir = self._create_temp_indices()

        from src.api import AgentAPI

        api = AgentAPI(indices_dir=temp_dir)

        result = api.cite(
            chunk_ids=["test_doc_micro_000001", "test_doc_micro_000002"],
            style="apa",
        )

        assert result.total_citations == 2
        assert result.bibliography is not None
        assert "[1]" in result.bibliography
        assert "[2]" in result.bibliography


# =============================================================================
# Tests de integración JSON
# =============================================================================

class TestJSONSerialization:
    """Tests para verificar que todos los outputs son JSON válido."""

    def test_all_results_json_serializable(self):
        """Verifica que todos los tipos de resultado son serializables."""
        from src.api.agent_interface import (
            ExploreResult, RetrieveResult, QueryResult,
            VerifyResult, CiteResult, ContentNode, SourceChunk,
            CitedClaim, VerificationEvidence, FormattedCitation,
            VerificationStatus,
        )

        # ExploreResult
        explore = ExploreResult(
            query="test",
            total_documents=1,
            total_relevant_chunks=5,
            content_tree=[ContentNode(
                id="1", title="Test", level="document", path="Test"
            )],
            topic_clusters=[],
            suggested_queries=["q1"],
            coverage_summary="Test",
        )
        json.loads(explore.to_json())

        # RetrieveResult
        retrieve = RetrieveResult(
            query="test",
            total_chunks=1,
            chunks=[SourceChunk(
                chunk_id="c1",
                content="content",
                doc_id="d1",
                doc_title="Doc",
                section_path="Sec",
            )],
            documents_covered=["d1"],
            sections_covered=["Sec"],
            total_tokens=100,
            retrieval_strategy="top_k",
        )
        json.loads(retrieve.to_json())

        # QueryResult
        query = QueryResult(
            query="test",
            answer="answer",
            claims=[CitedClaim(claim="c", citations=["c1"])],
            sources_used=[],
            confidence_score=0.9,
        )
        json.loads(query.to_json())

        # VerifyResult
        verify = VerifyResult(
            claim="test",
            status=VerificationStatus.SUPPORTED,
            confidence=0.9,
            evidence=[VerificationEvidence(
                chunk_id="c1",
                content_excerpt="excerpt",
                relevance="supports",
            )],
            explanation="exp",
            sources_checked=5,
        )
        json.loads(verify.to_json())

        # CiteResult
        cite = CiteResult(
            citations=[FormattedCitation(
                chunk_id="c1",
                formatted="Citation",
                style="apa",
                raw_components={},
            )],
            style="apa",
            total_citations=1,
        )
        json.loads(cite.to_json())
