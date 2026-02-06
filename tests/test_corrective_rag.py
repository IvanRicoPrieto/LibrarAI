"""Tests para CorrectiveRAG (CRAG)."""

import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
from typing import Dict, Any

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class MockResult:
    """Mock de RetrievalResult."""
    chunk_id: str = "chunk_001"
    content: str = "El algoritmo de Shor factoriza números en tiempo polinómico."
    score: float = 0.8
    doc_id: str = "doc1"
    doc_title: str = "Computación Cuántica"
    header_path: str = "Cap 5"
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockLLMResponse:
    def __init__(self, content=""):
        self.content = content
        self.ok = True
        self.model = "mock"
        self.provider = "mock"
        self.tokens_input = 100
        self.tokens_output = 50


class TestCorrectiveRAG:

    def _make_results(self, n=5) -> list:
        """Genera n resultados de prueba."""
        return [
            MockResult(
                chunk_id=f"chunk_{i:03d}",
                content=f"Contenido relevante sobre tema {i}.",
                score=0.8 - i * 0.1,
            )
            for i in range(n)
        ]

    @patch("src.llm_provider.complete")
    def test_all_correct_pass_through(self, mock_complete):
        """Test pass-through cuando todo es relevante."""
        assessments = [
            {"index": i, "verdict": "correct", "confidence": 0.9, "reasoning": "Relevante"}
            for i in range(3)
        ]
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps(assessments)
        )

        from src.retrieval.corrective_rag import CorrectiveRAG

        crag = CorrectiveRAG()
        results = self._make_results(3)
        crag_result = crag.correct("test query", results, top_k=3)

        assert crag_result.action_taken == "pass"
        assert len(crag_result.corrected_results) == 3

    @patch("src.llm_provider.complete")
    def test_all_incorrect_reformulate(self, mock_complete):
        """Test reformulación cuando todo es irrelevante."""
        # Primera llamada: evaluación
        assessments = [
            {"index": i, "verdict": "incorrect", "confidence": 0.9, "reasoning": "No relevante"}
            for i in range(5)
        ]
        # Segunda llamada: reformulación
        mock_complete.side_effect = [
            MockLLMResponse(content=json.dumps(assessments)),
            MockLLMResponse(content="query reformulada sobre computación cuántica"),
        ]

        from src.retrieval.corrective_rag import CorrectiveRAG

        crag = CorrectiveRAG(retriever=None)  # Sin retriever, no re-busca
        results = self._make_results(5)
        crag_result = crag.correct("test query", results, top_k=5)

        assert crag_result.action_taken == "reformulate"
        assert crag_result.reformulated_query is not None

    @patch("src.llm_provider.complete")
    def test_mixed_results_filter(self, mock_complete):
        """Test filtrado con resultados mixtos."""
        assessments = [
            {"index": 0, "verdict": "correct", "confidence": 0.9, "reasoning": "OK"},
            {"index": 1, "verdict": "incorrect", "confidence": 0.9, "reasoning": "No"},
            {"index": 2, "verdict": "ambiguous", "confidence": 0.5, "reasoning": "Parcial"},
        ]
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps(assessments)
        )

        from src.retrieval.corrective_rag import CorrectiveRAG

        crag = CorrectiveRAG()
        results = self._make_results(3)
        crag_result = crag.correct("test query", results, top_k=3)

        assert crag_result.action_taken == "filter"
        # Solo CORRECT y AMBIGUOUS pasan
        assert len(crag_result.corrected_results) == 2

    def test_heuristic_assessment(self):
        """Test evaluación heurística por keyword overlap."""
        from src.retrieval.corrective_rag import CorrectiveRAG, RelevanceVerdict

        crag = CorrectiveRAG(use_llm_assessment=False)

        results = [
            MockResult(content="El algoritmo de Shor factoriza números cuánticos."),
            MockResult(content="Las flores crecen en primavera con el sol."),
        ]

        assessments = crag._assess_relevance_heuristic(
            "algoritmo de Shor cuántico", results
        )

        assert len(assessments) == 2
        # El primer resultado debe ser más relevante
        assert assessments[0].relevance_score > assessments[1].relevance_score

    def test_empty_results(self):
        """Test con resultados vacíos."""
        from src.retrieval.corrective_rag import CorrectiveRAG

        crag = CorrectiveRAG()
        crag_result = crag.correct("test", [], top_k=5)

        assert crag_result.action_taken == "pass"
        assert len(crag_result.corrected_results) == 0

    def test_merge_and_dedup(self):
        """Test merge sin duplicados."""
        from src.retrieval.corrective_rag import CorrectiveRAG

        originals = [
            MockResult(chunk_id="a"),
            MockResult(chunk_id="b"),
        ]
        new_results = [
            MockResult(chunk_id="b"),  # Duplicado
            MockResult(chunk_id="c"),
        ]

        merged = CorrectiveRAG._merge_and_dedup(originals, new_results, top_k=5)

        ids = [r.chunk_id for r in merged]
        assert len(ids) == 3
        assert len(set(ids)) == 3  # Sin duplicados

    def test_stats_tracking(self):
        """Test que las stats se llenan correctamente."""
        from src.retrieval.corrective_rag import CorrectiveRAG

        crag = CorrectiveRAG(use_llm_assessment=False)
        results = self._make_results(3)
        crag_result = crag.correct("test query", results, top_k=3)

        assert "correct" in crag_result.stats
        assert "ambiguous" in crag_result.stats
        assert "incorrect" in crag_result.stats
        assert crag_result.stats["total"] == 3

    @patch("src.llm_provider.complete")
    def test_reformulate_with_retriever(self, mock_complete):
        """Test reformulación con re-búsqueda."""
        assessments = [
            {"index": i, "verdict": "incorrect", "confidence": 0.9, "reasoning": "No"}
            for i in range(3)
        ]
        mock_complete.side_effect = [
            MockLLMResponse(content=json.dumps(assessments)),
            MockLLMResponse(content="query reformulada"),
        ]

        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            MockResult(chunk_id="new_1", content="Nuevo resultado relevante"),
        ]

        from src.retrieval.corrective_rag import CorrectiveRAG

        crag = CorrectiveRAG(retriever=mock_retriever)
        results = self._make_results(3)
        crag_result = crag.correct("test query", results, top_k=5)

        assert crag_result.action_taken == "reformulate"
        assert mock_retriever.search.called
