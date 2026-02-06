"""Tests para AgenticRAGPipeline (Agentic RAG)."""

import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class MockResult:
    chunk_id: str = "chunk_001"
    content: str = "El algoritmo de Shor factoriza números."
    score: float = 0.8
    doc_id: str = "doc1"
    doc_title: str = "Computación Cuántica"
    header_path: str = "Cap 5"


@dataclass
class MockGeneratedResponse:
    content: str = "Respuesta generada."
    query: str = "test"
    query_type: str = "factual"
    sources_used: List[str] = field(default_factory=list)
    model: str = "mock"
    tokens_input: int = 100
    tokens_output: int = 50
    latency_ms: float = 100.0
    abstained: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockLLMResponse:
    def __init__(self, content=""):
        self.content = content
        self.ok = True
        self.model = "mock"
        self.provider = "mock"
        self.tokens_input = 100
        self.tokens_output = 50


class TestAgenticRAGPipeline:

    def _make_retriever(self, results=None):
        """Crea un mock retriever."""
        mock = MagicMock()
        if results is None:
            results = [
                MockResult(chunk_id=f"chunk_{i:03d}", score=0.8 - i * 0.1)
                for i in range(5)
            ]
        mock.search.return_value = results
        return mock

    def _make_synthesizer(self):
        """Crea un mock synthesizer."""
        mock = MagicMock()
        mock.generate.return_value = MockGeneratedResponse()
        return mock

    @patch("src.llm_provider.complete")
    def test_sufficient_first_iteration(self, mock_complete):
        """Test que termina en la primera iteración si contexto suficiente."""
        reflection = {
            "is_sufficient": True,
            "confidence": 0.9,
            "missing_aspects": [],
            "suggested_reformulation": None,
            "reasoning": "Contexto completo.",
        }
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps(reflection)
        )

        from src.agents.agentic_rag import AgenticRAGPipeline

        retriever = self._make_retriever()
        synthesizer = self._make_synthesizer()

        pipeline = AgenticRAGPipeline(
            retriever=retriever,
            synthesizer=synthesizer,
            max_iterations=4,
        )

        response, results, state = pipeline.ask("test query", top_k=5)

        assert state.iteration == 1
        assert state.is_sufficient is True
        assert synthesizer.generate.called

    @patch("src.llm_provider.complete")
    def test_reformulation(self, mock_complete):
        """Test que reformula cuando el contexto es insuficiente."""
        reflections = [
            {
                "is_sufficient": False,
                "confidence": 0.3,
                "missing_aspects": ["detalles de implementación"],
                "suggested_reformulation": "implementación del algoritmo de Shor paso a paso",
                "reasoning": "Falta detalle.",
            },
            {
                "is_sufficient": True,
                "confidence": 0.85,
                "missing_aspects": [],
                "suggested_reformulation": None,
                "reasoning": "Ahora sí.",
            },
        ]
        mock_complete.side_effect = [
            MockLLMResponse(content=json.dumps(reflections[0])),
            MockLLMResponse(content=json.dumps(reflections[1])),
        ]

        from src.agents.agentic_rag import AgenticRAGPipeline

        retriever = self._make_retriever()
        synthesizer = self._make_synthesizer()

        pipeline = AgenticRAGPipeline(
            retriever=retriever,
            synthesizer=synthesizer,
            max_iterations=4,
        )

        response, results, state = pipeline.ask("algoritmo de Shor", top_k=5)

        assert state.iteration == 2
        assert len(state.search_strategies_used) == 2
        assert state.search_strategies_used[1] != state.search_strategies_used[0]

    @patch("src.llm_provider.complete")
    def test_max_iterations(self, mock_complete):
        """Test que respeta max_iterations."""
        reflection = {
            "is_sufficient": False,
            "confidence": 0.2,
            "missing_aspects": ["todo"],
            "suggested_reformulation": "otra query",
            "reasoning": "Insuficiente.",
        }
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps(reflection)
        )

        from src.agents.agentic_rag import AgenticRAGPipeline

        retriever = self._make_retriever()
        synthesizer = self._make_synthesizer()

        pipeline = AgenticRAGPipeline(
            retriever=retriever,
            synthesizer=synthesizer,
            max_iterations=3,
        )

        response, results, state = pipeline.ask("test", top_k=5)

        assert state.iteration <= 3

    def test_accumulate_dedup(self):
        """Test deduplicación en acumulación de resultados."""
        from src.agents.agentic_rag import AgenticRAGPipeline, AgentState

        state = AgentState(
            original_query="test",
            current_query="test",
        )

        results1 = [
            MockResult(chunk_id="a"),
            MockResult(chunk_id="b"),
        ]
        results2 = [
            MockResult(chunk_id="b"),  # Duplicado
            MockResult(chunk_id="c"),
        ]

        pipeline = AgenticRAGPipeline(
            retriever=MagicMock(),
            synthesizer=MagicMock(),
        )

        n1 = pipeline._accumulate_results(state, results1)
        n2 = pipeline._accumulate_results(state, results2)

        assert n1 == 2
        assert n2 == 1  # Solo 'c' es nuevo
        assert len(state.accumulated_results) == 3
        assert len(state.accumulated_chunk_ids) == 3

    @patch("src.llm_provider.complete")
    def test_stops_when_no_new_results(self, mock_complete):
        """Test que para cuando no hay resultados nuevos."""
        reflection = {
            "is_sufficient": False,
            "confidence": 0.5,
            "missing_aspects": ["más info"],
            "suggested_reformulation": "otra query",
            "reasoning": "Falta.",
        }
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps(reflection)
        )

        # Retriever siempre devuelve los mismos resultados
        same_results = [MockResult(chunk_id="same_1"), MockResult(chunk_id="same_2")]
        retriever = self._make_retriever(same_results)
        synthesizer = self._make_synthesizer()

        from src.agents.agentic_rag import AgenticRAGPipeline

        pipeline = AgenticRAGPipeline(
            retriever=retriever,
            synthesizer=synthesizer,
            max_iterations=5,
            min_new_results=2,
        )

        response, results, state = pipeline.ask("test", top_k=5)

        # Debe parar pronto por falta de resultados nuevos
        assert state.iteration <= 3

    def test_reasoning_trace(self):
        """Test que el reasoning trace se llena correctamente."""
        from src.agents.agentic_rag import AgentState

        state = AgentState(
            original_query="test",
            current_query="test",
        )
        state.reasoning_trace.append("Paso 1")
        state.reasoning_trace.append("Paso 2")

        assert len(state.reasoning_trace) == 2

    @patch("src.llm_provider.complete")
    def test_reflection_llm_error_fallback(self, mock_complete):
        """Test fallback heurístico cuando reflexión LLM falla."""
        mock_complete.side_effect = Exception("API error")

        from src.agents.agentic_rag import AgenticRAGPipeline, AgentState

        pipeline = AgenticRAGPipeline(
            retriever=MagicMock(),
            synthesizer=MagicMock(),
        )

        state = AgentState(
            original_query="test",
            current_query="test",
            accumulated_results=[MockResult() for _ in range(5)],
        )

        reflection = pipeline._reflect(state)

        # Debe funcionar con heurístico
        assert isinstance(reflection.confidence, float)
        assert isinstance(reflection.is_sufficient, bool)

    def test_metadata_in_response(self):
        """Test que la metadata del agente se incluye en la respuesta."""
        from src.agents.agentic_rag import AgenticRAGPipeline

        retriever = self._make_retriever()
        synthesizer = self._make_synthesizer()

        pipeline = AgenticRAGPipeline(
            retriever=retriever,
            synthesizer=synthesizer,
            max_iterations=1,
        )

        with patch("src.llm_provider.complete") as mock_complete:
            reflection = {
                "is_sufficient": True,
                "confidence": 0.9,
                "missing_aspects": [],
                "suggested_reformulation": None,
                "reasoning": "OK",
            }
            mock_complete.return_value = MockLLMResponse(
                content=json.dumps(reflection)
            )

            response, results, state = pipeline.ask("test", top_k=5)

        assert "agentic_rag" in response.metadata
        assert "iterations" in response.metadata["agentic_rag"]
        assert "confidence" in response.metadata["agentic_rag"]
