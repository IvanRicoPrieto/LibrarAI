"""Tests para el módulo de Fine-tuning de Embeddings."""

import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
from typing import List, Optional

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class MockChunk:
    chunk_id: str = "doc1_micro_000001"
    content: str = "El algoritmo de Shor permite factorizar números en tiempo polinómico."
    level: str = "micro"
    doc_id: str = "doc1"
    doc_title: str = "Computación Cuántica"
    header_path: str = "Cap 5 > Shor"
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    token_count: int = 30
    content_hash: str = "abc123"


class MockLLMResponse:
    def __init__(self, content=""):
        self.content = content
        self.ok = True
        self.model = "mock"
        self.provider = "mock"
        self.tokens_input = 100
        self.tokens_output = 50


class TestSyntheticPairGenerator:

    @patch("src.llm_provider.complete")
    def test_generate_from_chunks(self, mock_complete):
        """Test generación de pares desde chunks."""
        response_data = {
            "chunks": [
                {
                    "index": 0,
                    "queries": [
                        {"query": "¿Quién propuso el algoritmo de Shor?", "type": "factual", "difficulty": "easy"},
                        {"query": "¿Cómo funciona la factorización cuántica?", "type": "conceptual", "difficulty": "medium"},
                        {"query": "Algoritmo cuántico para factorizar números", "type": "general", "difficulty": "easy"},
                    ],
                }
            ]
        }
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps(response_data)
        )

        from src.finetuning.pair_generator import SyntheticPairGenerator

        gen = SyntheticPairGenerator(queries_per_chunk=3, batch_size=5)
        chunks = [MockChunk()]
        pairs = gen.generate_from_chunks(chunks)

        assert len(pairs) == 3
        for pair in pairs:
            assert pair.chunk_id == chunks[0].chunk_id
            assert pair.positive_passage == chunks[0].content
            assert len(pair.query) >= 15

    @patch("src.llm_provider.complete")
    def test_deduplication(self, mock_complete):
        """Test deduplicación de queries."""
        response_data = {
            "chunks": [
                {
                    "index": 0,
                    "queries": [
                        {"query": "¿Qué es el algoritmo de Shor?", "type": "factual", "difficulty": "easy"},
                        {"query": "¿Qué es el algoritmo de Shor?", "type": "factual", "difficulty": "easy"},
                    ],
                }
            ]
        }
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps(response_data)
        )

        from src.finetuning.pair_generator import SyntheticPairGenerator

        gen = SyntheticPairGenerator()
        pairs = gen.generate_from_chunks([MockChunk()])

        # Duplicados deben eliminarse
        assert len(pairs) == 1

    def test_training_pair_hash(self):
        """Test que el hash se genera automáticamente."""
        from src.finetuning.pair_generator import TrainingPair

        pair = TrainingPair(
            query="¿Qué es un qubit?",
            positive_passage="Un qubit es...",
            chunk_id="chunk_001",
            doc_id="doc1",
            doc_title="Test",
            difficulty="easy",
            query_type="factual",
        )

        assert pair.query_hash != ""
        assert len(pair.query_hash) == 12


class TestOpenAIFineTuneFormatter:

    def test_format_pairs(self, tmp_path):
        """Test formateo de pares a JSONL."""
        from src.finetuning.pair_generator import TrainingPair
        from src.finetuning.formatter import OpenAIFineTuneFormatter

        pairs = [
            TrainingPair(
                query=f"Query {i}",
                positive_passage=f"Passage {i}",
                chunk_id=f"chunk_{i}",
                doc_id="doc1",
                doc_title="Test",
                difficulty="easy",
                query_type="factual",
            )
            for i in range(5)
        ]

        output_file = tmp_path / "train.jsonl"
        formatter = OpenAIFineTuneFormatter()
        stats = formatter.format_pairs(pairs, output_file)

        assert output_file.exists()
        assert stats["total_pairs"] == 5
        assert stats["format"] == "pair"

        # Verificar contenido
        with open(output_file) as f:
            lines = f.readlines()
        assert len(lines) == 5
        first = json.loads(lines[0])
        assert "prompt" in first
        assert "completion" in first

    def test_format_with_negatives(self, tmp_path):
        """Test formateo con hard negatives."""
        from src.finetuning.formatter import OpenAIFineTuneFormatter

        negatives = [
            {
                "query": "¿Qué es un qubit?",
                "positive": "Un qubit es...",
                "negatives": ["Negativo 1", "Negativo 2"],
            }
        ]

        output_file = tmp_path / "train_triplet.jsonl"
        formatter = OpenAIFineTuneFormatter()
        stats = formatter.format_pairs([], output_file, negatives=negatives)

        assert stats["total_pairs"] == 2  # 2 negatives
        assert stats["format"] == "triplet"

    def test_validate_format_valid(self, tmp_path):
        """Test validación de archivo válido."""
        from src.finetuning.formatter import OpenAIFineTuneFormatter

        file_path = tmp_path / "valid.jsonl"
        with open(file_path, "w") as f:
            f.write('{"prompt": "q1", "completion": "a1"}\n')
            f.write('{"prompt": "q2", "completion": "a2"}\n')

        formatter = OpenAIFineTuneFormatter()
        result = formatter.validate_format(file_path)

        assert result["valid"]
        assert result["total_lines"] == 2
        assert result["format_detected"] == "pair"

    def test_validate_format_invalid_json(self, tmp_path):
        """Test validación con JSON inválido."""
        from src.finetuning.formatter import OpenAIFineTuneFormatter

        file_path = tmp_path / "invalid.jsonl"
        with open(file_path, "w") as f:
            f.write("not json\n")

        formatter = OpenAIFineTuneFormatter()
        result = formatter.validate_format(file_path)

        assert not result["valid"]
        assert len(result["errors"]) > 0

    def test_validate_detects_duplicates(self, tmp_path):
        """Test que detecta duplicados."""
        from src.finetuning.formatter import OpenAIFineTuneFormatter

        file_path = tmp_path / "dupes.jsonl"
        with open(file_path, "w") as f:
            f.write('{"prompt": "q1", "completion": "a1"}\n')
            f.write('{"prompt": "q1", "completion": "a1"}\n')

        formatter = OpenAIFineTuneFormatter()
        result = formatter.validate_format(file_path)

        assert result["duplicates"] == 1

    def test_split_train_val(self, tmp_path):
        """Test split train/val."""
        from src.finetuning.formatter import OpenAIFineTuneFormatter

        input_file = tmp_path / "data.jsonl"
        with open(input_file, "w") as f:
            for i in range(100):
                f.write(f'{{"prompt": "q{i}", "completion": "a{i}"}}\n')

        formatter = OpenAIFineTuneFormatter()
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"

        stats = formatter.split_train_val(
            input_file, train_path, val_path, val_fraction=0.2
        )

        assert stats["total"] == 100
        assert stats["train"] == 80
        assert stats["val"] == 20
        assert train_path.exists()
        assert val_path.exists()

    def test_validate_nonexistent_file(self, tmp_path):
        """Test validación de archivo inexistente."""
        from src.finetuning.formatter import OpenAIFineTuneFormatter

        formatter = OpenAIFineTuneFormatter()
        result = formatter.validate_format(tmp_path / "no_existe.jsonl")

        assert not result["valid"]
        assert "no encontrado" in result.get("error", "").lower()


class TestFineTuneLauncher:

    def test_init_without_key(self):
        """Test inicialización sin API key."""
        from src.finetuning.launcher import FineTuneLauncher

        launcher = FineTuneLauncher(api_key=None)
        # No debe fallar en init, solo al llamar métodos
        assert launcher._client is None
