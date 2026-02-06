"""Tests para LLMGraphExtractor (LLM Knowledge Graph Extraction)."""

import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class MockLLMResponse:
    def __init__(self, content=""):
        self.content = content
        self.ok = True
        self.model = "mock"
        self.provider = "mock"
        self.tokens_input = 100
        self.tokens_output = 50


class TestOntologyPromptBuilder:

    def test_defaults_without_ontology(self):
        """Test tipos por defecto sin archivo de ontología."""
        from src.ingestion.graph_builder import OntologyPromptBuilder

        builder = OntologyPromptBuilder(ontology_path=None)
        assert len(builder.valid_entity_types) > 0
        assert "Algoritmo" in builder.valid_entity_types
        assert "Concepto" in builder.valid_entity_types

    def test_defaults_relations(self):
        """Test relaciones por defecto."""
        from src.ingestion.graph_builder import OntologyPromptBuilder

        builder = OntologyPromptBuilder()
        assert "MEJORA" in builder.valid_relation_types
        assert "DEPENDE_DE" in builder.valid_relation_types

    def test_build_extraction_prompt(self):
        """Test que el prompt contiene los tipos válidos."""
        from src.ingestion.graph_builder import OntologyPromptBuilder

        builder = OntologyPromptBuilder()
        prompts = builder.build_extraction_prompt("Texto de prueba")

        assert "system" in prompts
        assert "user" in prompts
        assert "Algoritmo" in prompts["system"]
        assert "MEJORA" in prompts["system"]
        assert "Texto de prueba" in prompts["user"]

    def test_load_ontology_from_yaml(self, tmp_path):
        """Test carga de ontología desde YAML."""
        ontology_content = {
            "entities": {
                "Qubit": {"description": "Un qubit", "examples": ["qubit"]},
                "Gate": {"description": "Una puerta", "examples": ["CNOT"]},
            },
            "relations": {
                "APLICA": {"description": "Aplica a"},
                "COMPONE": {"description": "Se compone de"},
            },
        }

        import yaml

        ontology_file = tmp_path / "ontology.yaml"
        with open(ontology_file, "w") as f:
            yaml.dump(ontology_content, f)

        from src.ingestion.graph_builder import OntologyPromptBuilder

        builder = OntologyPromptBuilder(ontology_path=ontology_file)

        assert "Qubit" in builder.valid_entity_types
        assert "Gate" in builder.valid_entity_types
        assert "APLICA" in builder.valid_relation_types


class TestLLMGraphExtractor:

    @patch("src.llm_provider.complete")
    def test_extract_from_chunk(self, mock_complete):
        """Test extracción de entidades y triples de un chunk."""
        response_data = {
            "entities": [
                {"name": "Algoritmo de Shor", "type": "Algoritmo"},
                {"name": "Factorización", "type": "Concepto"},
            ],
            "triples": [
                {
                    "source": "Algoritmo de Shor",
                    "source_type": "Algoritmo",
                    "relation": "IMPLEMENTA",
                    "target": "Factorización",
                    "target_type": "Concepto",
                    "confidence": 0.95,
                }
            ],
        }
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps(response_data)
        )

        from src.ingestion.graph_builder import LLMGraphExtractor

        extractor = LLMGraphExtractor()
        result = extractor.extract_from_chunk(
            "chunk_001",
            "El algoritmo de Shor implementa la factorización cuántica."
        )

        assert len(result.entities) == 2
        assert len(result.triples) == 1
        assert result.triples[0].source_name == "Algoritmo de Shor"
        assert result.triples[0].relation_type == "IMPLEMENTA"
        assert result.chunk_id == "chunk_001"

    @patch("src.llm_provider.complete")
    def test_extract_filters_low_confidence(self, mock_complete):
        """Test que se filtran triples con baja confianza."""
        response_data = {
            "entities": [
                {"name": "Shor", "type": "Algoritmo"},
            ],
            "triples": [
                {
                    "source": "Shor",
                    "source_type": "Algoritmo",
                    "relation": "USA",
                    "target": "QFT",
                    "target_type": "Concepto",
                    "confidence": 0.9,
                },
                {
                    "source": "Shor",
                    "source_type": "Algoritmo",
                    "relation": "MEJORA",
                    "target": "RSA",
                    "target_type": "Concepto",
                    "confidence": 0.3,  # Baja confianza
                },
            ],
        }
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps(response_data)
        )

        from src.ingestion.graph_builder import LLMGraphExtractor

        extractor = LLMGraphExtractor(min_confidence=0.7)
        result = extractor.extract_from_chunk("chunk_001", "Texto")

        assert len(result.triples) == 1
        assert result.triples[0].confidence >= 0.7

    @patch("src.llm_provider.complete")
    def test_extract_validates_entity_types(self, mock_complete):
        """Test que se filtran tipos de entidad inválidos."""
        response_data = {
            "entities": [
                {"name": "Shor", "type": "Algoritmo"},
                {"name": "Python", "type": "LenguajeProgramacion"},  # Tipo inválido
            ],
            "triples": [],
        }
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps(response_data)
        )

        from src.ingestion.graph_builder import LLMGraphExtractor

        extractor = LLMGraphExtractor()
        result = extractor.extract_from_chunk("chunk_001", "Texto")

        # Solo debe quedar Shor (Algoritmo es válido)
        assert len(result.entities) == 1
        assert result.entities[0]["name"] == "Shor"

    @patch("src.llm_provider.complete")
    def test_extract_batch(self, mock_complete):
        """Test extracción por batch."""
        response_data = {
            "0": {
                "entities": [{"name": "Shor", "type": "Algoritmo"}],
                "triples": [],
            },
            "1": {
                "entities": [{"name": "BB84", "type": "Protocolo"}],
                "triples": [],
            },
        }
        mock_complete.return_value = MockLLMResponse(
            content=json.dumps(response_data)
        )

        from src.ingestion.graph_builder import LLMGraphExtractor

        extractor = LLMGraphExtractor()
        chunks = [
            ("chunk_a", "Texto sobre Shor"),
            ("chunk_b", "Texto sobre BB84"),
        ]
        results = extractor.extract_batch(chunks)

        assert len(results) == 2
        assert results[0].chunk_id == "chunk_a"
        assert results[1].chunk_id == "chunk_b"

    @patch("src.llm_provider.complete")
    def test_extract_llm_error(self, mock_complete):
        """Test fallback cuando LLM falla."""
        mock_complete.side_effect = Exception("API error")

        from src.ingestion.graph_builder import LLMGraphExtractor

        extractor = LLMGraphExtractor()
        result = extractor.extract_from_chunk("chunk_001", "Texto")

        assert result.chunk_id == "chunk_001"
        assert result.entities == []
        assert result.triples == []

    def test_fuzzy_match_type(self):
        """Test mapeo case-insensitive de tipos."""
        from src.ingestion.graph_builder import LLMGraphExtractor

        valid = {"Algoritmo", "Concepto", "Gate"}
        assert LLMGraphExtractor._fuzzy_match_type("algoritmo", valid) == "Algoritmo"
        assert LLMGraphExtractor._fuzzy_match_type("CONCEPTO", valid) == "Concepto"
        assert LLMGraphExtractor._fuzzy_match_type("invalido", valid) is None

    def test_extracted_triple_dataclass(self):
        """Test del dataclass ExtractedTriple."""
        from src.ingestion.graph_builder import ExtractedTriple

        triple = ExtractedTriple(
            source_name="Shor",
            source_type="Algoritmo",
            relation_type="USA",
            target_name="QFT",
            target_type="Concepto",
            confidence=0.95,
            chunk_id="chunk_001",
        )
        assert triple.confidence == 0.95
        assert triple.chunk_id == "chunk_001"
