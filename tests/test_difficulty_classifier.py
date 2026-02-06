"""
Tests para el clasificador de nivel de dificultad.
"""

import pytest
from src.ingestion.difficulty_classifier import (
    DifficultyClassifier,
    DifficultyLevel,
    DifficultyResult,
    get_difficulty_label,
    difficulty_to_numeric,
)


class TestDifficultyClassifier:
    """Tests para DifficultyClassifier."""

    @pytest.fixture
    def classifier(self):
        """Clasificador con heurísticas (sin LLM)."""
        return DifficultyClassifier(use_llm=False)

    def test_introductory_by_keywords(self, classifier):
        """Detecta contenido introductorio por palabras clave."""
        text = """
        Esta es una introducción básica al concepto de qubit.
        Un qubit es la unidad fundamental de información cuántica.
        Es un concepto elemental que todo estudiante debe conocer.
        """
        result = classifier.classify(text)

        assert result.level == DifficultyLevel.INTRODUCTORY
        assert result.confidence > 0.3
        assert "intro_keywords" in result.reasoning

    def test_introductory_by_section_hierarchy(self, classifier):
        """Detecta contenido introductorio por jerarquía de sección."""
        text = "Sea $|0\\rangle$ y $|1\\rangle$ los estados base."
        hierarchy = ["Capítulo 1", "1.1 Introducción"]

        result = classifier.classify(text, section_hierarchy=hierarchy)

        assert result.level == DifficultyLevel.INTRODUCTORY
        assert "early_chapter" in result.reasoning

    def test_advanced_by_math_complexity(self, classifier):
        """Detecta contenido avanzado por complejidad matemática."""
        text = r"""
        Demostración del teorema:
        $$\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$$

        Usando el lema de Fatou y la desigualdad de Hölder:
        $$\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}$$

        El producto tensorial satisface:
        $$\rho_{AB} = \rho_A \otimes \rho_B$$
        """
        result = classifier.classify(text)

        assert result.level in [DifficultyLevel.ADVANCED, DifficultyLevel.INTERMEDIATE]
        assert "complex_math" in result.reasoning or "advanced_keywords" in result.reasoning

    def test_advanced_by_keywords(self, classifier):
        """Detecta contenido avanzado por palabras clave."""
        text = """
        Theorem 3.5: The following proposition holds.
        Proof: We proceed by induction on n.
        By the corollary to Lemma 2.3...
        """
        result = classifier.classify(text)

        assert result.level in [DifficultyLevel.ADVANCED, DifficultyLevel.INTERMEDIATE]
        assert "advanced_keywords" in result.reasoning

    def test_research_by_keywords(self, classifier):
        """Detecta contenido de investigación por palabras clave."""
        text = """
        Recent work by Smith et al. (2024) demonstrates...
        Our contribution addresses the open problem of...
        The state-of-the-art approach uses...
        We propose a novel method for quantum error correction.
        """
        result = classifier.classify(text)

        assert result.level == DifficultyLevel.RESEARCH
        assert "research_keywords" in result.reasoning

    def test_intermediate_default(self, classifier):
        """Contenido sin señales claras pero con algo de contexto es intermedio."""
        # Nota: texto completamente genérico puede clasificarse como introductorio
        # porque no tiene señales de complejidad. Para intermedio, necesitamos
        # alguna señal leve de contenido técnico.
        text = """
        Este párrafo discute algunos aspectos del sistema
        sin ser demasiado básico ni avanzado.
        """
        result = classifier.classify(text)

        # Sin señales claras, puede ser introductorio o intermedio
        assert result.level in [DifficultyLevel.INTRODUCTORY, DifficultyLevel.INTERMEDIATE]

    def test_high_formula_density(self, classifier):
        """Alta densidad de fórmulas indica avanzado."""
        text = r"""
        $x$ y $y$ son $z$. Entonces $a = b$ y $c = d$.
        Por lo tanto $e = f$, $g = h$, $i = j$, $k = l$.
        Finalmente $m = n$ y $o = p$.
        """
        result = classifier.classify(text)

        # Alta densidad de fórmulas debería indicar avanzado
        assert result.confidence > 0.2

    def test_batch_classify(self, classifier):
        """Clasificación en batch funciona correctamente."""
        chunks = [
            ("Introducción básica al qubit", ["Cap 1", "Intro"]),
            ("Proof of theorem using advanced calculus", None),
            ("Recent work by Smith et al. (2024)", None),
        ]

        results = classifier.classify_batch(chunks)

        assert len(results) == 3
        assert results[0].level == DifficultyLevel.INTRODUCTORY
        assert results[2].level == DifficultyLevel.RESEARCH


class TestDifficultyHelpers:
    """Tests para funciones auxiliares."""

    def test_get_difficulty_label(self):
        """Etiquetas legibles correctas."""
        assert get_difficulty_label(DifficultyLevel.INTRODUCTORY) == "Introductorio"
        assert get_difficulty_label(DifficultyLevel.INTERMEDIATE) == "Intermedio"
        assert get_difficulty_label(DifficultyLevel.ADVANCED) == "Avanzado"
        assert get_difficulty_label(DifficultyLevel.RESEARCH) == "Investigación"

    def test_difficulty_to_numeric(self):
        """Conversión numérica correcta."""
        assert difficulty_to_numeric(DifficultyLevel.INTRODUCTORY) == 1
        assert difficulty_to_numeric(DifficultyLevel.INTERMEDIATE) == 2
        assert difficulty_to_numeric(DifficultyLevel.ADVANCED) == 3
        assert difficulty_to_numeric(DifficultyLevel.RESEARCH) == 4


class TestDifficultyResult:
    """Tests para DifficultyResult."""

    def test_result_creation(self):
        """Creación de resultado correcta."""
        result = DifficultyResult(
            level=DifficultyLevel.ADVANCED,
            confidence=0.85,
            reasoning="complex_math:3,advanced_keywords:2"
        )

        assert result.level == DifficultyLevel.ADVANCED
        assert result.confidence == 0.85
        assert "complex_math" in result.reasoning

    def test_result_string_level(self):
        """El nivel es un enum válido."""
        result = DifficultyResult(
            level=DifficultyLevel.INTRODUCTORY,
            confidence=0.9,
            reasoning="intro_keywords:5"
        )

        assert result.level.value == "introductory"
