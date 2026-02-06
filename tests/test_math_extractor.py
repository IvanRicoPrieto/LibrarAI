"""
Tests para el extractor de términos matemáticos.
"""

import pytest
from src.ingestion.math_extractor import (
    MathExtractor,
    MathExtractionResult,
    normalize_math_query,
    formula_to_description,
)


class TestMathExtractor:
    """Tests para MathExtractor."""

    @pytest.fixture
    def extractor(self):
        """Extractor por defecto."""
        return MathExtractor()

    def test_extract_sum_terms(self, extractor):
        """Extrae términos de sumatorio."""
        text = r"El sumatorio $\sum_{i=1}^n x_i$ representa la suma total."
        result = extractor.extract(text)

        assert "sumatorio" in result.terms or "summation" in result.terms
        assert len(result.formulas) > 0

    def test_extract_integral_terms(self, extractor):
        """Extrae términos de integral."""
        text = r"La integral $\int_0^\infty e^{-x} dx$ converge."
        result = extractor.extract(text)

        assert "integral" in result.terms or "integration" in result.terms

    def test_extract_quantum_terms(self, extractor):
        """Extrae términos cuánticos."""
        text = r"El estado $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ está en superposición."
        result = extractor.extract(text)

        # Debe detectar notación de Dirac
        assert any("ket" in t.lower() or "dirac" in t.lower() or "estado" in t.lower()
                   for t in result.terms + result.concepts)

    def test_extract_matrix_terms(self, extractor):
        """Extrae términos de matrices."""
        text = r"El determinante $\det(A)$ y la traza $\Tr(B)$ son invariantes."
        result = extractor.extract(text)

        assert "determinante" in result.terms or "determinant" in result.terms
        assert "traza" in result.terms or "trace" in result.terms

    def test_extract_tensor_product(self, extractor):
        """Extrae término de producto tensorial."""
        text = r"El estado compuesto es $|\psi\rangle \otimes |\phi\rangle$."
        result = extractor.extract(text)

        assert "producto tensorial" in result.terms or "tensor product" in result.terms

    def test_extract_concept_entanglement(self, extractor):
        """Detecta concepto de entrelazamiento."""
        text = "Los qubits están entangled, mostrando entrelazamiento cuántico."
        result = extractor.extract(text)

        assert "entrelazamiento" in result.concepts or "entanglement" in result.concepts

    def test_extract_concept_superposition(self, extractor):
        """Detecta concepto de superposición."""
        text = "El qubit está en superposición de estados base."
        result = extractor.extract(text)

        assert "superposición" in result.concepts or "superposition" in result.concepts

    def test_extract_eigenvalues(self, extractor):
        """Detecta concepto de autovalores."""
        text = "Los eigenvalues del operador hamiltoniano determinan las energías."
        result = extractor.extract(text)

        assert "eigenvalue" in result.concepts or "autovalor" in result.concepts

    def test_extract_quantum_algorithms(self, extractor):
        """Detecta algoritmos cuánticos."""
        text = "El algoritmo de Shor permite factorizar eficientemente."
        result = extractor.extract(text)

        assert any("shor" in t.lower() for t in result.concepts)

    def test_extract_grover_algorithm(self, extractor):
        """Detecta algoritmo de Grover."""
        text = "Grover's algorithm provides quadratic speedup for search."
        result = extractor.extract(text)

        assert any("grover" in t.lower() for t in result.concepts)

    def test_extract_greek_symbols(self, extractor):
        """Extrae símbolos griegos."""
        text = r"Sea $\alpha$, $\beta$, $\gamma$ los ángulos y $\psi$, $\phi$ los estados."
        result = extractor.extract(text)

        assert "alpha" in result.symbols or "psi" in result.symbols

    def test_extract_formulas(self, extractor):
        """Extrae fórmulas LaTeX correctamente."""
        text = r"Inline $a^2 + b^2 = c^2$ y display $$E = mc^2$$ son importantes."
        result = extractor.extract(text)

        assert len(result.formulas) >= 2
        assert any("c^2" in f for f in result.formulas)

    def test_empty_text(self, extractor):
        """Texto vacío no causa errores."""
        result = extractor.extract("")

        assert result.terms == []
        assert result.formulas == []
        assert result.concepts == []

    def test_text_without_math(self, extractor):
        """Texto sin matemáticas devuelve listas vacías."""
        text = "Este es un texto simple sin contenido matemático."
        result = extractor.extract(text)

        # Puede detectar algunos conceptos pero no fórmulas
        assert len(result.formulas) == 0

    def test_batch_extract(self, extractor):
        """Extracción en batch funciona."""
        texts = [
            r"$\sum_i x_i$",
            r"$\int f(x) dx$",
            "Texto sin matemáticas",
        ]
        results = extractor.extract_batch(texts)

        assert len(results) == 3
        assert len(results[0].terms) > 0
        assert len(results[1].terms) > 0

    def test_get_searchable_text(self, extractor):
        """Genera texto enriquecido para búsqueda."""
        text = r"El producto tensorial $A \otimes B$ es importante."
        enriched = extractor.get_searchable_text(text)

        assert text in enriched
        assert "Términos:" in enriched
        # Debe contener algún término matemático
        assert "tensorial" in enriched.lower() or "tensor" in enriched.lower()

    def test_max_terms_limit(self):
        """Respeta límite máximo de términos."""
        extractor = MathExtractor(max_terms=5)
        text = r"""
        $\sum$ $\int$ $\nabla$ $\partial$ $\det$ $\Tr$ $\otimes$ $\oplus$
        entrelazamiento superposición eigenvalue hermitiano unitario
        """
        result = extractor.extract(text)

        assert len(result.terms) <= 5


class TestNormalizeMathQuery:
    """Tests para normalización de queries."""

    def test_normalize_sumatorio(self):
        """Normaliza 'sumatorio' a términos equivalentes."""
        query = "buscar sumatorio"
        normalized = normalize_math_query(query)

        assert "summation" in normalized
        assert "sum" in normalized or "sigma" in normalized

    def test_normalize_integral(self):
        """Normaliza 'integral' a términos equivalentes."""
        query = "encontrar integral"
        normalized = normalize_math_query(query)

        assert "integration" in normalized or "integral" in normalized

    def test_normalize_autovalor(self):
        """Normaliza 'autovalor' a términos equivalentes."""
        query = "calcular autovalor"
        normalized = normalize_math_query(query)

        assert "eigenvalue" in normalized

    def test_normalize_entrelazamiento(self):
        """Normaliza 'entrelazamiento' a términos equivalentes."""
        query = "qué es entrelazamiento"
        normalized = normalize_math_query(query)

        assert "entanglement" in normalized

    def test_no_change_without_keywords(self):
        """No cambia queries sin palabras clave matemáticas."""
        query = "texto simple sin matemáticas"
        normalized = normalize_math_query(query)

        assert normalized == query


class TestFormulaToDescription:
    """Tests para descripción de fórmulas."""

    def test_describe_sum(self):
        """Describe sumatorio."""
        formula = r"\sum_{i=1}^{n} x_i"
        desc = formula_to_description(formula)

        assert "sumatorio" in desc

    def test_describe_integral(self):
        """Describe integral."""
        formula = r"\int_0^\infty f(x) dx"
        desc = formula_to_description(formula)

        assert "integral" in desc

    def test_describe_fraction(self):
        """Describe fracción."""
        formula = r"\frac{a}{b}"
        desc = formula_to_description(formula)

        assert "fracción" in desc

    def test_describe_sqrt(self):
        """Describe raíz cuadrada."""
        formula = r"\sqrt{x}"
        desc = formula_to_description(formula)

        assert "raíz" in desc

    def test_describe_limit(self):
        """Describe límite."""
        formula = r"\lim_{x \to 0} f(x)"
        desc = formula_to_description(formula)

        assert "límite" in desc

    def test_describe_unknown(self):
        """Formula desconocida devuelve descripción genérica."""
        formula = "xyz"
        desc = formula_to_description(formula)

        assert "expresión" in desc


class TestMathExtractionResult:
    """Tests para MathExtractionResult."""

    def test_result_creation(self):
        """Creación de resultado correcta."""
        result = MathExtractionResult(
            terms=["integral", "sumatorio"],
            formulas=[r"$\int f$", r"$\sum x$"],
            symbols=["alpha", "beta"],
            concepts=["entrelazamiento"]
        )

        assert len(result.terms) == 2
        assert len(result.formulas) == 2
        assert len(result.symbols) == 2
        assert len(result.concepts) == 1
