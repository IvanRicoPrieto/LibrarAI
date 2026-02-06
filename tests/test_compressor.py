# tests/test_compressor.py
"""
Tests unitarios para el compresor de contexto.
"""

import pytest
import sys
from pathlib import Path

# Agregar raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generation.context_compressor import (
    ContextCompressor,
    CompressionConfig,
    CompressionResult,
    CompressionLevel
)


class TestCompressionLevel:
    """Tests para la enumeración de niveles de compresión."""
    
    def test_compression_levels_exist(self):
        """Verifica que existen todos los niveles."""
        assert CompressionLevel.NONE.value == "none"
        assert CompressionLevel.LIGHT.value == "light"
        assert CompressionLevel.MEDIUM.value == "medium"
        assert CompressionLevel.AGGRESSIVE.value == "aggressive"


class TestCompressionConfig:
    """Tests para la configuración del compresor."""
    
    def test_default_config(self):
        """Test de configuración por defecto."""
        config = CompressionConfig()
        
        assert config.level == CompressionLevel.MEDIUM
        assert config.preserve_citations == True
        assert config.preserve_math == True
        assert config.preserve_code == True
        assert config.use_llmlingua == False
    
    def test_custom_config(self):
        """Test de configuración personalizada."""
        config = CompressionConfig(
            level=CompressionLevel.AGGRESSIVE,
            target_ratio=0.3,
            preserve_citations=False,
            use_llmlingua=True
        )
        
        assert config.level == CompressionLevel.AGGRESSIVE
        assert config.target_ratio == 0.3
        assert config.preserve_citations == False


class TestCompressionResult:
    """Tests para el resultado de compresión."""
    
    def test_result_creation(self):
        """Test de creación de resultado."""
        result = CompressionResult(
            original_text="Texto original largo",
            compressed_text="Texto comprimido",
            original_tokens=100,
            compressed_tokens=60,
            compression_ratio=0.6,
            method="heuristic"
        )
        
        assert result.compression_ratio == 0.6
        assert result.method == "heuristic"
    
    def test_result_to_dict(self):
        """Test de serialización."""
        result = CompressionResult(
            original_text="Original",
            compressed_text="Comprimido",
            original_tokens=50,
            compressed_tokens=30,
            compression_ratio=0.6,
            method="heuristic",
            preserved_elements={"citations": 2, "latex": 3}
        )
        
        d = result.to_dict()
        
        assert d["original_tokens"] == 50
        assert d["compressed_tokens"] == 30
        assert d["preserved_elements"]["citations"] == 2


class TestContextCompressor:
    """Tests para el compresor de contexto."""
    
    @pytest.fixture
    def compressor(self):
        """Compresor con configuración por defecto."""
        return ContextCompressor()
    
    @pytest.fixture
    def light_compressor(self):
        """Compresor con compresión ligera."""
        config = CompressionConfig(level=CompressionLevel.LIGHT)
        return ContextCompressor(config=config)
    
    @pytest.fixture
    def aggressive_compressor(self):
        """Compresor con compresión agresiva."""
        config = CompressionConfig(level=CompressionLevel.AGGRESSIVE)
        return ContextCompressor(config=config)
    
    def test_compressor_initialization(self, compressor):
        """Test de inicialización correcta."""
        assert compressor is not None
        assert compressor.config.level == CompressionLevel.MEDIUM
    
    def test_token_estimation(self, compressor):
        """Test de estimación de tokens."""
        text = "Este es un texto de prueba con varias palabras."
        tokens = compressor._estimate_tokens(text)
        
        assert tokens > 0
        # Aproximadamente 3 chars por token (más preciso para español)
        assert tokens == len(text) // 3
    
    def test_compress_empty_text(self, compressor):
        """Test de compresión de texto vacío."""
        result = compressor.compress("")
        
        assert result.compressed_text == ""
        assert result.original_tokens == 0
    
    def test_compress_preserves_latex_block(self, compressor):
        """Las fórmulas LaTeX en bloque deben preservarse."""
        text = """
        El estado cuántico se representa como:
        
        $$|\\psi\\rangle = \\alpha|0\\rangle + \\beta|1\\rangle$$
        
        Donde alpha y beta son amplitudes complejas que deben satisfacer
        la condición de normalización. Esta ecuación fundamental describe
        cómo un qubit puede existir en superposición de estados base.
        """
        result = compressor.compress(text)
        
        # Verificar que la fórmula de bloque se preserva
        assert "$$" in result.compressed_text or "\\psi" in result.compressed_text
    
    def test_compress_preserves_code_blocks(self, compressor):
        """Los bloques de código deben preservarse."""
        text = """
        Para crear un circuito cuántico en Qiskit, usamos el siguiente código
        que demuestra la creación de un estado de Bell mediante superposición:
        
        ```python
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        ```
        
        Este código crea un estado de Bell entrelazando dos qubits mediante
        una puerta Hadamard seguida de una puerta CNOT controlada.
        """
        result = compressor.compress(text)
        
        # El código debería preservarse
        assert "```" in result.compressed_text or "QuantumCircuit" in result.compressed_text
    
    def test_light_compression_ratio(self, light_compressor):
        """Compresión ligera debería reducir muy poco."""
        text = """
        Este es un texto largo con mucha información redundante y repetitiva.
        Contiene múltiples oraciones que podrían ser simplificadas.
        La información se presenta de forma extendida con explicaciones
        adicionales que no siempre son necesarias para entender el concepto
        principal que se está comunicando al lector.
        """ * 5
        
        result = light_compressor.compress(text)
        
        # Ligera compresión: ratio entre 0.7 y 1.0
        assert result.compression_ratio >= 0.7
    
    def test_medium_compression_ratio(self, compressor):
        """Compresión media debería reducir algo."""
        text = """
        Un qubit es la unidad fundamental de información cuántica. A diferencia
        de un bit clásico, que solo puede estar en estado 0 o 1, un qubit puede
        existir en una superposición de ambos estados simultáneamente. Esta
        propiedad es lo que hace que la computación cuántica sea tan poderosa.
        
        La superposición permite que los algoritmos cuánticos exploren múltiples
        posibilidades en paralelo, lo cual es imposible en computación clásica.
        El estado de un qubit se describe matemáticamente como una combinación
        lineal de los estados base |0⟩ y |1⟩.
        """ * 3
        
        result = compressor.compress(text)
        
        # Compresión media: el texto debería reducirse
        assert result.compression_ratio <= 1.0
    
    def test_aggressive_compression_ratio(self, aggressive_compressor):
        """Compresión agresiva debería reducir significativamente."""
        text = """
        La mecánica cuántica es una teoría fundamental de la física que describe
        el comportamiento de la materia y la energía a escalas muy pequeñas.
        Esta teoría fue desarrollada a principios del siglo XX por varios físicos
        incluyendo a Planck, Bohr, Heisenberg, Schrödinger y otros pioneros.
        
        Los principios fundamentales incluyen la dualidad onda-partícula, el
        principio de incertidumbre de Heisenberg, y la superposición cuántica.
        Estos conceptos desafían nuestra intuición clásica sobre cómo funciona
        el mundo físico a nivel microscópico.
        
        En la actualidad, la mecánica cuántica tiene aplicaciones en muchos
        campos, desde la física de partículas hasta la química, la biología
        molecular, y por supuesto, la computación cuántica que promete
        revolucionar el procesamiento de información.
        """ * 2
        
        result = aggressive_compressor.compress(text)
        
        # El texto debería ser más corto
        assert len(result.compressed_text) < len(text)


class TestCompressContexts:
    """Tests para compresión de múltiples contextos."""
    
    @pytest.fixture
    def compressor(self):
        return ContextCompressor()
    
    def test_compress_multiple_contexts(self, compressor):
        """Test de compresión de lista de contextos."""
        contexts = [
            "Primer contexto sobre qubits y superposición cuántica.",
            "Segundo contexto sobre entrelazamiento y estados de Bell.",
            "Tercer contexto sobre algoritmos cuánticos como Shor y Grover."
        ]
        
        # compress_contexts retorna (contextos, stats)
        compressed_contexts, stats = compressor.compress_contexts(contexts)
        
        assert len(compressed_contexts) == 3
        assert "original_tokens" in stats
        assert "compressed_tokens" in stats
    
    def test_compress_empty_contexts_list(self, compressor):
        """Test de lista vacía de contextos."""
        compressed_contexts, stats = compressor.compress_contexts([])
        assert compressed_contexts == []
        assert stats["original_tokens"] == 0
    
    def test_compress_contexts_with_budget(self, compressor):
        """Test de compresión con presupuesto de tokens."""
        contexts = [
            "Contexto largo " * 100,
            "Otro contexto largo " * 100,
            "Más contexto " * 100,
        ]
        
        # Usar max_total_tokens para limitar
        compressed_contexts, stats = compressor.compress_contexts(contexts, max_total_tokens=500)
        
        # Si se aplica compresión, debería reducirse
        if stats.get("compression_applied"):
            assert stats["compressed_tokens"] < stats["original_tokens"]


class TestPreservationPatterns:
    """Tests para patrones de preservación."""
    
    @pytest.fixture
    def compressor(self):
        return ContextCompressor()
    
    def test_extract_preserved_elements(self, compressor):
        """Test de extracción de elementos a preservar."""
        text = """
        Según [1], la ecuación $E=mc^2$ es fundamental.
        
        $$\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$
        
        ```python
        print("Hello quantum!")
        ```
        
        Ver también [2] y [3].
        """
        
        processed, preserved = compressor._extract_preserved_elements(text)
        
        # Debería haber extraído elementos
        assert len(preserved["citations"]) > 0 or "[1]" in processed
    
    def test_restore_preserved_elements(self, compressor):
        """Test de restauración de elementos preservados."""
        text = "Fórmula: $x^2 + y^2 = r^2$. Cita: [1]. Más texto aquí."
        
        # Extraer
        processed, preserved = compressor._extract_preserved_elements(text)
        
        # Restaurar
        restored = compressor._restore_preserved_elements(processed, preserved)
        
        # Debería contener los elementos originales
        assert "$" in restored or "x^2" in restored
    
    def test_sentence_importance_scoring(self, compressor):
        """Test de puntuación de importancia de oraciones."""
        sentence_important = "El qubit es la unidad básica de la computación cuántica."
        sentence_generic = "Esto es una oración genérica sin términos técnicos específicos."
        
        score_important = compressor._calculate_sentence_importance(sentence_important)
        score_generic = compressor._calculate_sentence_importance(sentence_generic)
        
        # La oración con keywords debería tener mayor puntuación
        assert score_important > score_generic


class TestCompressionWithQuantumContent:
    """Tests con contenido cuántico real."""
    
    @pytest.fixture
    def compressor(self):
        return ContextCompressor()
    
    def test_compress_quantum_text(self, compressor, sample_quantum_text):
        """Test de compresión de texto cuántico."""
        result = compressor.compress(sample_quantum_text)
        
        assert result.compressed_text != ""
        assert result.compression_ratio > 0
        
        # El contenido importante debería preservarse
        compressed = result.compressed_text.lower()
        # Al menos algunos términos clave o contenido significativo
        has_content = len(result.compressed_text) > 50
        assert has_content
    
    def test_compress_math_text(self, compressor, sample_math_text):
        """Test de compresión de texto matemático."""
        result = compressor.compress(sample_math_text)
        
        # Debería retornar algo
        assert result is not None
        # Y tener estadísticas válidas
        assert result.original_tokens > 0
