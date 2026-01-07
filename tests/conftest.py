# tests/conftest.py
"""
Pytest fixtures compartidos para LibrarAI tests.
"""

import pytest
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_quantum_text():
    """Texto de ejemplo sobre computación cuántica."""
    return """
# Introducción a los Qubits

Un **qubit** es la unidad básica de información cuántica. A diferencia de un bit 
clásico que solo puede estar en estado 0 o 1, un qubit puede estar en una 
superposición de ambos estados.

## Estado de un Qubit

El estado general de un qubit se representa como:

$$|\\psi\\rangle = \\alpha|0\\rangle + \\beta|1\\rangle$$

Donde $\\alpha$ y $\\beta$ son amplitudes complejas que satisfacen la condición 
de normalización $|\\alpha|^2 + |\\beta|^2 = 1$.

## Esfera de Bloch

La esfera de Bloch proporciona una representación geométrica del estado de un 
qubit. Cualquier estado puro puede escribirse como:

$$|\\psi\\rangle = \\cos(\\theta/2)|0\\rangle + e^{i\\phi}\\sin(\\theta/2)|1\\rangle$$

Donde $\\theta$ y $\\phi$ son las coordenadas esféricas.

## Puertas Cuánticas

Las puertas cuánticas transforman el estado de los qubits. Las puertas más 
comunes son:

- **Pauli-X**: Equivalente al NOT clásico
- **Hadamard (H)**: Crea superposición
- **CNOT**: Puerta de dos qubits para entrelazamiento

```python
# Ejemplo con Qiskit
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)  # Hadamard en qubit 0
qc.cx(0, 1)  # CNOT
```

Teorema: Todo algoritmo cuántico puede descomponerse en puertas universales.
"""


@pytest.fixture
def sample_math_text():
    """Texto de ejemplo con contenido matemático."""
    return """
# Espacios de Hilbert

## Definición

Un espacio de Hilbert $\\mathcal{H}$ es un espacio vectorial completo con 
producto interno.

**Definición 1.1**: Sea $V$ un espacio vectorial sobre $\\mathbb{C}$. Un 
producto interno es una función $\\langle \\cdot, \\cdot \\rangle: V \\times V \\to \\mathbb{C}$ 
que satisface:

1. Linealidad: $\\langle a u + b v, w \\rangle = a\\langle u, w \\rangle + b\\langle v, w \\rangle$
2. Conjugación: $\\langle u, v \\rangle = \\overline{\\langle v, u \\rangle}$
3. Positividad: $\\langle u, u \\rangle \\geq 0$ con igualdad sii $u = 0$

**Teorema 1.2** (Desigualdad de Cauchy-Schwarz): Para todo $u, v \\in \\mathcal{H}$:

$$|\\langle u, v \\rangle|^2 \\leq \\langle u, u \\rangle \\cdot \\langle v, v \\rangle$$

**Demostración**: Sea $t \\in \\mathbb{R}$ y consideremos $\\langle u + tv, u + tv \\rangle \\geq 0$...
"""


@pytest.fixture
def sample_chunks():
    """Lista de chunks de ejemplo para tests de fusión."""
    return [
        {
            "chunk_id": "chunk_001",
            "content": "Un qubit es la unidad básica de información cuántica.",
            "doc_id": "doc_001",
            "doc_title": "Intro Cuántica",
            "header_path": "Capítulo 1 > Qubits",
            "level": "MICRO",
            "token_count": 15,
        },
        {
            "chunk_id": "chunk_002",
            "content": "La superposición permite que un qubit esté en múltiples estados simultáneamente.",
            "doc_id": "doc_001",
            "doc_title": "Intro Cuántica",
            "header_path": "Capítulo 1 > Superposición",
            "level": "MICRO",
            "token_count": 18,
        },
        {
            "chunk_id": "chunk_003",
            "content": "El entrelazamiento cuántico es un fenómeno que conecta partículas de forma no clásica.",
            "doc_id": "doc_002",
            "doc_title": "Fenómenos Cuánticos",
            "header_path": "Capítulo 2 > Entrelazamiento",
            "level": "MICRO",
            "token_count": 20,
        },
        {
            "chunk_id": "chunk_004",
            "content": "BB84 es un protocolo de distribución de claves cuánticas propuesto por Bennett y Brassard.",
            "doc_id": "doc_003",
            "doc_title": "QKD Protocols",
            "header_path": "Capítulo 1 > BB84",
            "level": "MICRO",
            "token_count": 22,
        },
        {
            "chunk_id": "chunk_005",
            "content": "El algoritmo de Shor factoriza números enteros en tiempo polinómico.",
            "doc_id": "doc_004",
            "doc_title": "Algoritmos Cuánticos",
            "header_path": "Capítulo 3 > Shor",
            "level": "MICRO",
            "token_count": 16,
        }
    ]


@pytest.fixture
def temp_dir():
    """Directorio temporal para tests que necesitan persistencia."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Mock Objects
# ============================================================================

@dataclass
class MockVectorResult:
    """Mock de resultado de búsqueda vectorial."""
    chunk_id: str
    content: str
    score: float
    doc_id: str = "doc_001"
    doc_title: str = "Test Doc"
    header_path: str = "Test > Path"
    parent_id: Optional[str] = None
    level: str = "MICRO"
    token_count: int = 50


@dataclass  
class MockBM25Result:
    """Mock de resultado de búsqueda BM25."""
    chunk_id: str
    content: str
    score: float
    doc_id: str = "doc_001"
    doc_title: str = "Test Doc"
    header_path: str = "Test > Path"
    parent_id: Optional[str] = None
    level: str = "MICRO"
    token_count: int = 50


@dataclass
class MockGraphResult:
    """Mock de resultado de búsqueda de grafo."""
    chunk_id: str
    content: str
    score: float
    doc_id: str = "doc_001"
    doc_title: str = "Test Doc"
    header_path: str = "Test > Path"
    parent_id: Optional[str] = None
    level: str = "MICRO"
    token_count: int = 50
    related_concepts: List[str] = None
    
    def __post_init__(self):
        if self.related_concepts is None:
            self.related_concepts = []


@pytest.fixture
def mock_vector_results():
    """Resultados mock de búsqueda vectorial."""
    return [
        MockVectorResult("chunk_001", "Contenido sobre qubits", 0.92),
        MockVectorResult("chunk_002", "Contenido sobre superposición", 0.88),
        MockVectorResult("chunk_003", "Contenido sobre entrelazamiento", 0.85),
    ]


@pytest.fixture
def mock_bm25_results():
    """Resultados mock de búsqueda BM25."""
    return [
        MockBM25Result("chunk_002", "Contenido sobre superposición", 8.5),
        MockBM25Result("chunk_004", "Contenido sobre BB84", 7.2),
        MockBM25Result("chunk_001", "Contenido sobre qubits", 6.8),
    ]


@pytest.fixture
def mock_graph_results():
    """Resultados mock de búsqueda de grafo."""
    return [
        MockGraphResult("chunk_003", "Contenido sobre entrelazamiento", 0.78, 
                       related_concepts=["qubit", "bell_state"]),
        MockGraphResult("chunk_005", "Contenido sobre Shor", 0.65,
                       related_concepts=["factorization", "quantum_algorithm"]),
    ]


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def cache_config():
    """Configuración de caché para tests."""
    return {
        "memory_cache_size": 100,
        "persistent": True,
        "max_age_seconds": 3600,
        "track_stats": True
    }


@pytest.fixture
def fusion_config():
    """Configuración de fusión para tests."""
    return {
        "k": 60,
        "vector_weight": 0.5,
        "bm25_weight": 0.3,
        "graph_weight": 0.2
    }
