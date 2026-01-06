"""
LibrarAI - Sistema RAG para biblioteca de computación cuántica.

Módulos:
- ingestion: Parsing, chunking e indexación
- retrieval: Búsqueda vectorial, BM25 y grafo
- generation: Generación de respuestas con citas
- agents: Router, planner y critic
- cli: Interfaz de línea de comandos
"""

__version__ = "1.0.0"
__author__ = "Quantum Computing Master - UNIR"

from pathlib import Path

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
INDICES_DIR = PROJECT_ROOT / "indices"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
