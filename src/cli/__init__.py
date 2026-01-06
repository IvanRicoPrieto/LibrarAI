"""
Módulo CLI - Interfaz de línea de comandos.

Comandos principales:
- ask_library: Consultar la biblioteca
- ingest_library: Indexar documentos
"""

from .ask_library import main as ask_main
from .ingest_library import main as ingest_main

__all__ = ["ask_main", "ingest_main"]
