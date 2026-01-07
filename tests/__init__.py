# tests/__init__.py
"""
LibrarAI Test Suite.

Estructura de tests:
- test_chunker.py: Tests unitarios del chunker jerárquico
- test_fusion.py: Tests de fusión RRF y re-ranking
- test_cache.py: Tests del sistema de caché de embeddings
- test_compressor.py: Tests de compresión de contexto
- test_integration.py: Tests de integración end-to-end
- conftest.py: Fixtures compartidos

Ejecutar todos los tests:
    pytest tests/ -v

Ejecutar con cobertura:
    pytest tests/ --cov=src --cov-report=html

Ejecutar tests específicos:
    pytest tests/test_chunker.py -v
"""
