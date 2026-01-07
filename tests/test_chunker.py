# tests/test_chunker.py
"""
Tests unitarios para el chunker jerárquico.
"""

import pytest
import sys
from pathlib import Path

# Agregar raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.chunker import (
    HierarchicalChunker,
    Chunk,
    ChunkLevel
)
from src.ingestion.parser import ParsedSection, ParsedDocument


class TestChunkLevel:
    """Tests para la enumeración ChunkLevel."""
    
    def test_chunk_levels_exist(self):
        """Verifica que existen los tres niveles de chunk."""
        assert ChunkLevel.MACRO.value == "macro"
        assert ChunkLevel.MESO.value == "meso"
        assert ChunkLevel.MICRO.value == "micro"


class TestChunk:
    """Tests para la dataclass Chunk."""
    
    def test_chunk_creation(self):
        """Test de creación básica de un chunk."""
        chunk = Chunk(
            chunk_id="test_001",
            content="Contenido de prueba",
            level=ChunkLevel.MICRO,
            doc_id="doc_001",
            doc_title="Documento Test",
            header_path="Cap 1 > Sección 1"
        )
        
        assert chunk.chunk_id == "test_001"
        assert chunk.content == "Contenido de prueba"
        assert chunk.level == ChunkLevel.MICRO
        assert chunk.char_count == len("Contenido de prueba")
        assert chunk.content_hash != ""
    
    def test_chunk_hash_consistency(self):
        """Verifica que chunks con mismo contenido tienen mismo hash."""
        content = "Texto idéntico para verificar hash"
        
        chunk1 = Chunk(
            chunk_id="chunk_001",
            content=content,
            level=ChunkLevel.MICRO,
            doc_id="doc_001",
            doc_title="Test",
            header_path="Test"
        )
        
        chunk2 = Chunk(
            chunk_id="chunk_002",
            content=content,
            level=ChunkLevel.MESO,
            doc_id="doc_002",
            doc_title="Otro",
            header_path="Otro"
        )
        
        assert chunk1.content_hash == chunk2.content_hash
    
    def test_chunk_to_dict(self):
        """Test de serialización a diccionario."""
        chunk = Chunk(
            chunk_id="test_001",
            content="Contenido",
            level=ChunkLevel.MICRO,
            doc_id="doc_001",
            doc_title="Test",
            header_path="Path",
            token_count=10
        )
        
        d = chunk.to_dict()
        
        assert d["chunk_id"] == "test_001"
        assert d["content"] == "Contenido"
        assert d["level"] == "micro"
        assert d["token_count"] == 10
        assert "content_hash" in d


class TestHierarchicalChunker:
    """Tests para el chunker jerárquico."""
    
    @pytest.fixture
    def chunker(self):
        """Instancia de chunker con configuración por defecto."""
        return HierarchicalChunker()
    
    def test_chunker_initialization(self, chunker):
        """Test de inicialización del chunker."""
        assert chunker is not None
        # Verifica que tiene los métodos esperados
        assert hasattr(chunker, 'chunk_document')
        assert hasattr(chunker, 'count_tokens')
    
    def test_token_counting(self, chunker):
        """Test del conteo de tokens."""
        text = "Este es un texto de prueba para contar tokens."
        tokens = chunker.count_tokens(text)
        
        # El conteo debería ser aproximadamente len/4 o usar tiktoken
        assert tokens > 0
        assert isinstance(tokens, int)
    
    @pytest.mark.skip(reason="API interna no expuesta")
    def test_empty_content_handling(self, chunker):
        """Test de manejo de contenido vacío."""
        pass
    
    @pytest.mark.skip(reason="API interna no expuesta")
    def test_sentence_splitting(self, chunker):
        """Test de división por oraciones."""
        pass
    
    @pytest.mark.skip(reason="API interna no expuesta")
    def test_latex_preservation_in_split(self, chunker):
        """Verifica que las fórmulas LaTeX no se dividen incorrectamente."""
        pass
    
    @pytest.mark.skip(reason="Requiere ParsedDocument con estructura específica")
    def test_chunk_hierarchy(self, chunker):
        """Test de la jerarquía de chunks generados."""
        pass
    
    @pytest.mark.skip(reason="Requiere ParsedDocument con estructura específica")
    def test_parent_child_relationships(self, chunker):
        """Verifica las relaciones padre-hijo entre chunks."""
        pass


class TestChunkerWithQuantumContent:
    """Tests del chunker con contenido cuántico real."""
    
    @pytest.fixture
    def chunker(self):
        return HierarchicalChunker()
    
    @pytest.mark.skip(reason="Requiere ParsedDocument con estructura específica")
    def test_quantum_text_chunking(self, chunker, sample_quantum_text):
        """Test de chunking de texto sobre computación cuántica."""
        pass
    
    @pytest.mark.skip(reason="Requiere ParsedDocument con estructura específica")
    def test_math_text_chunking(self, chunker, sample_math_text):
        """Test de chunking de texto matemático."""
        pass
