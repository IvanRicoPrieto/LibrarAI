"""
Chunker Jerárquico con Auto-Merge.

Este módulo implementa la estrategia de chunking de 3 niveles:
- Macro (2048-4096 tokens): Secciones/subcapítulos completos
- Meso (512 tokens): Párrafos relacionados
- Micro (128-256 tokens): Definiciones, teoremas, fórmulas

El sistema indexa chunks Micro pero mantiene referencias a padres
para poder hacer auto-merge cuando sea necesario.
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
import logging

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from .parser import ParsedSection, ParsedDocument

logger = logging.getLogger(__name__)


class ChunkLevel(Enum):
    """Niveles de granularidad de chunks."""
    MACRO = "macro"   # 2048-4096 tokens
    MESO = "meso"     # 512 tokens
    MICRO = "micro"   # 128-256 tokens


@dataclass
class Chunk:
    """Representa un fragmento de texto con metadatos."""
    chunk_id: str
    content: str
    level: ChunkLevel
    
    # Metadatos de ubicación
    doc_id: str
    doc_title: str
    header_path: str
    
    # Jerarquía
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    # Posición en documento original
    start_char: int = 0
    end_char: int = 0
    
    # Estadísticas
    token_count: int = 0
    char_count: int = 0
    
    # Hash para deduplicación
    content_hash: str = field(default="")
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.content.encode()
            ).hexdigest()[:16]
        self.char_count = len(self.content)
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario para serialización."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "level": self.level.value,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "header_path": self.header_path,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "content_hash": self.content_hash
        }


class HierarchicalChunker:
    """
    Chunker que crea fragmentos en 3 niveles jerárquicos.
    
    Estrategia:
    1. Divide el documento en chunks Macro (secciones)
    2. Subdivide cada Macro en chunks Meso (párrafos)
    3. Subdivide cada Meso en chunks Micro (oraciones/definiciones)
    4. Mantiene referencias padre-hijo para auto-merge
    """
    
    # Separadores por prioridad
    DEFAULT_SEPARATORS = [
        "\n## ",      # H2
        "\n### ",     # H3
        "\n#### ",    # H4
        "\n\n",       # Párrafos
        "\n",         # Líneas
        ". ",         # Oraciones
        " ",          # Palabras
    ]
    
    def __init__(
        self,
        micro_size: int = 200,
        meso_size: int = 512,
        macro_size: int = 2048,
        overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        """
        Args:
            micro_size: Tamaño objetivo para chunks Micro (tokens)
            meso_size: Tamaño objetivo para chunks Meso (tokens)
            macro_size: Tamaño objetivo para chunks Macro (tokens)
            overlap: Solapamiento entre chunks adyacentes (tokens)
            encoding_name: Nombre del encoding de tiktoken
        """
        self.micro_size = micro_size
        self.meso_size = meso_size
        self.macro_size = macro_size
        self.overlap = overlap
        
        # Inicializar tokenizador
        if TIKTOKEN_AVAILABLE:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        else:
            self.tokenizer = None
            logger.warning(
                "tiktoken no disponible, usando estimación de tokens"
            )
        
        # Contadores para IDs únicos
        self._chunk_counter = 0
        
        # Cache de hashes para deduplicación
        self._seen_hashes: Set[str] = set()
    
    def count_tokens(self, text: str) -> int:
        """Cuenta tokens en un texto."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Estimación: ~4 caracteres por token en inglés/español
            return len(text) // 4
    
    def chunk_document(self, document: ParsedDocument) -> List[Chunk]:
        """
        Procesa un documento completo y genera chunks jerárquicos.
        
        Args:
            document: Documento parseado
            
        Returns:
            Lista de todos los chunks (todos los niveles)
        """
        all_chunks = []
        self._seen_hashes.clear()
        
        for section in document.sections:
            section_chunks = self._chunk_section(section)
            all_chunks.extend(section_chunks)
        
        logger.info(
            f"Documento '{document.doc_title}': "
            f"{len(all_chunks)} chunks generados"
        )
        
        return all_chunks
    
    def _chunk_section(self, section: ParsedSection) -> List[Chunk]:
        """
        Divide una sección en chunks de los 3 niveles.
        
        Args:
            section: Sección parseada
            
        Returns:
            Lista de chunks de todos los niveles
        """
        chunks = []
        content = section.content
        
        # Nivel MACRO: la sección completa (si no es demasiado grande)
        macro_chunks = self._split_by_size(
            content, 
            self.macro_size,
            ChunkLevel.MACRO,
            section
        )
        
        for macro in macro_chunks:
            # Deduplicar
            if macro.content_hash in self._seen_hashes:
                continue
            self._seen_hashes.add(macro.content_hash)
            
            chunks.append(macro)
            
            # Nivel MESO: subdividir el macro
            meso_chunks = self._split_by_size(
                macro.content,
                self.meso_size,
                ChunkLevel.MESO,
                section,
                parent_id=macro.chunk_id
            )
            
            for meso in meso_chunks:
                if meso.content_hash in self._seen_hashes:
                    continue
                self._seen_hashes.add(meso.content_hash)
                
                chunks.append(meso)
                macro.children_ids.append(meso.chunk_id)
                
                # Nivel MICRO: subdividir el meso
                micro_chunks = self._split_by_size(
                    meso.content,
                    self.micro_size,
                    ChunkLevel.MICRO,
                    section,
                    parent_id=meso.chunk_id
                )
                
                for micro in micro_chunks:
                    if micro.content_hash in self._seen_hashes:
                        continue
                    self._seen_hashes.add(micro.content_hash)
                    
                    chunks.append(micro)
                    meso.children_ids.append(micro.chunk_id)
        
        return chunks
    
    def _split_by_size(
        self,
        text: str,
        target_size: int,
        level: ChunkLevel,
        section: ParsedSection,
        parent_id: Optional[str] = None
    ) -> List[Chunk]:
        """
        Divide texto en chunks del tamaño objetivo.
        
        Args:
            text: Texto a dividir
            target_size: Tamaño objetivo en tokens
            level: Nivel del chunk
            section: Sección original (para metadatos)
            parent_id: ID del chunk padre
            
        Returns:
            Lista de chunks
        """
        chunks = []
        
        # Si el texto cabe en el tamaño objetivo, devolver como está
        token_count = self.count_tokens(text)
        if token_count <= target_size:
            chunk = self._create_chunk(
                text, level, section, parent_id, token_count
            )
            return [chunk]
        
        # Dividir por separadores
        parts = self._split_with_separators(text, target_size)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            part_tokens = self.count_tokens(part)
            chunk = self._create_chunk(
                part, level, section, parent_id, part_tokens
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_with_separators(
        self, 
        text: str, 
        target_size: int
    ) -> List[str]:
        """
        Divide texto usando separadores, respetando bloques atómicos.
        
        Args:
            text: Texto a dividir
            target_size: Tamaño objetivo en tokens
            
        Returns:
            Lista de fragmentos
        """
        # Proteger bloques de código y fórmulas
        protected_blocks = {}
        counter = 0
        
        # Proteger bloques de código
        def protect_block(match):
            nonlocal counter
            key = f"__PROTECTED_BLOCK_{counter}__"
            protected_blocks[key] = match.group(0)
            counter += 1
            return key
        
        protected_text = re.sub(
            r'```[\s\S]*?```', 
            protect_block, 
            text
        )
        protected_text = re.sub(
            r'\$\$[\s\S]*?\$\$', 
            protect_block, 
            protected_text
        )
        
        # Dividir por separadores
        parts = self._recursive_split(protected_text, target_size, 0)
        
        # Restaurar bloques protegidos
        result = []
        for part in parts:
            for key, value in protected_blocks.items():
                part = part.replace(key, value)
            result.append(part)
        
        return result
    
    def _recursive_split(
        self, 
        text: str, 
        target_size: int,
        sep_index: int
    ) -> List[str]:
        """
        División recursiva usando separadores por prioridad.
        
        Args:
            text: Texto a dividir
            target_size: Tamaño objetivo
            sep_index: Índice del separador actual
            
        Returns:
            Lista de fragmentos
        """
        if sep_index >= len(self.DEFAULT_SEPARATORS):
            # No más separadores, devolver como está
            return [text]
        
        separator = self.DEFAULT_SEPARATORS[sep_index]
        
        # Dividir por el separador actual
        if separator in text:
            parts = text.split(separator)
        else:
            # Probar siguiente separador
            return self._recursive_split(text, target_size, sep_index + 1)
        
        # Recombinar partes para alcanzar tamaño objetivo
        result = []
        current = ""
        
        for i, part in enumerate(parts):
            # Añadir separador excepto para la primera parte
            if i > 0:
                part = separator + part
            
            potential = current + part
            potential_tokens = self.count_tokens(potential)
            
            if potential_tokens <= target_size:
                current = potential
            else:
                if current:
                    result.append(current)
                
                # Si la parte individual es muy grande, subdividir
                part_tokens = self.count_tokens(part)
                if part_tokens > target_size:
                    sub_parts = self._recursive_split(
                        part, target_size, sep_index + 1
                    )
                    result.extend(sub_parts[:-1])
                    current = sub_parts[-1] if sub_parts else ""
                else:
                    current = part
        
        if current:
            result.append(current)
        
        return result
    
    def _create_chunk(
        self,
        content: str,
        level: ChunkLevel,
        section: ParsedSection,
        parent_id: Optional[str],
        token_count: int
    ) -> Chunk:
        """Crea un nuevo chunk con ID único."""
        self._chunk_counter += 1
        chunk_id = f"{section.doc_id}_{level.value}_{self._chunk_counter:06d}"
        
        return Chunk(
            chunk_id=chunk_id,
            content=content,
            level=level,
            doc_id=section.doc_id,
            doc_title=section.doc_title,
            header_path=section.header_path,
            parent_id=parent_id,
            start_char=section.start_char,
            end_char=section.end_char,
            token_count=token_count
        )
    
    def get_micro_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Filtra solo los chunks de nivel Micro (para indexación)."""
        return [c for c in chunks if c.level == ChunkLevel.MICRO]
    
    def get_parent_chunk(
        self, 
        chunk: Chunk, 
        all_chunks: List[Chunk]
    ) -> Optional[Chunk]:
        """Obtiene el chunk padre de un chunk dado."""
        if not chunk.parent_id:
            return None
        
        for c in all_chunks:
            if c.chunk_id == chunk.parent_id:
                return c
        return None
    
    def auto_merge(
        self,
        retrieved_chunks: List[Chunk],
        all_chunks: List[Chunk],
        threshold: float = 0.5
    ) -> List[Chunk]:
        """
        Implementa Auto-Merge: si >threshold de hijos de un padre
        están en los recuperados, devuelve el padre.
        
        Args:
            retrieved_chunks: Chunks recuperados por la búsqueda
            all_chunks: Todos los chunks del documento
            threshold: Umbral de hijos necesarios para merge
            
        Returns:
            Lista de chunks (posiblemente con padres en lugar de hijos)
        """
        # Mapear chunk_id a chunk
        chunk_map = {c.chunk_id: c for c in all_chunks}
        retrieved_ids = {c.chunk_id for c in retrieved_chunks}
        
        # Contar hijos recuperados por padre
        parent_child_count: Dict[str, int] = {}
        parent_total_children: Dict[str, int] = {}
        
        for chunk in retrieved_chunks:
            if chunk.parent_id and chunk.parent_id in chunk_map:
                parent = chunk_map[chunk.parent_id]
                parent_child_count[parent.chunk_id] = \
                    parent_child_count.get(parent.chunk_id, 0) + 1
                parent_total_children[parent.chunk_id] = \
                    len(parent.children_ids)
        
        # Determinar qué padres hacer merge
        parents_to_merge = set()
        for parent_id, count in parent_child_count.items():
            total = parent_total_children.get(parent_id, 1)
            if count / total >= threshold:
                parents_to_merge.add(parent_id)
        
        # Construir resultado
        result = []
        seen_parents = set()
        
        for chunk in retrieved_chunks:
            if chunk.parent_id in parents_to_merge:
                if chunk.parent_id not in seen_parents:
                    result.append(chunk_map[chunk.parent_id])
                    seen_parents.add(chunk.parent_id)
            else:
                result.append(chunk)
        
        return result


if __name__ == "__main__":
    # Test básico
    from .parser import MarkdownParser
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        
        parser = MarkdownParser()
        chunker = HierarchicalChunker()
        
        doc = parser.parse_file(test_file)
        chunks = chunker.chunk_document(doc)
        
        # Estadísticas
        macro = [c for c in chunks if c.level == ChunkLevel.MACRO]
        meso = [c for c in chunks if c.level == ChunkLevel.MESO]
        micro = [c for c in chunks if c.level == ChunkLevel.MICRO]
        
        print(f"\n📊 Estadísticas de Chunking:")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   - MACRO: {len(macro)}")
        print(f"   - MESO:  {len(meso)}")
        print(f"   - MICRO: {len(micro)}")
        
        print(f"\n📝 Primeros 5 chunks MICRO:")
        for chunk in micro[:5]:
            print(f"\n   [{chunk.chunk_id}]")
            print(f"   Header: {chunk.header_path}")
            print(f"   Tokens: {chunk.token_count}")
            print(f"   Preview: {chunk.content[:100]}...")
