"""
Chunking Semántico Adaptativo.

Este módulo extiende el chunker jerárquico con detección de límites
semánticos naturales:
- Definiciones
- Teoremas/Lemas/Corolarios
- Demostraciones
- Ejemplos
- Ejercicios
- Notas/Observaciones
- Ecuaciones destacadas

En lugar de cortar por tamaño fijo, detecta estas estructuras y las
preserva como unidades atómicas.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Pattern
from enum import Enum
import logging

from .chunker import Chunk, ChunkLevel, HierarchicalChunker
from .parser import ParsedSection, ParsedDocument

logger = logging.getLogger(__name__)


class SemanticBlockType(Enum):
    """Tipos de bloques semánticos detectables."""
    DEFINITION = "definition"           # Definición
    THEOREM = "theorem"                 # Teorema
    LEMMA = "lemma"                     # Lema
    COROLLARY = "corollary"             # Corolario
    PROPOSITION = "proposition"         # Proposición
    PROOF = "proof"                     # Demostración
    EXAMPLE = "example"                 # Ejemplo
    EXERCISE = "exercise"               # Ejercicio
    REMARK = "remark"                   # Observación/Nota
    ALGORITHM = "algorithm"             # Algoritmo
    EQUATION = "equation"               # Ecuación destacada
    PROTOCOL = "protocol"               # Protocolo (QKD, etc.)
    PROPERTY = "property"               # Propiedad
    AXIOM = "axiom"                     # Axioma/Postulado
    CODE_BLOCK = "code_block"           # Bloque de código
    REGULAR = "regular"                 # Texto regular


@dataclass
class SemanticBlock:
    """Representa un bloque semántico detectado."""
    block_type: SemanticBlockType
    content: str
    title: Optional[str] = None         # "Teorema 3.1", "Definición de qubit"
    label: Optional[str] = None         # Label para referencias
    start_pos: int = 0
    end_pos: int = 0
    is_atomic: bool = True              # No debe ser cortado
    token_count: int = 0


@dataclass
class SemanticChunkerConfig:
    """Configuración del chunker semántico."""
    # Tamaños máximos antes de forzar división
    max_definition_tokens: int = 300
    max_theorem_tokens: int = 500
    max_proof_tokens: int = 800
    max_example_tokens: int = 600
    max_regular_tokens: int = 200
    
    # Si un bloque excede el máximo, ¿dividirlo?
    split_large_blocks: bool = True
    
    # Incluir contexto de título en el chunk
    include_title_context: bool = True
    
    # Preservar bloques de código completos
    preserve_code_blocks: bool = True
    
    # Detectar ecuaciones LaTeX como bloques
    detect_latex_equations: bool = True


class SemanticChunker(HierarchicalChunker):
    """
    Chunker que detecta límites semánticos naturales.
    
    Extiende HierarchicalChunker con:
    1. Detección de bloques semánticos (definiciones, teoremas, etc.)
    2. Preservación de unidades atómicas
    3. División inteligente respetando estructura
    """
    
    # Patrones de detección de bloques semánticos
    # Soporta formatos comunes en español e inglés
    BLOCK_PATTERNS: Dict[SemanticBlockType, List[Pattern]] = {
        SemanticBlockType.DEFINITION: [
            re.compile(r'(?:^|\n)\*\*(?:Definición|Definition)\s*[\d\.]*[:\.]?\*\*[:\s]*(.*?)(?=\n\*\*(?:Definición|Teorema|Lema|Ejemplo|Definition|Theorem|Lemma|Example)|$)', re.IGNORECASE | re.DOTALL),
            re.compile(r'(?:^|\n)>\s*\*\*(?:Definición|Definition)\s*[\d\.]*[:\.]?\*\*[:\s]*(.*?)(?=\n[^>]|\n$|$)', re.IGNORECASE | re.DOTALL),
            re.compile(r'(?:^|\n)#{1,4}\s*(?:Definición|Definition)\s*[\d\.]*[:\.]?\s*(.*?)(?=\n#{1,4}\s|$)', re.IGNORECASE | re.DOTALL),
        ],
        SemanticBlockType.THEOREM: [
            re.compile(r'(?:^|\n)\*\*(?:Teorema|Theorem)\s*[\d\.]*[:\.]?\*\*[:\s]*(.*?)(?=\n\*\*(?:Demostración|Proof|Definición|Teorema|Lema|Definition|Theorem|Lemma)|$)', re.IGNORECASE | re.DOTALL),
            re.compile(r'(?:^|\n)>\s*\*\*(?:Teorema|Theorem)\s*[\d\.]*[:\.]?\*\*[:\s]*(.*?)(?=\n[^>]|\n$|$)', re.IGNORECASE | re.DOTALL),
        ],
        SemanticBlockType.LEMMA: [
            re.compile(r'(?:^|\n)\*\*(?:Lema|Lemma)\s*[\d\.]*[:\.]?\*\*[:\s]*(.*?)(?=\n\*\*(?:Demostración|Proof|Definición|Teorema|Definition|Theorem)|$)', re.IGNORECASE | re.DOTALL),
        ],
        SemanticBlockType.COROLLARY: [
            re.compile(r'(?:^|\n)\*\*(?:Corolario|Corollary)\s*[\d\.]*[:\.]?\*\*[:\s]*(.*?)(?=\n\*\*(?:Demostración|Proof|Definición|Teorema|Definition|Theorem)|$)', re.IGNORECASE | re.DOTALL),
        ],
        SemanticBlockType.PROPOSITION: [
            re.compile(r'(?:^|\n)\*\*(?:Proposición|Proposition)\s*[\d\.]*[:\.]?\*\*[:\s]*(.*?)(?=\n\*\*|$)', re.IGNORECASE | re.DOTALL),
        ],
        SemanticBlockType.PROOF: [
            re.compile(r'(?:^|\n)\*\*(?:Demostración|Proof)[:\.]?\*\*[:\s]*(.*?)(?=\n\*\*(?:Definición|Teorema|Lema|Ejemplo|Definition|Theorem|Lemma|Example)|(?:□|∎|QED|$))', re.IGNORECASE | re.DOTALL),
            re.compile(r'(?:^|\n)_(?:Demostración|Proof)[:\.]?_[:\s]*(.*?)(?=\n\*\*|□|∎|QED|$)', re.IGNORECASE | re.DOTALL),
        ],
        SemanticBlockType.EXAMPLE: [
            re.compile(r'(?:^|\n)\*\*(?:Ejemplo|Example)\s*[\d\.]*[:\.]?\*\*[:\s]*(.*?)(?=\n\*\*(?:Definición|Teorema|Lema|Ejemplo|Definition|Theorem|Lemma|Example)|$)', re.IGNORECASE | re.DOTALL),
            re.compile(r'(?:^|\n)#{1,4}\s*(?:Ejemplo|Example)\s*[\d\.]*[:\.]?\s*(.*?)(?=\n#{1,4}\s|$)', re.IGNORECASE | re.DOTALL),
        ],
        SemanticBlockType.EXERCISE: [
            re.compile(r'(?:^|\n)\*\*(?:Ejercicio|Exercise)\s*[\d\.]*[:\.]?\*\*[:\s]*(.*?)(?=\n\*\*|$)', re.IGNORECASE | re.DOTALL),
        ],
        SemanticBlockType.REMARK: [
            re.compile(r'(?:^|\n)\*\*(?:Nota|Observación|Note|Remark)\s*[\d\.]*[:\.]?\*\*[:\s]*(.*?)(?=\n\*\*|$)', re.IGNORECASE | re.DOTALL),
            re.compile(r'(?:^|\n)>\s*\*\*(?:Nota|Note)\*\*[:\s]*(.*?)(?=\n[^>]|$)', re.IGNORECASE | re.DOTALL),
        ],
        SemanticBlockType.ALGORITHM: [
            re.compile(r'(?:^|\n)\*\*(?:Algoritmo|Algorithm)\s*[\d\.]*[:\.]?\*\*[:\s]*(.*?)(?=\n\*\*|$)', re.IGNORECASE | re.DOTALL),
            re.compile(r'(?:^|\n)#{1,4}\s*(?:Algoritmo|Algorithm)\s*[\d\.]*[:\.]?\s*(.*?)(?=\n#{1,4}\s|$)', re.IGNORECASE | re.DOTALL),
        ],
        SemanticBlockType.PROTOCOL: [
            re.compile(r'(?:^|\n)\*\*(?:Protocolo|Protocol)\s*[\d\.]*[:\.]?\*\*[:\s]*(.*?)(?=\n\*\*|$)', re.IGNORECASE | re.DOTALL),
            re.compile(r'(?:^|\n)#{1,4}\s*(?:Protocolo|Protocol)\s*[\w\-]*[:\.]?\s*(.*?)(?=\n#{1,4}\s|$)', re.IGNORECASE | re.DOTALL),
        ],
        SemanticBlockType.PROPERTY: [
            re.compile(r'(?:^|\n)\*\*(?:Propiedad|Property)\s*[\d\.]*[:\.]?\*\*[:\s]*(.*?)(?=\n\*\*|$)', re.IGNORECASE | re.DOTALL),
        ],
        SemanticBlockType.AXIOM: [
            re.compile(r'(?:^|\n)\*\*(?:Axioma|Postulado|Axiom|Postulate)\s*[\d\.]*[:\.]?\*\*[:\s]*(.*?)(?=\n\*\*|$)', re.IGNORECASE | re.DOTALL),
        ],
        SemanticBlockType.CODE_BLOCK: [
            re.compile(r'```[\w]*\n(.*?)```', re.DOTALL),
        ],
        SemanticBlockType.EQUATION: [
            re.compile(r'\$\$(.*?)\$\$', re.DOTALL),
        ],
    }
    
    # Patrones para detectar títulos de bloques
    TITLE_PATTERN = re.compile(
        r'(?:Definición|Teorema|Lema|Corolario|Proposición|Ejemplo|'
        r'Ejercicio|Algoritmo|Protocolo|Propiedad|Axioma|'
        r'Definition|Theorem|Lemma|Corollary|Proposition|Example|'
        r'Exercise|Algorithm|Protocol|Property|Axiom)\s*[\d\.\-]*'
        r'(?:\s*\(([^)]+)\))?',  # Captura nombre entre paréntesis
        re.IGNORECASE
    )
    
    def __init__(
        self,
        config: Optional[SemanticChunkerConfig] = None,
        micro_size: int = 200,
        meso_size: int = 512,
        macro_size: int = 2048,
        overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        """
        Inicializa el chunker semántico.
        
        Args:
            config: Configuración específica del chunker semántico
            micro_size: Tamaño base para chunks micro
            meso_size: Tamaño base para chunks meso
            macro_size: Tamaño base para chunks macro
            overlap: Solapamiento entre chunks
            encoding_name: Encoding de tiktoken
        """
        super().__init__(
            micro_size=micro_size,
            meso_size=meso_size,
            macro_size=macro_size,
            overlap=overlap,
            encoding_name=encoding_name
        )
        
        self.config = config or SemanticChunkerConfig()
        
        # Estadísticas de detección
        self.stats: Dict[str, int] = {
            block_type.value: 0 for block_type in SemanticBlockType
        }
    
    def detect_semantic_blocks(self, text: str) -> List[SemanticBlock]:
        """
        Detecta bloques semánticos en el texto.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Lista de bloques semánticos ordenados por posición
        """
        blocks: List[SemanticBlock] = []
        covered_ranges: List[Tuple[int, int]] = []
        
        # Detectar cada tipo de bloque
        for block_type, patterns in self.BLOCK_PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    start = match.start()
                    end = match.end()
                    
                    # Evitar solapamientos
                    if any(
                        start < r[1] and end > r[0] 
                        for r in covered_ranges
                    ):
                        continue
                    
                    content = match.group(0).strip()
                    
                    # Extraer título si existe
                    title_match = self.TITLE_PATTERN.search(content[:100])
                    title = title_match.group(0) if title_match else None
                    
                    # Determinar si es atómico
                    is_atomic = block_type not in [
                        SemanticBlockType.REGULAR,
                        SemanticBlockType.PROOF  # Las demostraciones largas se pueden dividir
                    ]
                    
                    block = SemanticBlock(
                        block_type=block_type,
                        content=content,
                        title=title,
                        start_pos=start,
                        end_pos=end,
                        is_atomic=is_atomic,
                        token_count=self.count_tokens(content)
                    )
                    
                    blocks.append(block)
                    covered_ranges.append((start, end))
                    self.stats[block_type.value] += 1
        
        # Marcar texto no cubierto como REGULAR
        blocks.sort(key=lambda b: b.start_pos)
        regular_blocks = self._extract_regular_text(text, blocks)
        blocks.extend(regular_blocks)
        blocks.sort(key=lambda b: b.start_pos)
        
        return blocks
    
    def _extract_regular_text(
        self, 
        text: str, 
        blocks: List[SemanticBlock]
    ) -> List[SemanticBlock]:
        """Extrae texto regular no cubierto por bloques semánticos."""
        regular_blocks = []
        current_pos = 0
        
        for block in sorted(blocks, key=lambda b: b.start_pos):
            if block.start_pos > current_pos:
                regular_content = text[current_pos:block.start_pos].strip()
                if regular_content and len(regular_content) > 10:
                    regular_block = SemanticBlock(
                        block_type=SemanticBlockType.REGULAR,
                        content=regular_content,
                        start_pos=current_pos,
                        end_pos=block.start_pos,
                        is_atomic=False,
                        token_count=self.count_tokens(regular_content)
                    )
                    regular_blocks.append(regular_block)
                    self.stats["regular"] += 1
            
            current_pos = max(current_pos, block.end_pos)
        
        # Texto final
        if current_pos < len(text):
            final_content = text[current_pos:].strip()
            if final_content and len(final_content) > 10:
                regular_block = SemanticBlock(
                    block_type=SemanticBlockType.REGULAR,
                    content=final_content,
                    start_pos=current_pos,
                    end_pos=len(text),
                    is_atomic=False,
                    token_count=self.count_tokens(final_content)
                )
                regular_blocks.append(regular_block)
                self.stats["regular"] += 1
        
        return regular_blocks
    
    def _get_max_tokens_for_type(self, block_type: SemanticBlockType) -> int:
        """Obtiene el máximo de tokens permitido para un tipo de bloque."""
        type_limits = {
            SemanticBlockType.DEFINITION: self.config.max_definition_tokens,
            SemanticBlockType.THEOREM: self.config.max_theorem_tokens,
            SemanticBlockType.LEMMA: self.config.max_theorem_tokens,
            SemanticBlockType.COROLLARY: self.config.max_theorem_tokens,
            SemanticBlockType.PROPOSITION: self.config.max_theorem_tokens,
            SemanticBlockType.PROOF: self.config.max_proof_tokens,
            SemanticBlockType.EXAMPLE: self.config.max_example_tokens,
            SemanticBlockType.EXERCISE: self.config.max_example_tokens,
            SemanticBlockType.REMARK: self.config.max_definition_tokens,
            SemanticBlockType.ALGORITHM: self.config.max_theorem_tokens,
            SemanticBlockType.PROTOCOL: self.config.max_theorem_tokens,
            SemanticBlockType.PROPERTY: self.config.max_definition_tokens,
            SemanticBlockType.AXIOM: self.config.max_definition_tokens,
            SemanticBlockType.CODE_BLOCK: self.config.max_proof_tokens,
            SemanticBlockType.EQUATION: self.config.max_definition_tokens,
            SemanticBlockType.REGULAR: self.config.max_regular_tokens,
        }
        return type_limits.get(block_type, self.micro_size)
    
    def _chunk_section(self, section: ParsedSection) -> List[Chunk]:
        """
        Override: Divide una sección respetando límites semánticos.
        
        Args:
            section: Sección parseada
            
        Returns:
            Lista de chunks
        """
        chunks = []
        content = section.content
        
        # Detectar bloques semánticos
        semantic_blocks = self.detect_semantic_blocks(content)
        
        if not semantic_blocks:
            # Fallback al chunking por tamaño
            return super()._chunk_section(section)
        
        # Procesar cada bloque semántico
        for block in semantic_blocks:
            max_tokens = self._get_max_tokens_for_type(block.block_type)
            
            # Si el bloque es atómico y cabe en el límite, crear un chunk
            if block.is_atomic and block.token_count <= max_tokens:
                chunk = self._create_semantic_chunk(
                    block, section, ChunkLevel.MICRO
                )
                chunks.append(chunk)
            
            # Si el bloque es muy grande, dividirlo
            elif block.token_count > max_tokens and self.config.split_large_blocks:
                sub_chunks = self._split_large_block(block, section, max_tokens)
                chunks.extend(sub_chunks)
            
            # Bloque grande pero atómico: mantenerlo como MESO
            elif block.is_atomic:
                chunk = self._create_semantic_chunk(
                    block, section, ChunkLevel.MESO
                )
                chunks.append(chunk)
            
            # Bloque regular: dividir con método estándar
            else:
                sub_chunks = self._split_by_size(
                    block.content,
                    self.micro_size,
                    ChunkLevel.MICRO,
                    section
                )
                chunks.extend(sub_chunks)
        
        # Añadir referencias de jerarquía (MACRO → MESO → MICRO)
        chunks = self._add_hierarchy_refs(chunks, section)
        
        logger.debug(
            f"Sección '{section.header_path}': "
            f"{len(chunks)} chunks semánticos"
        )
        
        return chunks
    
    def _create_semantic_chunk(
        self,
        block: SemanticBlock,
        section: ParsedSection,
        level: ChunkLevel
    ) -> Chunk:
        """
        Crea un chunk a partir de un bloque semántico.
        
        Args:
            block: Bloque semántico
            section: Sección original
            level: Nivel del chunk
            
        Returns:
            Chunk creado
        """
        content = block.content
        
        # Añadir contexto de título si está configurado
        if (
            self.config.include_title_context 
            and block.title 
            and block.title not in content[:50]
        ):
            content = f"**{block.title}**: {content}"
        
        self._chunk_counter += 1
        chunk_id = (
            f"{section.doc_id}_{level.value}_"
            f"{block.block_type.value}_{self._chunk_counter:06d}"
        )
        
        return Chunk(
            chunk_id=chunk_id,
            content=content,
            level=level,
            doc_id=section.doc_id,
            doc_title=section.doc_title,
            header_path=section.header_path,
            start_char=section.start_char + block.start_pos,
            end_char=section.start_char + block.end_pos,
            token_count=block.token_count
        )
    
    def _split_large_block(
        self,
        block: SemanticBlock,
        section: ParsedSection,
        max_tokens: int
    ) -> List[Chunk]:
        """
        Divide un bloque grande intentando preservar estructura interna.
        
        Para demostraciones: divide por párrafos/pasos
        Para ejemplos: divide por sub-ejemplos
        Para código: divide por funciones/bloques
        
        Args:
            block: Bloque a dividir
            section: Sección original
            max_tokens: Máximo de tokens por sub-chunk
            
        Returns:
            Lista de chunks
        """
        chunks = []
        content = block.content
        
        # Separadores específicos por tipo
        if block.block_type == SemanticBlockType.PROOF:
            # Dividir demostraciones por pasos numerados o párrafos
            separators = [
                r'\n\d+\.\s',      # Pasos numerados
                r'\n\*\s',         # Bullets
                r'\n\n',           # Párrafos
            ]
        elif block.block_type == SemanticBlockType.CODE_BLOCK:
            # Dividir código por funciones/clases/bloques
            separators = [
                r'\ndef\s',        # Funciones Python
                r'\nclass\s',      # Clases
                r'\n\n',           # Bloques vacíos
            ]
        else:
            # División general por párrafos
            separators = [r'\n\n', r'\n']
        
        # Intentar dividir preservando el título
        title_prefix = ""
        if block.title:
            title_prefix = f"**{block.title}** (cont.): "
        
        parts = [content]
        for sep_pattern in separators:
            new_parts = []
            for part in parts:
                split_parts = re.split(sep_pattern, part)
                new_parts.extend(split_parts)
            parts = new_parts
            
            # Si ya tenemos partes suficientemente pequeñas, parar
            if all(self.count_tokens(p) <= max_tokens for p in parts):
                break
        
        # Crear chunks a partir de las partes
        current_content = ""
        part_num = 0
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            potential = current_content + "\n" + part if current_content else part
            
            if self.count_tokens(potential) <= max_tokens:
                current_content = potential
            else:
                if current_content:
                    part_num += 1
                    chunk_content = (
                        title_prefix + current_content 
                        if part_num > 1 else current_content
                    )
                    
                    self._chunk_counter += 1
                    chunk_id = (
                        f"{section.doc_id}_micro_"
                        f"{block.block_type.value}_{self._chunk_counter:06d}"
                    )
                    
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        content=chunk_content,
                        level=ChunkLevel.MICRO,
                        doc_id=section.doc_id,
                        doc_title=section.doc_title,
                        header_path=section.header_path,
                        token_count=self.count_tokens(chunk_content)
                    )
                    chunks.append(chunk)
                
                current_content = part
        
        # Último chunk
        if current_content:
            part_num += 1
            chunk_content = (
                title_prefix + current_content 
                if part_num > 1 else current_content
            )
            
            self._chunk_counter += 1
            chunk_id = (
                f"{section.doc_id}_micro_"
                f"{block.block_type.value}_{self._chunk_counter:06d}"
            )
            
            chunk = Chunk(
                chunk_id=chunk_id,
                content=chunk_content,
                level=ChunkLevel.MICRO,
                doc_id=section.doc_id,
                doc_title=section.doc_title,
                header_path=section.header_path,
                token_count=self.count_tokens(chunk_content)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _add_hierarchy_refs(
        self,
        chunks: List[Chunk],
        section: ParsedSection
    ) -> List[Chunk]:
        """
        Añade referencias de jerarquía padre-hijo a los chunks.
        
        Crea un chunk MACRO virtual que agrupa todos los MICRO de la sección.
        
        Args:
            chunks: Lista de chunks de la sección
            section: Sección original
            
        Returns:
            Lista de chunks con referencias actualizadas
        """
        if not chunks:
            return chunks
        
        # Crear chunk MACRO para la sección
        full_content = "\n\n".join(c.content for c in chunks)
        macro_tokens = self.count_tokens(full_content)
        
        self._chunk_counter += 1
        macro_id = f"{section.doc_id}_macro_section_{self._chunk_counter:06d}"
        
        macro_chunk = Chunk(
            chunk_id=macro_id,
            content=full_content[:self.macro_size * 4],  # Truncar si es muy largo
            level=ChunkLevel.MACRO,
            doc_id=section.doc_id,
            doc_title=section.doc_title,
            header_path=section.header_path,
            children_ids=[c.chunk_id for c in chunks],
            token_count=min(macro_tokens, self.macro_size)
        )
        
        # Actualizar parent_id de los chunks hijos
        for chunk in chunks:
            chunk.parent_id = macro_id
        
        # Añadir el macro chunk al inicio
        return [macro_chunk] + chunks
    
    def get_detection_stats(self) -> Dict[str, int]:
        """Devuelve estadísticas de detección de bloques."""
        return dict(self.stats)
    
    def reset_stats(self):
        """Resetea las estadísticas de detección."""
        self.stats = {
            block_type.value: 0 for block_type in SemanticBlockType
        }


def create_semantic_chunker(
    config: Optional[SemanticChunkerConfig] = None,
    **kwargs
) -> SemanticChunker:
    """
    Factory function para crear un SemanticChunker.
    
    Args:
        config: Configuración del chunker
        **kwargs: Argumentos adicionales para HierarchicalChunker
        
    Returns:
        Instancia de SemanticChunker
    """
    return SemanticChunker(config=config, **kwargs)


if __name__ == "__main__":
    # Test de detección semántica
    import sys
    
    logging.basicConfig(level=logging.DEBUG)
    
    test_text = r"""
## Espacios de Hilbert

**Definición 2.1:** Un *espacio de Hilbert* $\mathcal{H}$ es un espacio 
vectorial complejo con producto interno $\langle \cdot | \cdot \rangle$ 
que es completo respecto a la norma inducida.

**Teorema 2.2 (Riesz):** Todo funcional lineal continuo sobre un espacio 
de Hilbert puede representarse como un producto interno.

**Demostración:** Sea $f: \mathcal{H} \to \mathbb{C}$ un funcional lineal 
continuo. Consideremos el kernel $\ker(f) = \{x \in \mathcal{H} : f(x) = 0\}$.

1. Si $\ker(f) = \mathcal{H}$, entonces $f = 0$ y tomamos $y = 0$.
2. Si $\ker(f) \neq \mathcal{H}$, existe $z \perp \ker(f)$ con $f(z) \neq 0$.
3. Definimos $y = \overline{f(z)} z / \|z\|^2$.

Por tanto, $f(x) = \langle x | y \rangle$ para todo $x$. □

**Ejemplo 2.3:** El espacio $L^2([0,1])$ con el producto interno
$$\langle f | g \rangle = \int_0^1 f(x)\overline{g(x)} dx$$
es un espacio de Hilbert.

**Nota:** Este resultado es fundamental en mecánica cuántica.
"""
    
    chunker = SemanticChunker()
    blocks = chunker.detect_semantic_blocks(test_text)
    
    print("\n📊 Bloques semánticos detectados:")
    print("=" * 60)
    
    for block in blocks:
        print(f"\n[{block.block_type.value}]")
        if block.title:
            print(f"  Título: {block.title}")
        print(f"  Tokens: {block.token_count}")
        print(f"  Atómico: {block.is_atomic}")
        print(f"  Preview: {block.content[:80]}...")
    
    print("\n📈 Estadísticas:")
    for block_type, count in chunker.get_detection_stats().items():
        if count > 0:
            print(f"  {block_type}: {count}")
