"""
Parser de Markdown con extracción de Header Path.

Este módulo se encarga de:
1. Leer archivos Markdown de la biblioteca
2. Extraer la jerarquía de encabezados
3. Generar metadatos de ubicación (header_path) para cada sección
4. Preservar bloques de código y fórmulas LaTeX como unidades atómicas
"""

import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParsedSection:
    """Representa una sección parseada del documento."""
    content: str
    header_path: str
    level: int
    doc_id: str
    doc_title: str
    start_char: int
    end_char: int
    hash: str = field(default="")
    
    def __post_init__(self):
        if not self.hash:
            self.hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class ParsedDocument:
    """Representa un documento Markdown parseado completo."""
    doc_id: str
    doc_title: str
    file_path: str
    sections: List[ParsedSection]
    full_hash: str
    total_chars: int
    
    @property
    def total_sections(self) -> int:
        return len(self.sections)


class MarkdownParser:
    """
    Parser de Markdown que extrae estructura jerárquica y metadatos.

    Características:
    - Extrae jerarquía de encabezados (#, ##, ###, etc.)
    - Genera header_path para cada sección
    - Preserva bloques de código y fórmulas como unidades atómicas
    - Calcula hashes para detección de cambios
    - (Opcional) Describe imágenes referenciadas via vision LLM
    """

    # Patrones regex
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```', re.MULTILINE)
    LATEX_BLOCK_PATTERN = re.compile(r'\$\$[\s\S]*?\$\$', re.MULTILINE)
    LATEX_INLINE_PATTERN = re.compile(r'\$[^\$\n]+\$')

    def __init__(self, min_section_length: int = 50):
        """
        Args:
            min_section_length: Longitud mínima de una sección para ser incluida
        """
        self.min_section_length = min_section_length
        self._image_describer = None

    def set_image_describer(self, cache_dir: Path):
        """Habilita descripción de imágenes con vision LLM."""
        from .image_describer import ImageDescriber
        self._image_describer = ImageDescriber(cache_dir)
    
    def parse_file(self, file_path: Path) -> ParsedDocument:
        """
        Parsea un archivo Markdown y extrae su estructura.
        
        Args:
            file_path: Ruta al archivo .md
            
        Returns:
            ParsedDocument con todas las secciones y metadatos
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        if not file_path.suffix.lower() == '.md':
            raise ValueError(f"El archivo debe ser .md: {file_path}")
        
        # Leer contenido
        content = file_path.read_text(encoding='utf-8')

        # Describir imágenes (si habilitado)
        if self._image_describer:
            content = self._image_describer.process_document_images(content, file_path)

        # Generar IDs
        doc_id = self._generate_doc_id(file_path)
        doc_title = self._extract_title(content, file_path)
        full_hash = hashlib.sha256(content.encode()).hexdigest()

        # Parsear secciones
        sections = self._parse_sections(content, doc_id, doc_title)
        
        logger.info(f"Parseado: {doc_title} ({len(sections)} secciones)")
        
        return ParsedDocument(
            doc_id=doc_id,
            doc_title=doc_title,
            file_path=str(file_path),
            sections=sections,
            full_hash=full_hash,
            total_chars=len(content)
        )
    
    def _generate_doc_id(self, file_path: Path) -> str:
        """Genera un ID único para el documento basado en su ruta."""
        # Usar el nombre del archivo sin extensión, normalizado
        name = file_path.stem
        # Limpiar caracteres especiales
        name = re.sub(r'[^\w\s-]', '', name)
        name = re.sub(r'[-\s]+', '_', name)
        return name.lower()[:50]
    
    def _extract_title(self, content: str, file_path: Path) -> str:
        """Extrae el título del documento (primer H1 o nombre de archivo)."""
        # Buscar primer encabezado H1
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        
        # Fallback: usar nombre del archivo
        return file_path.stem
    
    def _parse_sections(
        self, 
        content: str, 
        doc_id: str, 
        doc_title: str
    ) -> List[ParsedSection]:
        """
        Parsea el contenido y divide en secciones con header_path.
        
        Args:
            content: Contenido completo del Markdown
            doc_id: ID del documento
            doc_title: Título del documento
            
        Returns:
            Lista de ParsedSection
        """
        sections = []
        
        # Encontrar todos los encabezados con sus posiciones
        headers = list(self.HEADER_PATTERN.finditer(content))
        
        if not headers:
            # Documento sin encabezados: tratar como una sola sección
            sections.append(ParsedSection(
                content=content,
                header_path=doc_title,
                level=0,
                doc_id=doc_id,
                doc_title=doc_title,
                start_char=0,
                end_char=len(content)
            ))
            return sections
        
        # Mantener stack de encabezados para construir header_path
        header_stack: List[Tuple[int, str]] = []
        
        for i, match in enumerate(headers):
            level = len(match.group(1))  # Número de #
            title = match.group(2).strip()
            start_pos = match.start()
            
            # Determinar fin de esta sección
            if i + 1 < len(headers):
                end_pos = headers[i + 1].start()
            else:
                end_pos = len(content)
            
            # Actualizar stack de encabezados
            # Eliminar encabezados de nivel igual o mayor
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            
            header_stack.append((level, title))
            
            # Construir header_path
            header_path = " > ".join([h[1] for h in header_stack])
            
            # Extraer contenido de la sección (sin el encabezado)
            section_content = content[match.end():end_pos].strip()
            
            # Filtrar secciones muy cortas
            if len(section_content) >= self.min_section_length:
                sections.append(ParsedSection(
                    content=section_content,
                    header_path=header_path,
                    level=level,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    start_char=match.end(),
                    end_char=end_pos
                ))
        
        # Si hay contenido antes del primer encabezado
        if headers and headers[0].start() > 0:
            pre_content = content[:headers[0].start()].strip()
            if len(pre_content) >= self.min_section_length:
                sections.insert(0, ParsedSection(
                    content=pre_content,
                    header_path=f"{doc_title} > Introducción",
                    level=0,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    start_char=0,
                    end_char=headers[0].start()
                ))
        
        return sections
    
    def find_atomic_blocks(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Encuentra bloques que no deben dividirse (código, fórmulas).
        
        Args:
            text: Texto a analizar
            
        Returns:
            Lista de (start, end, tipo) para cada bloque atómico
        """
        blocks = []
        
        # Bloques de código
        for match in self.CODE_BLOCK_PATTERN.finditer(text):
            blocks.append((match.start(), match.end(), 'code'))
        
        # Bloques LaTeX ($$...$$)
        for match in self.LATEX_BLOCK_PATTERN.finditer(text):
            blocks.append((match.start(), match.end(), 'latex_block'))
        
        # Ordenar por posición de inicio
        blocks.sort(key=lambda x: x[0])
        
        return blocks
    
    def is_position_in_atomic_block(
        self, 
        pos: int, 
        atomic_blocks: List[Tuple[int, int, str]]
    ) -> bool:
        """Verifica si una posición está dentro de un bloque atómico."""
        for start, end, _ in atomic_blocks:
            if start <= pos < end:
                return True
        return False


def parse_library(
    markdown_dir: Path,
    recursive: bool = True
) -> List[ParsedDocument]:
    """
    Parsea todos los archivos Markdown de un directorio.
    
    Args:
        markdown_dir: Directorio raíz de la biblioteca
        recursive: Si True, busca recursivamente en subdirectorios
        
    Returns:
        Lista de ParsedDocument
    """
    parser = MarkdownParser()
    documents = []
    
    markdown_dir = Path(markdown_dir)
    
    if recursive:
        md_files = list(markdown_dir.rglob("*.md"))
    else:
        md_files = list(markdown_dir.glob("*.md"))
    
    logger.info(f"Encontrados {len(md_files)} archivos Markdown")
    
    for file_path in md_files:
        try:
            doc = parser.parse_file(file_path)
            documents.append(doc)
        except Exception as e:
            logger.error(f"Error parseando {file_path}: {e}")
    
    return documents


if __name__ == "__main__":
    # Test básico
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
        parser = MarkdownParser()
        doc = parser.parse_file(test_file)
        
        print(f"\n📄 Documento: {doc.doc_title}")
        print(f"   ID: {doc.doc_id}")
        print(f"   Hash: {doc.full_hash[:16]}...")
        print(f"   Secciones: {doc.total_sections}")
        print("\n📑 Secciones:")
        for i, section in enumerate(doc.sections[:10]):
            print(f"   [{i+1}] {section.header_path}")
            print(f"       Nivel: {section.level}, Chars: {len(section.content)}")
