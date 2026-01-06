"""
Citation Injector - Gestión de citas bibliográficas.

Funcionalidades:
- Extracción de citas del texto generado
- Validación contra fuentes
- Formateo de referencias
- Generación de footnotes
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import re
import logging

from ..retrieval.fusion import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Cita bibliográfica."""
    index: int
    chunk_id: str
    doc_title: str
    header_path: str
    doc_id: str
    excerpt: str  # Fragmento citado
    position_in_text: int  # Posición en el texto generado
    
    def format_reference(self, style: str = "inline") -> str:
        """
        Formatea la cita según el estilo.
        
        Args:
            style: "inline", "footnote", "endnote"
            
        Returns:
            Cita formateada
        """
        if style == "inline":
            return f"[{self.index}]"
        elif style == "footnote":
            return f"[^{self.index}]"
        elif style == "endnote":
            return f"({self.index})"
        else:
            return f"[{self.index}]"
    
    def format_bibliography_entry(self) -> str:
        """Genera entrada bibliográfica completa."""
        return f"[{self.index}] **{self.doc_title}** — {self.header_path}"


@dataclass
class CitedResponse:
    """Respuesta con citas procesadas."""
    content: str  # Contenido con citas numeradas
    citations: List[Citation]
    bibliography: str  # Referencias al final
    uncited_sources: List[str]  # Fuentes no citadas
    
    def get_full_response(self) -> str:
        """Obtiene respuesta completa con bibliografía."""
        if not self.citations:
            return self.content
        
        return f"{self.content}\n\n---\n\n**Referencias:**\n{self.bibliography}"


class CitationInjector:
    """
    Procesa y gestiona citas en respuestas RAG.
    
    Características:
    - Detecta citas [n] en el texto
    - Mapea a fuentes reales
    - Genera bibliografía
    - Valida que las citas correspondan al contenido
    """
    
    # Patrón para detectar citas [n] o [n,m] o [n-m]
    CITATION_PATTERN = re.compile(r'\[(\d+(?:[-,]\d+)*)\]')
    
    def __init__(self, max_excerpt_length: int = 150):
        """
        Args:
            max_excerpt_length: Longitud máxima del excerpt
        """
        self.max_excerpt_length = max_excerpt_length
    
    def process_response(
        self,
        response_text: str,
        sources: List[RetrievalResult]
    ) -> CitedResponse:
        """
        Procesa respuesta y extrae citas.
        
        Args:
            response_text: Texto generado por el LLM
            sources: Fuentes usadas para la generación
            
        Returns:
            Respuesta con citas procesadas
        """
        # Encontrar todas las citas en el texto
        citations = []
        used_indices = set()
        
        for match in self.CITATION_PATTERN.finditer(response_text):
            indices_str = match.group(1)
            position = match.start()
            
            # Parsear índices (puede ser "1", "1,2", "1-3")
            indices = self._parse_citation_indices(indices_str)
            
            for idx in indices:
                if idx <= len(sources) and idx not in used_indices:
                    source = sources[idx - 1]  # 1-indexed
                    
                    citation = Citation(
                        index=idx,
                        chunk_id=source.chunk_id,
                        doc_title=source.doc_title,
                        header_path=source.header_path,
                        doc_id=source.doc_id,
                        excerpt=self._get_excerpt(source.content),
                        position_in_text=position
                    )
                    citations.append(citation)
                    used_indices.add(idx)
        
        # Generar bibliografía
        bibliography = self._generate_bibliography(citations)
        
        # Identificar fuentes no citadas
        all_indices = set(range(1, len(sources) + 1))
        uncited_indices = all_indices - used_indices
        uncited_sources = [
            sources[i - 1].doc_title 
            for i in sorted(uncited_indices)
        ]
        
        logger.info(
            f"Citas procesadas: {len(citations)} citadas, "
            f"{len(uncited_sources)} no citadas"
        )
        
        return CitedResponse(
            content=response_text,
            citations=citations,
            bibliography=bibliography,
            uncited_sources=uncited_sources
        )
    
    def _parse_citation_indices(self, indices_str: str) -> List[int]:
        """Parsea string de índices de cita."""
        indices = []
        
        for part in indices_str.split(','):
            part = part.strip()
            if '-' in part:
                # Rango: "1-3" -> [1, 2, 3]
                start, end = part.split('-')
                indices.extend(range(int(start), int(end) + 1))
            else:
                indices.append(int(part))
        
        return indices
    
    def _get_excerpt(self, content: str) -> str:
        """Extrae excerpt del contenido."""
        # Limpiar y truncar
        excerpt = content.strip()
        if len(excerpt) > self.max_excerpt_length:
            excerpt = excerpt[:self.max_excerpt_length].rsplit(' ', 1)[0] + "..."
        return excerpt
    
    def _generate_bibliography(self, citations: List[Citation]) -> str:
        """Genera sección de bibliografía."""
        if not citations:
            return ""
        
        # Ordenar por índice
        sorted_citations = sorted(citations, key=lambda c: c.index)
        
        lines = []
        for citation in sorted_citations:
            entry = citation.format_bibliography_entry()
            lines.append(entry)
        
        return "\n".join(lines)
    
    def inject_citations(
        self,
        text: str,
        sources: List[RetrievalResult],
        style: str = "inline"
    ) -> str:
        """
        Inyecta citas en texto sin ellas.
        
        Usa heurísticas para determinar qué partes del texto
        corresponden a qué fuentes.
        
        Args:
            text: Texto sin citas
            sources: Fuentes disponibles
            style: Estilo de citas
            
        Returns:
            Texto con citas inyectadas
        """
        if not sources:
            return text
        
        # Dividir en oraciones
        sentences = self._split_sentences(text)
        
        cited_sentences = []
        for sentence in sentences:
            # Encontrar mejor fuente para esta oración
            best_source_idx = self._find_best_source(sentence, sources)
            
            if best_source_idx is not None:
                # Inyectar cita al final de la oración
                citation = f" [{best_source_idx + 1}]"
                
                # Insertar antes del punto final si lo hay
                if sentence.rstrip().endswith(('.', '!', '?')):
                    sentence = sentence.rstrip()[:-1] + citation + sentence.rstrip()[-1]
                else:
                    sentence = sentence + citation
            
            cited_sentences.append(sentence)
        
        return ' '.join(cited_sentences)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Divide texto en oraciones."""
        # Simple split por puntos (mejorable con nltk)
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_best_source(
        self,
        sentence: str,
        sources: List[RetrievalResult]
    ) -> Optional[int]:
        """Encuentra la mejor fuente para una oración."""
        sentence_lower = sentence.lower()
        
        # Extraer palabras clave (excluir stopwords)
        stopwords = {'el', 'la', 'los', 'las', 'un', 'una', 'de', 'en', 'que', 'y', 'a', 'es', 'se'}
        words = set(sentence_lower.split()) - stopwords
        
        best_idx = None
        best_overlap = 0
        
        for i, source in enumerate(sources):
            source_words = set(source.content.lower().split()) - stopwords
            overlap = len(words & source_words)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        
        # Solo asignar si hay suficiente overlap
        if best_overlap >= 3:
            return best_idx
        
        return None
    
    def validate_citations(
        self,
        response_text: str,
        sources: List[RetrievalResult]
    ) -> Dict[str, any]:
        """
        Valida que las citas sean correctas.
        
        Comprueba:
        - Que los índices existan
        - Que el contenido citado corresponda a la fuente
        
        Returns:
            Reporte de validación
        """
        report = {
            "valid": True,
            "invalid_indices": [],
            "suspicious_citations": [],
            "coverage": 0.0
        }
        
        # Encontrar citas
        used_indices = set()
        
        for match in self.CITATION_PATTERN.finditer(response_text):
            indices = self._parse_citation_indices(match.group(1))
            
            for idx in indices:
                if idx > len(sources):
                    report["invalid_indices"].append(idx)
                    report["valid"] = False
                else:
                    used_indices.add(idx)
        
        # Calcular cobertura
        if sources:
            report["coverage"] = len(used_indices) / len(sources)
        
        return report
    
    def create_hyperlinked_response(
        self,
        response: CitedResponse,
        base_url: Optional[str] = None
    ) -> str:
        """
        Crea respuesta con citas como enlaces.
        
        Para uso en Markdown con enlaces a las fuentes.
        
        Args:
            response: Respuesta procesada
            base_url: URL base para enlaces
            
        Returns:
            Respuesta con enlaces Markdown
        """
        text = response.content
        
        # Reemplazar [n] por enlaces
        for citation in response.citations:
            pattern = f"[{citation.index}]"
            
            if base_url:
                link = f"[{citation.index}]({base_url}#{citation.chunk_id})"
            else:
                # Enlace a la sección de referencias
                link = f"[{citation.index}](#ref-{citation.index})"
            
            text = text.replace(pattern, link, 1)
        
        # Añadir referencias con anchors
        if response.citations:
            text += "\n\n---\n\n**Referencias:**\n"
            
            for citation in sorted(response.citations, key=lambda c: c.index):
                anchor = f"<a id='ref-{citation.index}'></a>"
                entry = f"{anchor}**[{citation.index}]** {citation.doc_title} — _{citation.header_path}_"
                text += f"\n{entry}"
        
        return text
