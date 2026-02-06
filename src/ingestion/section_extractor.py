"""
Section Extractor - Extracción LLM de jerarquía de secciones.

Extrae metadata estructurada de cada chunk para identificación precisa:
- section_hierarchy: lista ordenada de secciones anidadas
- section_number: numeración normalizada (ej: "5.2.1")
- topic_summary: descripción breve del contenido
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import re

logger = logging.getLogger(__name__)


@dataclass
class SectionMetadata:
    """Metadata de sección extraída por LLM."""
    chunk_id: str
    section_hierarchy: List[str]  # ["Capítulo 5: Algoritmo de Shor", "5.2 Factorización"]
    section_number: str  # "5.2.1"
    topic_summary: str  # "Explicación del período cuántico en factorización"
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "section_hierarchy": self.section_hierarchy,
            "section_number": self.section_number,
            "topic_summary": self.topic_summary,
            "confidence": self.confidence,
        }

    def format_citation(self, doc_title: str) -> str:
        """Formatea para cita legible."""
        parts = [doc_title]
        if self.section_hierarchy:
            parts.append(", ".join(self.section_hierarchy[:2]))  # Max 2 niveles
        if self.section_number and self.section_number not in str(self.section_hierarchy):
            parts.append(f"§{self.section_number}")
        return " — ".join(parts)


class SectionExtractor:
    """
    Extrae jerarquía de secciones de chunks usando LLM.

    Procesa chunks en batches para eficiencia, extrayendo:
    - Jerarquía de secciones (capítulo > sección > subsección)
    - Numeración normalizada
    - Resumen temático breve
    """

    EXTRACTION_PROMPT = """Analiza este fragmento de texto académico y extrae su ubicación estructural.

TEXTO:
{content}

CONTEXTO DEL DOCUMENTO:
- Título: {doc_title}
- Header path detectado: {header_path}

Devuelve JSON con:
{{
  "section_hierarchy": ["Nivel 1: título", "Nivel 2: título", ...],
  "section_number": "X.Y.Z" o "" si no hay numeración clara,
  "topic_summary": "Frase breve (max 15 palabras) describiendo el tema específico"
}}

Reglas:
- section_hierarchy: lista ordenada del más general al más específico
- Si hay numeración explícita (1.2.3, Chapter 5, etc.), inclúyela en section_hierarchy
- section_number: solo la numeración limpia, sin texto
- topic_summary: qué concepto/tema específico trata este fragmento

Responde SOLO con el JSON, sin explicaciones."""

    def __init__(self, batch_size: int = 10):
        """
        Args:
            batch_size: chunks por llamada LLM
        """
        self.batch_size = batch_size

    def extract_single(
        self,
        chunk_id: str,
        content: str,
        doc_title: str,
        header_path: str
    ) -> SectionMetadata:
        """Extrae metadata de un solo chunk."""
        from src.llm_provider import complete as llm_complete

        prompt = self.EXTRACTION_PROMPT.format(
            content=content[:2000],  # Truncar si muy largo
            doc_title=doc_title,
            header_path=header_path or "No detectado"
        )

        try:
            response = llm_complete(
                prompt=prompt,
                system="Eres un asistente que extrae estructura de documentos académicos. Responde solo JSON válido.",
                temperature=0,
                max_tokens=300,
                json_mode=True
            )

            data = json.loads(response.content)
            return SectionMetadata(
                chunk_id=chunk_id,
                section_hierarchy=data.get("section_hierarchy", []),
                section_number=data.get("section_number", ""),
                topic_summary=data.get("topic_summary", ""),
                confidence=1.0
            )

        except Exception as e:
            logger.warning(f"Error extrayendo sección para {chunk_id}: {e}")
            return self._fallback_extraction(chunk_id, content, header_path)

    def extract_batch(
        self,
        chunks: List[Tuple[str, str, str, str]]  # [(chunk_id, content, doc_title, header_path), ...]
    ) -> List[SectionMetadata]:
        """
        Extrae metadata de múltiples chunks en una llamada LLM.

        Args:
            chunks: lista de tuplas (chunk_id, content, doc_title, header_path)

        Returns:
            Lista de SectionMetadata
        """
        if not chunks:
            return []

        from src.llm_provider import complete as llm_complete

        # Construir prompt para batch
        chunks_text = []
        for i, (chunk_id, content, doc_title, header_path) in enumerate(chunks):
            chunks_text.append(f"""
--- CHUNK {i} (id: {chunk_id}) ---
Documento: {doc_title}
Header detectado: {header_path or "ninguno"}
Contenido:
{content[:1500]}
""")

        batch_prompt = f"""Analiza estos {len(chunks)} fragmentos de textos académicos y extrae la ubicación estructural de cada uno.

{chr(10).join(chunks_text)}

Devuelve un JSON array con un objeto por cada chunk, en el mismo orden:
[
  {{
    "chunk_index": 0,
    "section_hierarchy": ["Nivel 1", "Nivel 2", ...],
    "section_number": "X.Y.Z",
    "topic_summary": "descripción breve"
  }},
  ...
]

Reglas:
- section_hierarchy: lista ordenada del más general al más específico
- section_number: solo numeración limpia (ej: "5.2.1"), "" si no hay
- topic_summary: max 15 palabras describiendo el tema específico del fragmento

Responde SOLO con el JSON array."""

        try:
            response = llm_complete(
                prompt=batch_prompt,
                system="Eres un asistente que extrae estructura de documentos académicos. Responde solo JSON válido.",
                temperature=0,
                max_tokens=200 * len(chunks),
                json_mode=True
            )

            results_data = json.loads(response.content)

            # Asegurar que es una lista
            if isinstance(results_data, dict) and "results" in results_data:
                results_data = results_data["results"]

            results = []
            for i, (chunk_id, content, doc_title, header_path) in enumerate(chunks):
                # Buscar resultado correspondiente
                chunk_data = None
                for r in results_data:
                    if r.get("chunk_index") == i:
                        chunk_data = r
                        break

                if chunk_data:
                    results.append(SectionMetadata(
                        chunk_id=chunk_id,
                        section_hierarchy=chunk_data.get("section_hierarchy", []),
                        section_number=chunk_data.get("section_number", ""),
                        topic_summary=chunk_data.get("topic_summary", ""),
                        confidence=1.0
                    ))
                else:
                    # Fallback si no encontramos el resultado
                    results.append(self._fallback_extraction(chunk_id, content, header_path))

            return results

        except Exception as e:
            logger.warning(f"Error en batch extraction: {e}. Usando fallback individual.")
            return [
                self._fallback_extraction(cid, content, hp)
                for cid, content, _, hp in chunks
            ]

    def extract_for_document(
        self,
        chunks: List[Tuple[str, str, str]],  # [(chunk_id, content, header_path), ...]
        doc_title: str
    ) -> List[SectionMetadata]:
        """
        Extrae metadata para todos los chunks de un documento.
        Procesa en batches.

        Args:
            chunks: lista de (chunk_id, content, header_path)
            doc_title: título del documento

        Returns:
            Lista de SectionMetadata
        """
        results = []

        # Preparar chunks con doc_title
        full_chunks = [(cid, content, doc_title, hp) for cid, content, hp in chunks]

        # Procesar en batches
        for i in range(0, len(full_chunks), self.batch_size):
            batch = full_chunks[i:i + self.batch_size]
            batch_results = self.extract_batch(batch)
            results.extend(batch_results)

            if i + self.batch_size < len(full_chunks):
                logger.info(f"Secciones extraídas: {len(results)}/{len(full_chunks)}")

        return results

    def _fallback_extraction(
        self,
        chunk_id: str,
        content: str,
        header_path: str
    ) -> SectionMetadata:
        """Extracción heurística sin LLM como fallback."""
        # Intentar extraer numeración del header_path o contenido
        section_number = ""
        section_hierarchy = []

        # Parsear header_path si existe
        if header_path:
            parts = [p.strip() for p in header_path.split(">")]
            section_hierarchy = parts

            # Buscar numeración en los headers
            for part in parts:
                match = re.search(r'^(\d+(?:\.\d+)*)', part)
                if match:
                    section_number = match.group(1)
                    break

        # Buscar numeración en el contenido si no la encontramos
        if not section_number:
            match = re.search(r'(?:Chapter|Capítulo|Section|Sección)\s*(\d+(?:\.\d+)*)', content[:500], re.I)
            if match:
                section_number = match.group(1)

        # Generar topic_summary básico
        # Tomar primera oración no trivial
        sentences = re.split(r'[.!?]\s+', content[:500])
        topic_summary = ""
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20 and not sent.startswith(('#', '-', '*')):
                topic_summary = sent[:100] + ("..." if len(sent) > 100 else "")
                break

        return SectionMetadata(
            chunk_id=chunk_id,
            section_hierarchy=section_hierarchy,
            section_number=section_number,
            topic_summary=topic_summary,
            confidence=0.5  # Menor confianza para fallback
        )
