"""
Citation Grounding - Sistema de citas inline verificables.

Implementa:
1. Generación con citas obligatorias [n] para cada afirmación
2. Verificación de que cada cita está fundamentada en el contexto
3. Extracción del pasaje exacto que soporta cada cita
4. Puntuación de grounding para la respuesta completa

Basado en: "ALCE: Automatic Citation Localization and Extraction"
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import re
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class CitationEvidence:
    """Evidencia para una cita específica."""
    citation_id: int              # [1], [2], etc.
    claim: str                    # La afirmación que se cita
    source_chunk_id: str          # Chunk citado
    source_passage: str           # Pasaje exacto que soporta
    support_score: float          # 0-1, qué tan bien soporta
    is_verified: bool             # Si se verificó exitosamente
    verification_notes: str = ""  # Notas de verificación

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "claim": self.claim,
            "source_chunk_id": self.source_chunk_id,
            "source_passage": self.source_passage[:500],
            "support_score": self.support_score,
            "is_verified": self.is_verified,
            "verification_notes": self.verification_notes
        }


@dataclass
class GroundingResult:
    """Resultado del análisis de grounding."""
    total_citations: int
    verified_citations: int
    unverified_citations: List[int]
    grounding_score: float        # Proporción verificada
    evidences: List[CitationEvidence]
    claims_without_citations: List[str]
    overall_assessment: str

    def is_acceptable(self, min_grounding: float = 0.8) -> bool:
        """Determina si el grounding es aceptable."""
        return self.grounding_score >= min_grounding

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_citations": self.total_citations,
            "verified_citations": self.verified_citations,
            "unverified": self.unverified_citations,
            "grounding_score": self.grounding_score,
            "evidences": [e.to_dict() for e in self.evidences],
            "claims_without_citations": self.claims_without_citations,
            "assessment": self.overall_assessment
        }


class CitationGroundingSystem:
    """
    Sistema de citas inline con verificación.

    Características:
    - Genera respuestas con citas obligatorias [n]
    - Verifica cada cita contra el contexto original
    - Extrae el pasaje exacto que soporta la cita
    - Calcula score de grounding

    Uso:
        system = CitationGroundingSystem()
        response = system.generate_with_citations(query, contexts)
        grounding = system.verify_citations(response, contexts)
    """

    CITATION_GENERATION_PROMPT = """Eres un asistente académico que responde preguntas basándose ÚNICAMENTE en las fuentes proporcionadas.

REGLAS ESTRICTAS:
1. CADA afirmación factual DEBE tener una cita [n] donde n es el número de la fuente
2. NO hagas afirmaciones sin cita a menos que sean conectores lógicos
3. Si no puedes citar una afirmación, NO la incluyas
4. Usa el formato: "afirmación [n]" o "afirmación [n, m]" para múltiples fuentes
5. Las citas van DESPUÉS de la afirmación, no antes

FUENTES DISPONIBLES:
{sources}

PREGUNTA: {query}

Responde con citas inline obligatorias. Cada hecho debe tener su [n]."""

    VERIFICATION_PROMPT = """Verifica si esta afirmación está soportada por el pasaje fuente.

AFIRMACIÓN: {claim}

PASAJE FUENTE:
{source}

Evalúa:
1. ¿El pasaje contiene información que soporte la afirmación?
2. ¿La afirmación es fiel al contenido del pasaje (no distorsiona)?
3. ¿Hay suficiente evidencia o es una inferencia débil?

Responde JSON:
{{
  "is_supported": true/false,
  "support_score": 0.0-1.0,
  "supporting_passage": "Extracto exacto que soporta (o vacío si no hay)",
  "notes": "Explicación breve"
}}"""

    CLAIM_EXTRACTION_PROMPT = """Extrae las afirmaciones factuales de este texto, identificando qué cita [n] corresponde a cada una.

TEXTO:
{text}

Extrae cada afirmación factual con su cita asociada.
Ignora conectores, introducciones y opiniones.

Responde JSON:
{{
  "claims": [
    {{"claim": "afirmación 1", "citation_ids": [1]}},
    {{"claim": "afirmación 2", "citation_ids": [2, 3]}},
    ...
  ],
  "uncited_claims": ["afirmación sin cita 1", ...]
}}"""

    def __init__(
        self,
        min_grounding_score: float = 0.8,
        require_all_cited: bool = True,
        use_llm_verification: bool = True
    ):
        """
        Args:
            min_grounding_score: Score mínimo de grounding aceptable
            require_all_cited: Si exigir cita para toda afirmación
            use_llm_verification: Si usar LLM para verificar (False = heurístico)
        """
        self.min_grounding_score = min_grounding_score
        self.require_all_cited = require_all_cited
        self.use_llm_verification = use_llm_verification

    def generate_with_citations(
        self,
        query: str,
        contexts: List[Any],
        synthesizer=None
    ) -> str:
        """
        Genera respuesta con citas inline obligatorias.

        Args:
            query: Pregunta del usuario
            contexts: Lista de RetrievalResult como contexto
            synthesizer: Sintetizador opcional (usa default si None)

        Returns:
            Respuesta con citas [n] inline
        """
        from src.llm_provider import complete as llm_complete

        # Formatear fuentes con números
        sources_text = "\n\n".join([
            f"[{i+1}] {c.doc_title} - {c.header_path}\n{c.content}"
            for i, c in enumerate(contexts)
        ])

        prompt = self.CITATION_GENERATION_PROMPT.format(
            sources=sources_text,
            query=query
        )

        response = llm_complete(
            prompt=prompt,
            system="Eres un asistente académico riguroso. Toda afirmación debe tener cita.",
            temperature=0.3,
            max_tokens=2000
        )

        return response.content

    def verify_citations(
        self,
        response: str,
        contexts: List[Any]
    ) -> GroundingResult:
        """
        Verifica que las citas en la respuesta estén fundamentadas.

        Args:
            response: Respuesta con citas [n]
            contexts: Contextos originales

        Returns:
            GroundingResult con análisis completo
        """
        # 1. Extraer claims y citas
        claims_data = self._extract_claims(response)

        # 2. Crear mapa de contextos
        context_map = {i+1: c for i, c in enumerate(contexts)}

        # 3. Verificar cada cita
        evidences = []
        verified_count = 0
        unverified = []

        for claim_info in claims_data.get("claims", []):
            claim = claim_info.get("claim", "")
            citation_ids = claim_info.get("citation_ids", [])

            for cid in citation_ids:
                if cid not in context_map:
                    unverified.append(cid)
                    evidences.append(CitationEvidence(
                        citation_id=cid,
                        claim=claim,
                        source_chunk_id="",
                        source_passage="",
                        support_score=0.0,
                        is_verified=False,
                        verification_notes=f"Cita [{cid}] no corresponde a ninguna fuente"
                    ))
                    continue

                context = context_map[cid]

                # Verificar
                if self.use_llm_verification:
                    verification = self._verify_with_llm(claim, context.content)
                else:
                    verification = self._verify_heuristic(claim, context.content)

                evidence = CitationEvidence(
                    citation_id=cid,
                    claim=claim,
                    source_chunk_id=context.chunk_id,
                    source_passage=verification.get("supporting_passage", ""),
                    support_score=verification.get("support_score", 0.0),
                    is_verified=verification.get("is_supported", False),
                    verification_notes=verification.get("notes", "")
                )
                evidences.append(evidence)

                if evidence.is_verified:
                    verified_count += 1
                else:
                    unverified.append(cid)

        # 4. Calcular score
        total = len(evidences) if evidences else 1
        grounding_score = verified_count / total if total > 0 else 0.0

        # 5. Evaluar claims sin cita
        uncited = claims_data.get("uncited_claims", [])

        # Assessment
        if grounding_score >= 0.9 and not uncited:
            assessment = "Excelente: Todas las afirmaciones están bien fundamentadas"
        elif grounding_score >= 0.7:
            assessment = f"Aceptable: {verified_count}/{total} citas verificadas"
        elif grounding_score >= 0.5:
            assessment = f"Mejorable: Solo {verified_count}/{total} citas verificadas"
        else:
            assessment = f"Insuficiente: Muchas afirmaciones sin soporte adecuado"

        if uncited:
            assessment += f". {len(uncited)} afirmaciones sin cita."

        return GroundingResult(
            total_citations=total,
            verified_citations=verified_count,
            unverified_citations=unverified,
            grounding_score=grounding_score,
            evidences=evidences,
            claims_without_citations=uncited,
            overall_assessment=assessment
        )

    def _extract_claims(self, text: str) -> Dict[str, Any]:
        """Extrae claims y sus citas del texto."""
        from src.llm_provider import complete as llm_complete

        try:
            response = llm_complete(
                prompt=self.CLAIM_EXTRACTION_PROMPT.format(text=text),
                system="Extrae afirmaciones factuales con sus citas. Responde JSON.",
                temperature=0,
                max_tokens=1000,
                json_mode=True
            )

            return json.loads(response.content)

        except Exception as e:
            logger.warning(f"Error extrayendo claims: {e}")
            # Fallback: regex simple
            return self._extract_claims_regex(text)

    def _extract_claims_regex(self, text: str) -> Dict[str, Any]:
        """Extrae claims con regex (fallback)."""
        # Buscar patrones como "afirmación [n]" o "afirmación [n, m]"
        pattern = r'([^.!?\[\]]+)\s*\[(\d+(?:\s*,\s*\d+)*)\]'
        matches = re.findall(pattern, text)

        claims = []
        for claim_text, citations_str in matches:
            citation_ids = [int(c.strip()) for c in citations_str.split(',')]
            claims.append({
                "claim": claim_text.strip(),
                "citation_ids": citation_ids
            })

        # Buscar oraciones sin citas (posibles problemas)
        sentences = re.split(r'[.!?]', text)
        uncited = []
        for s in sentences:
            s = s.strip()
            if s and not re.search(r'\[\d+\]', s) and len(s) > 30:
                # Ignorar conectores comunes
                if not any(s.lower().startswith(c) for c in
                          ['en resumen', 'por lo tanto', 'así', 'es decir', 'además']):
                    uncited.append(s[:100])

        return {"claims": claims, "uncited_claims": uncited[:5]}

    def _verify_with_llm(self, claim: str, source: str) -> Dict[str, Any]:
        """Verifica claim contra source usando LLM."""
        from src.llm_provider import complete as llm_complete

        try:
            response = llm_complete(
                prompt=self.VERIFICATION_PROMPT.format(
                    claim=claim,
                    source=source[:1500]
                ),
                system="Eres un verificador de afirmaciones. Sé estricto. Responde JSON.",
                temperature=0,
                max_tokens=300,
                json_mode=True
            )

            return json.loads(response.content)

        except Exception as e:
            logger.warning(f"Error en verificación LLM: {e}")
            return self._verify_heuristic(claim, source)

    def _verify_heuristic(self, claim: str, source: str) -> Dict[str, Any]:
        """Verificación heurística sin LLM."""
        claim_lower = claim.lower()
        source_lower = source.lower()

        # Extraer palabras clave del claim
        words = set(re.findall(r'\b\w{4,}\b', claim_lower))
        # Quitar stopwords comunes
        stopwords = {'esta', 'esto', 'como', 'para', 'desde', 'sobre', 'entre', 'cuando'}
        keywords = words - stopwords

        if not keywords:
            return {
                "is_supported": False,
                "support_score": 0.0,
                "supporting_passage": "",
                "notes": "No se encontraron keywords en el claim"
            }

        # Contar keywords presentes en source
        matches = sum(1 for kw in keywords if kw in source_lower)
        ratio = matches / len(keywords)

        # Buscar pasaje con más matches
        sentences = source.split('.')
        best_sentence = ""
        best_score = 0

        for sent in sentences:
            sent_lower = sent.lower()
            sent_matches = sum(1 for kw in keywords if kw in sent_lower)
            if sent_matches > best_score:
                best_score = sent_matches
                best_sentence = sent.strip()

        is_supported = ratio >= 0.5
        support_score = min(ratio * 1.2, 1.0)

        return {
            "is_supported": is_supported,
            "support_score": support_score,
            "supporting_passage": best_sentence[:300] if is_supported else "",
            "notes": f"{matches}/{len(keywords)} keywords encontradas"
        }

    def enforce_grounding(
        self,
        response: str,
        contexts: List[Any],
        max_retries: int = 2
    ) -> Tuple[str, GroundingResult]:
        """
        Genera respuesta y asegura grounding mínimo.

        Si el grounding es insuficiente, regenera con más énfasis.

        Args:
            response: Respuesta inicial
            contexts: Contextos
            max_retries: Intentos máximos de regeneración

        Returns:
            Tuple (respuesta con buen grounding, resultado de verificación)
        """
        grounding = self.verify_citations(response, contexts)

        if grounding.is_acceptable(self.min_grounding_score):
            return response, grounding

        # Intentar mejorar
        for attempt in range(max_retries):
            logger.info(f"Grounding insuficiente ({grounding.grounding_score:.2f}), reintento {attempt+1}")

            # Regenerar con énfasis en citas
            improved = self._regenerate_with_feedback(
                response,
                contexts,
                grounding.unverified_citations,
                grounding.claims_without_citations
            )

            new_grounding = self.verify_citations(improved, contexts)

            if new_grounding.grounding_score > grounding.grounding_score:
                response = improved
                grounding = new_grounding

            if grounding.is_acceptable(self.min_grounding_score):
                break

        return response, grounding

    def _regenerate_with_feedback(
        self,
        response: str,
        contexts: List[Any],
        unverified: List[int],
        uncited: List[str]
    ) -> str:
        """Regenera respuesta con feedback de grounding."""
        from src.llm_provider import complete as llm_complete

        sources_text = "\n\n".join([
            f"[{i+1}] {c.doc_title}\n{c.content}"
            for i, c in enumerate(contexts)
        ])

        feedback = []
        if unverified:
            feedback.append(f"Las citas {unverified} no están bien soportadas por las fuentes.")
        if uncited:
            feedback.append(f"Estas afirmaciones necesitan cita: {uncited[:3]}")

        prompt = f"""Mejora esta respuesta para que TODAS las afirmaciones tengan citas [n] verificables.

RESPUESTA ORIGINAL:
{response}

PROBLEMAS DETECTADOS:
{' '.join(feedback)}

FUENTES DISPONIBLES:
{sources_text}

INSTRUCCIONES:
1. Reformula las afirmaciones problemáticas usando texto directo de las fuentes
2. Añade citas [n] a toda afirmación factual
3. Si no puedes citar algo, elimínalo
4. Sé más conservador: solo afirma lo que está claramente en las fuentes

Responde con la versión mejorada:"""

        response = llm_complete(
            prompt=prompt,
            system="Mejora la respuesta para maximizar el grounding en las fuentes.",
            temperature=0.2,
            max_tokens=2000
        )

        return response.content


class GroundedSynthesizer:
    """
    Sintetizador que garantiza grounding de citas.

    Wrapper sobre synthesizer normal que añade verificación.
    """

    def __init__(
        self,
        base_synthesizer,
        grounding_system: Optional[CitationGroundingSystem] = None,
        min_grounding: float = 0.8
    ):
        """
        Args:
            base_synthesizer: Sintetizador base
            grounding_system: Sistema de grounding (crea uno si None)
            min_grounding: Score mínimo de grounding
        """
        self.synthesizer = base_synthesizer
        self.grounding = grounding_system or CitationGroundingSystem(
            min_grounding_score=min_grounding
        )
        self.min_grounding = min_grounding

    def generate(
        self,
        query: str,
        results: List[Any],
        enforce_grounding: bool = True,
        **kwargs
    ):
        """
        Genera respuesta con grounding garantizado.

        Args:
            query: Query del usuario
            results: Resultados de retrieval
            enforce_grounding: Si forzar grounding mínimo
            **kwargs: Parámetros para synthesizer

        Returns:
            GeneratedResponse con metadata de grounding
        """
        # Generar con citas
        cited_response = self.grounding.generate_with_citations(query, results)

        if enforce_grounding:
            final_response, grounding_result = self.grounding.enforce_grounding(
                cited_response,
                results
            )
        else:
            final_response = cited_response
            grounding_result = self.grounding.verify_citations(cited_response, results)

        # Crear GeneratedResponse
        from .synthesizer import GeneratedResponse

        return GeneratedResponse(
            content=final_response,
            query=query,
            query_type="grounded",
            sources_used=[r.chunk_id for r in results],
            model="grounded_synthesis",
            tokens_input=0,
            tokens_output=0,
            latency_ms=0,
            metadata={
                "grounding": grounding_result.to_dict(),
                "grounding_score": grounding_result.grounding_score,
                "is_well_grounded": grounding_result.is_acceptable(self.min_grounding)
            }
        )
