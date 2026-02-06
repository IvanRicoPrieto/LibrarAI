"""
Response Critic - Evalúa y mejora respuestas generadas.

Funcionalidades:
- Verificación de citas
- Detección de alucinaciones
- Evaluación de completitud
- Sugerencias de mejora
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import re

from ..retrieval.fusion import RetrievalResult
from ..generation.synthesizer import GeneratedResponse

logger = logging.getLogger(__name__)


class CritiqueCategory(Enum):
    """Categorías de crítica."""
    HALLUCINATION = "hallucination"  # Información no en fuentes
    MISSING_CITATION = "missing_citation"  # Falta cita
    INCOMPLETE = "incomplete"  # Respuesta incompleta
    INCONSISTENT = "inconsistent"  # Contradicción con fuentes
    QUALITY = "quality"  # Calidad general


@dataclass
class CritiqueIssue:
    """Problema detectado en la respuesta."""
    category: CritiqueCategory
    severity: str  # "low", "medium", "high"
    description: str
    location: Optional[str] = None  # Fragmento problemático
    suggestion: Optional[str] = None


@dataclass
class CritiqueResult:
    """Resultado de la evaluación crítica."""
    overall_score: float  # 0-1
    issues: List[CritiqueIssue]
    strengths: List[str]
    suggestions: List[str]
    citation_coverage: float  # % de fuentes citadas
    grounded_score: float  # % fundamentado en fuentes
    
    def to_dict(self) -> Dict:
        return {
            "overall_score": self.overall_score,
            "issues": [
                {
                    "category": i.category.value,
                    "severity": i.severity,
                    "description": i.description,
                    "location": i.location,
                    "suggestion": i.suggestion
                }
                for i in self.issues
            ],
            "strengths": self.strengths,
            "suggestions": self.suggestions,
            "citation_coverage": self.citation_coverage,
            "grounded_score": self.grounded_score
        }
    
    def passes_quality_threshold(self, threshold: float = 0.7) -> bool:
        """Verifica si pasa el umbral de calidad."""
        return self.overall_score >= threshold


class ResponseCritic:
    """
    Crítico de respuestas RAG.
    
    Evalúa:
    - Fundamentación en fuentes
    - Cobertura de citas
    - Detección de alucinaciones
    - Calidad general
    """
    
    def __init__(
        self,
        use_llm_critic: bool = False,
        strict_mode: bool = False
    ):
        """
        Args:
            use_llm_critic: Si usar LLM para crítica
            strict_mode: Si ser estricto con alucinaciones
        """
        self.use_llm_critic = use_llm_critic
        self.strict_mode = strict_mode
    
    def critique(
        self,
        response: GeneratedResponse,
        sources: List[RetrievalResult],
        query: str,
        validate_citations: bool = True
    ) -> CritiqueResult:
        """
        Evalúa una respuesta generada.
        
        Args:
            response: Respuesta a evaluar
            sources: Fuentes usadas
            query: Consulta original
            validate_citations: Si validar que cada cita tenga soporte real
            
        Returns:
            Resultado de la crítica
        """
        issues = []
        strengths = []
        suggestions = []
        
        # 1. Verificar cobertura de citas
        citation_coverage = self._check_citation_coverage(response.content, sources)
        
        if citation_coverage < 0.5:
            issues.append(CritiqueIssue(
                category=CritiqueCategory.MISSING_CITATION,
                severity="medium",
                description=f"Solo {citation_coverage*100:.0f}% de fuentes citadas",
                suggestion="Añadir más referencias a las fuentes disponibles"
            ))
        elif citation_coverage > 0.8:
            strengths.append("Excelente uso de citas bibliográficas")
        
        # 1.5 Validación profunda de citas (nuevo)
        if validate_citations:
            citation_issues = self._validate_citations(response.content, sources)
            issues.extend(citation_issues)
            
            if not citation_issues:
                strengths.append("Todas las citas tienen soporte en las fuentes")
            else:
                high_severity = sum(1 for i in citation_issues if i.severity == "high")
                if high_severity > 0:
                    suggestions.append(f"Revisar {high_severity} citas con bajo soporte en fuentes")
        
        # 2. Verificar fundamentación
        grounded_score = self._check_grounding(response.content, sources)
        
        if grounded_score < 0.6:
            issues.append(CritiqueIssue(
                category=CritiqueCategory.HALLUCINATION,
                severity="high" if self.strict_mode else "medium",
                description="Posible contenido no fundamentado en fuentes",
                suggestion="Verificar que toda la información proviene de las fuentes"
            ))
        elif grounded_score > 0.85:
            strengths.append("Respuesta bien fundamentada en las fuentes")
        
        # 3. Verificar completitud
        completeness = self._check_completeness(response.content, query, sources)
        
        if completeness < 0.5:
            issues.append(CritiqueIssue(
                category=CritiqueCategory.INCOMPLETE,
                severity="medium",
                description="La respuesta podría ser más completa",
                suggestion="Considerar incluir más información de las fuentes"
            ))
        
        # 4. Verificar consistencia
        inconsistencies = self._check_consistency(response.content, sources)
        issues.extend(inconsistencies)
        
        if not inconsistencies:
            strengths.append("No se detectaron inconsistencias")
        
        # 5. Evaluación con LLM (opcional)
        if self.use_llm_critic:
            llm_critique = self._critique_with_llm(response.content, sources, query)
            issues.extend(llm_critique.get("issues", []))
            suggestions.extend(llm_critique.get("suggestions", []))
        
        # Calcular score general
        severity_weights = {"low": 0.1, "medium": 0.25, "high": 0.4}
        penalty = sum(severity_weights.get(i.severity, 0.2) for i in issues)
        overall_score = max(0, 1 - penalty)
        
        # Ajustar con métricas
        overall_score = (
            overall_score * 0.4 +
            citation_coverage * 0.3 +
            grounded_score * 0.3
        )
        
        # Generar sugerencias si no hay
        if not suggestions and issues:
            suggestions = [i.suggestion for i in issues if i.suggestion]
        
        result = CritiqueResult(
            overall_score=overall_score,
            issues=issues,
            strengths=strengths,
            suggestions=suggestions,
            citation_coverage=citation_coverage,
            grounded_score=grounded_score
        )
        
        logger.info(
            f"Crítica completada: score={overall_score:.2f}, "
            f"issues={len(issues)}, grounded={grounded_score:.2f}"
        )
        
        return result
    
    def _check_citation_coverage(
        self,
        content: str,
        sources: List[RetrievalResult]
    ) -> float:
        """Verifica qué porcentaje de fuentes se citaron."""
        if not sources:
            return 1.0
        
        # Encontrar citas [n] en el texto
        citation_pattern = re.compile(r'\[(\d+)\]')
        cited_indices = set()
        
        for match in citation_pattern.finditer(content):
            try:
                idx = int(match.group(1))
                if 1 <= idx <= len(sources):
                    cited_indices.add(idx)
            except ValueError:
                pass
        
        return len(cited_indices) / len(sources)
    
    def _validate_citations(
        self,
        content: str,
        sources: List[RetrievalResult]
    ) -> List[CritiqueIssue]:
        """
        Valida que cada cita [n] en la respuesta tenga soporte real en la fuente n.
        
        Esta es una validación más estricta que verifica que el contenido
        citado realmente aparece en la fuente referenciada.
        """
        issues = []
        
        if not sources:
            return issues
        
        # Encontrar todas las afirmaciones con citas: "texto [n]" o "texto[n]"
        citation_pattern = re.compile(r'([^.!?\n]+?)(?:\s*\[(\d+(?:,\s*\d+)*)\])', re.MULTILINE)
        
        for match in citation_pattern.finditer(content):
            claim = match.group(1).strip()
            citation_refs = match.group(2)
            
            # Extraer todos los índices citados
            indices = [int(i.strip()) for i in citation_refs.split(',')]
            
            # Verificar cada cita
            for idx in indices:
                if idx < 1 or idx > len(sources):
                    issues.append(CritiqueIssue(
                        category=CritiqueCategory.MISSING_CITATION,
                        severity="high",
                        description=f"Cita [{idx}] referencia fuente inexistente (solo hay {len(sources)} fuentes)",
                        location=claim[:50] + "...",
                        suggestion=f"Corregir referencia [{idx}]"
                    ))
                    continue
                
                source = sources[idx - 1]
                
                # Extraer palabras clave de la afirmación (>4 chars)
                claim_keywords = set(re.findall(r'\b\w{5,}\b', claim.lower()))
                source_text = source.content.lower()
                
                # Contar cuántas keywords aparecen en la fuente
                if claim_keywords:
                    matched = sum(1 for kw in claim_keywords if kw in source_text)
                    coverage = matched / len(claim_keywords)
                    
                    if coverage < 0.3:  # Menos del 30% de overlap
                        issues.append(CritiqueIssue(
                            category=CritiqueCategory.HALLUCINATION,
                            severity="medium" if coverage > 0.1 else "high",
                            description=f"Cita [{idx}] tiene bajo soporte en la fuente ({coverage*100:.0f}% overlap)",
                            location=claim[:80] + "..." if len(claim) > 80 else claim,
                            suggestion=f"Verificar que [{idx}] respalda: '{claim[:50]}...'"
                        ))
        
        return issues
    
    def _check_grounding(
        self,
        content: str,
        sources: List[RetrievalResult]
    ) -> float:
        """Verifica qué tan fundamentada está la respuesta."""
        if not sources:
            return 0.5  # Neutral si no hay fuentes
        
        # Crear conjunto de términos importantes de las fuentes
        source_terms = set()
        for source in sources:
            # Extraer términos significativos (>4 chars)
            words = re.findall(r'\b\w{5,}\b', source.content.lower())
            source_terms.update(words)
        
        # Contar términos de la respuesta que están en las fuentes
        response_words = re.findall(r'\b\w{5,}\b', content.lower())
        
        if not response_words:
            return 0.5
        
        grounded_words = sum(1 for w in response_words if w in source_terms)
        
        return grounded_words / len(response_words)
    
    def _check_completeness(
        self,
        content: str,
        query: str,
        sources: List[RetrievalResult]
    ) -> float:
        """Verifica si la respuesta aborda la consulta completamente."""
        # Extraer keywords de la query
        query_terms = set(re.findall(r'\b\w{4,}\b', query.lower()))
        
        # Verificar presencia en respuesta
        content_lower = content.lower()
        addressed = sum(1 for term in query_terms if term in content_lower)
        
        if not query_terms:
            return 1.0
        
        return addressed / len(query_terms)
    
    def _check_consistency(
        self,
        content: str,
        sources: List[RetrievalResult]
    ) -> List[CritiqueIssue]:
        """Detecta posibles inconsistencias con las fuentes."""
        issues = []
        
        # Buscar números en la respuesta
        numbers_in_response = re.findall(r'\b\d+(?:\.\d+)?\b', content)
        
        # Verificar si los números aparecen en las fuentes
        source_numbers = set()
        for source in sources:
            source_numbers.update(re.findall(r'\b\d+(?:\.\d+)?\b', source.content))
        
        for num in numbers_in_response:
            if num not in source_numbers and len(num) > 1:
                # Número no en fuentes (podría ser alucinación)
                issues.append(CritiqueIssue(
                    category=CritiqueCategory.INCONSISTENT,
                    severity="low",
                    description=f"Número '{num}' no encontrado en las fuentes",
                    location=num,
                    suggestion="Verificar la precisión de los datos numéricos"
                ))
        
        return issues[:3]  # Limitar issues de este tipo
    
    def _critique_with_llm(
        self,
        content: str,
        sources: List[RetrievalResult],
        query: str
    ) -> Dict[str, Any]:
        """Crítica usando LLM."""
        from src.llm_provider import complete as llm_complete

        # Preparar contexto de fuentes
        sources_text = "\n\n".join(
            f"[{i+1}] {s.content[:500]}..."
            for i, s in enumerate(sources[:5])
        )

        prompt = f"""Evalúa esta respuesta RAG por calidad y precisión.

PREGUNTA: {query}

FUENTES DISPONIBLES:
{sources_text}

RESPUESTA A EVALUAR:
{content}

Identifica:
1. Posibles alucinaciones (info no en fuentes)
2. Información faltante importante
3. Inconsistencias con las fuentes
4. Sugerencias de mejora

Responde en JSON:
{{
    "issues": [
        {{
            "type": "hallucination|missing|inconsistent",
            "severity": "low|medium|high",
            "description": "...",
            "suggestion": "..."
        }}
    ],
    "suggestions": ["..."]
}}"""

        try:
            response = llm_complete(prompt=prompt, json_mode=True, temperature=0.1)

            import json
            result = json.loads(response.content)

            # Convertir a CritiqueIssue
            issues = []
            type_mapping = {
                "hallucination": CritiqueCategory.HALLUCINATION,
                "missing": CritiqueCategory.INCOMPLETE,
                "inconsistent": CritiqueCategory.INCONSISTENT
            }

            for i in result.get("issues", []):
                category = type_mapping.get(
                    i.get("type", ""),
                    CritiqueCategory.QUALITY
                )
                issues.append(CritiqueIssue(
                    category=category,
                    severity=i.get("severity", "medium"),
                    description=i.get("description", ""),
                    suggestion=i.get("suggestion")
                ))

            return {
                "issues": issues,
                "suggestions": result.get("suggestions", [])
            }

        except Exception as e:
            logger.warning(f"LLM critique failed: {e}")
            return {"issues": [], "suggestions": []}


class IterativeRefiner:
    """
    Refinador iterativo de respuestas.
    
    Usa el crítico para mejorar respuestas
    hasta alcanzar un umbral de calidad.
    """
    
    def __init__(
        self,
        critic: ResponseCritic,
        synthesizer: Any,  # ResponseSynthesizer
        max_iterations: int = 3,
        quality_threshold: float = 0.8
    ):
        self.critic = critic
        self.synthesizer = synthesizer
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
    
    def refine(
        self,
        initial_response: GeneratedResponse,
        sources: List[RetrievalResult],
        query: str
    ) -> GeneratedResponse:
        """
        Refina iterativamente una respuesta.
        
        Args:
            initial_response: Respuesta inicial
            sources: Fuentes
            query: Consulta original
            
        Returns:
            Respuesta refinada
        """
        current_response = initial_response
        
        for iteration in range(self.max_iterations):
            critique = self.critic.critique(current_response, sources, query)
            
            if critique.passes_quality_threshold(self.quality_threshold):
                logger.info(f"Respuesta aceptada en iteración {iteration + 1}")
                break
            
            # Crear prompt de refinamiento
            refinement_prompt = self._create_refinement_prompt(
                current_response.content,
                critique,
                query
            )
            
            # Regenerar con feedback usando LLM directamente
            logger.info(f"Iteración {iteration + 1}: score={critique.overall_score:.2f}")

            try:
                from src.llm_provider import complete as llm_complete

                llm_response = llm_complete(
                    prompt=refinement_prompt,
                    system=(
                        "Eres un experto en computación cuántica y matemáticas. "
                        "Mejora la respuesta abordando los problemas detectados. "
                        "Mantén las citas a fuentes y la precisión técnica."
                    ),
                    temperature=0.2,
                    max_tokens=2000,
                )

                current_response = GeneratedResponse(
                    content=llm_response.content,
                    query=query,
                    query_type=current_response.query_type,
                    sources_used=current_response.sources_used,
                    model=llm_response.model,
                    tokens_input=llm_response.tokens_input,
                    tokens_output=llm_response.tokens_output,
                    latency_ms=current_response.latency_ms,
                    metadata={
                        **current_response.metadata,
                        "refinement_iteration": iteration + 1,
                        "pre_refinement_score": critique.overall_score,
                    },
                )

            except Exception as e:
                logger.warning(f"Refinamiento fallido en iteración {iteration + 1}: {e}")
                break

        return current_response
    
    def _create_refinement_prompt(
        self,
        current_content: str,
        critique: CritiqueResult,
        query: str
    ) -> str:
        """Crea prompt para refinamiento."""
        issues_text = "\n".join(
            f"- {i.description}" for i in critique.issues
        )
        
        return f"""Mejora esta respuesta basándote en el feedback:

RESPUESTA ACTUAL:
{current_content}

PROBLEMAS DETECTADOS:
{issues_text}

SUGERENCIAS:
{chr(10).join(f"- {s}" for s in critique.suggestions)}

Genera una versión mejorada que aborde estos problemas."""
