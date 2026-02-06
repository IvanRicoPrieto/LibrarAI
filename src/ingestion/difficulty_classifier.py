"""
Clasificador de nivel de dificultad para chunks.

Clasifica chunks en 4 niveles:
- introductory: Conceptos básicos, definiciones simples, primeros capítulos
- intermediate: Teoremas, demostraciones simples, aplicaciones
- advanced: Matemáticas complejas, demostraciones rigurosas
- research: Papers, resultados de vanguardia, referencias a literatura

Usa heurísticas rápidas por defecto, con opción de clasificación LLM.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    """Niveles de dificultad ordenados."""
    INTRODUCTORY = "introductory"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    RESEARCH = "research"


@dataclass
class DifficultyResult:
    """Resultado de clasificación de dificultad."""
    level: DifficultyLevel
    confidence: float  # 0.0 - 1.0
    reasoning: str  # Breve explicación


# Patrones heurísticos para detección de nivel
_INTRODUCTORY_PATTERNS = [
    r"\bintroducci[oó]n\b",
    r"\bb[aá]sico\b",
    r"\bfundamental\b",
    r"\bqu[eé] es\b",
    r"\bdefinici[oó]n\b",
    r"\bconcepto\b",
    r"\bprimero\b",
    r"\bsimple\b",
    r"\belemental\b",
    r"\bpreliminar\b",
    r"\bwhat is\b",
    r"\bbasic\b",
    r"\bintroduction\b",
    r"\bfundamentals?\b",
    r"\bdefinition\b",
    r"\bprimer\b",
]

_ADVANCED_PATTERNS = [
    r"\bdemostraci[oó]n\b",
    r"\bteorema\b",
    r"\bproof\b",
    r"\btheorem\b",
    r"\blemma\b",
    r"\bcorollary\b",
    r"\bproposition\b",
    r"\baxiom\b",
    r"\brigorous\b",
    r"\bformal\b",
    r"\bnon-trivial\b",
    r"\badvanced\b",
]

_RESEARCH_PATTERNS = [
    r"\bet al\.\b",
    r"\b\d{4}\b.*\bcite",  # Año + cite
    r"\barxiv\b",
    r"\bpreprint\b",
    r"\brecent work\b",
    r"\bstate[- ]of[- ]the[- ]art\b",
    r"\bnovel\b",
    r"\bwe propose\b",
    r"\bour contribution\b",
    r"\bexperimental results\b",
    r"\bopen problem\b",
    r"\bconjectured?\b",
]

# Indicadores matemáticos de complejidad
_COMPLEX_MATH_PATTERNS = [
    r"\\int_",  # Integrales definidas
    r"\\sum_",  # Sumatorios
    r"\\prod_",  # Productorios
    r"\\lim_",  # Límites
    r"\\partial",  # Derivadas parciales
    r"\\nabla",  # Gradiente
    r"\\otimes",  # Producto tensorial
    r"\\oplus",  # Suma directa
    r"\\mathcal{H}",  # Espacio de Hilbert
    r"\\bra\{.*\}\\ket\{.*\}",  # Notación bra-ket
    r"\\rho",  # Matrices de densidad
    r"\\Tr\b",  # Traza
    r"\\det\b",  # Determinante
    r"\\begin\{(align|equation|matrix|pmatrix)\}",  # Entornos matemáticos
]


class DifficultyClassifier:
    """
    Clasificador de nivel de dificultad para chunks de texto.

    Usa heurísticas por defecto (rápido, sin coste).
    Opcionalmente puede usar LLM para clasificación más precisa.
    """

    def __init__(self, use_llm: bool = False, batch_size: int = 20):
        """
        Args:
            use_llm: Si True, usa LLM para clasificación (más lento, más preciso)
            batch_size: Tamaño de batch para clasificación LLM
        """
        self.use_llm = use_llm
        self.batch_size = batch_size

        # Compilar patrones regex
        self._intro_patterns = [re.compile(p, re.IGNORECASE) for p in _INTRODUCTORY_PATTERNS]
        self._advanced_patterns = [re.compile(p, re.IGNORECASE) for p in _ADVANCED_PATTERNS]
        self._research_patterns = [re.compile(p, re.IGNORECASE) for p in _RESEARCH_PATTERNS]
        self._math_patterns = [re.compile(p) for p in _COMPLEX_MATH_PATTERNS]

    def classify(self, text: str, section_hierarchy: Optional[List[str]] = None) -> DifficultyResult:
        """
        Clasifica el nivel de dificultad de un texto.

        Args:
            text: Contenido del chunk
            section_hierarchy: Jerarquía de secciones ["Capítulo 1", "1.1 Introducción"]

        Returns:
            DifficultyResult con nivel, confianza y razonamiento
        """
        if self.use_llm:
            return self._classify_llm(text, section_hierarchy)
        return self._classify_heuristic(text, section_hierarchy)

    def classify_batch(self, chunks: List[Tuple[str, Optional[List[str]]]]) -> List[DifficultyResult]:
        """
        Clasifica un batch de chunks.

        Args:
            chunks: Lista de (text, section_hierarchy) tuples

        Returns:
            Lista de DifficultyResult
        """
        if self.use_llm:
            return self._classify_batch_llm(chunks)
        return [self._classify_heuristic(text, hierarchy) for text, hierarchy in chunks]

    def _classify_heuristic(
        self,
        text: str,
        section_hierarchy: Optional[List[str]] = None
    ) -> DifficultyResult:
        """Clasificación basada en heurísticas (rápido, sin coste)."""
        scores = {
            DifficultyLevel.INTRODUCTORY: 0.0,
            DifficultyLevel.INTERMEDIATE: 0.0,
            DifficultyLevel.ADVANCED: 0.0,
            DifficultyLevel.RESEARCH: 0.0,
        }
        reasons = []

        # 1. Analizar jerarquía de secciones
        if section_hierarchy:
            hierarchy_text = " ".join(section_hierarchy).lower()

            # Primeros capítulos suelen ser introductorios
            if any(x in hierarchy_text for x in ["capítulo 1", "chapter 1", "1.", "introducción", "introduction", "preliminaries", "background"]):
                scores[DifficultyLevel.INTRODUCTORY] += 2.0
                reasons.append("early_chapter")

            # Capítulos finales suelen ser avanzados
            if any(x in hierarchy_text for x in ["apéndice", "appendix", "advanced topics", "applications"]):
                scores[DifficultyLevel.ADVANCED] += 1.5
                reasons.append("late_chapter")

        # 2. Buscar patrones de texto
        intro_matches = sum(1 for p in self._intro_patterns if p.search(text))
        advanced_matches = sum(1 for p in self._advanced_patterns if p.search(text))
        research_matches = sum(1 for p in self._research_patterns if p.search(text))

        scores[DifficultyLevel.INTRODUCTORY] += intro_matches * 1.0
        scores[DifficultyLevel.ADVANCED] += advanced_matches * 1.5
        scores[DifficultyLevel.RESEARCH] += research_matches * 2.0

        if intro_matches > 0:
            reasons.append(f"intro_keywords:{intro_matches}")
        if advanced_matches > 0:
            reasons.append(f"advanced_keywords:{advanced_matches}")
        if research_matches > 0:
            reasons.append(f"research_keywords:{research_matches}")

        # 3. Analizar complejidad matemática
        math_complexity = sum(1 for p in self._math_patterns if p.search(text))

        if math_complexity == 0:
            # Sin matemáticas complejas → probablemente introductorio
            scores[DifficultyLevel.INTRODUCTORY] += 0.5
        elif math_complexity <= 2:
            scores[DifficultyLevel.INTERMEDIATE] += 1.0
            reasons.append(f"moderate_math:{math_complexity}")
        elif math_complexity <= 5:
            scores[DifficultyLevel.ADVANCED] += 1.5
            reasons.append(f"complex_math:{math_complexity}")
        else:
            scores[DifficultyLevel.ADVANCED] += 2.0
            reasons.append(f"heavy_math:{math_complexity}")

        # 4. Detectar densidad de fórmulas
        latex_count = len(re.findall(r"\$[^$]+\$|\$\$[^$]+\$\$", text))
        text_length = len(text)

        if text_length > 0:
            formula_density = latex_count / (text_length / 100)
            if formula_density > 5:
                scores[DifficultyLevel.ADVANCED] += 1.0
                reasons.append("high_formula_density")

        # 5. Determinar nivel final
        # Si no hay señales claras, default a intermediate
        if max(scores.values()) < 0.5:
            scores[DifficultyLevel.INTERMEDIATE] = 1.0
            reasons.append("default_intermediate")

        # Encontrar nivel con mayor score
        best_level = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_level] / total_score if total_score > 0 else 0.5

        return DifficultyResult(
            level=best_level,
            confidence=min(confidence, 1.0),
            reasoning=",".join(reasons) if reasons else "no_clear_signals"
        )

    def _classify_llm(
        self,
        text: str,
        section_hierarchy: Optional[List[str]] = None
    ) -> DifficultyResult:
        """Clasificación usando LLM (más preciso, con coste)."""
        try:
            from src.llm_provider import complete as llm_complete
        except ImportError:
            logger.warning("LLM provider not available, falling back to heuristics")
            return self._classify_heuristic(text, section_hierarchy)

        hierarchy_str = " > ".join(section_hierarchy) if section_hierarchy else "Unknown"

        prompt = f"""Clasifica el siguiente fragmento de texto académico según su nivel de dificultad.

Niveles posibles:
- introductory: Conceptos básicos, definiciones simples, sin matemáticas complejas
- intermediate: Teoremas básicos, demostraciones simples, matemáticas moderadas
- advanced: Demostraciones rigurosas, matemáticas complejas, conceptos avanzados
- research: Resultados de investigación, papers, problemas abiertos

Sección: {hierarchy_str}

Texto:
{text[:1500]}

Responde SOLO con JSON:
{{"level": "introductory|intermediate|advanced|research", "confidence": 0.0-1.0, "reasoning": "breve explicación"}}"""

        try:
            response = llm_complete(
                prompt=prompt,
                system="Eres un clasificador de nivel de dificultad para textos académicos. Responde solo con JSON.",
                json_mode=True,
                temperature=0.1,
                max_tokens=150,
            )

            import json
            result = json.loads(response.content)

            level_map = {
                "introductory": DifficultyLevel.INTRODUCTORY,
                "intermediate": DifficultyLevel.INTERMEDIATE,
                "advanced": DifficultyLevel.ADVANCED,
                "research": DifficultyLevel.RESEARCH,
            }

            return DifficultyResult(
                level=level_map.get(result.get("level", "intermediate"), DifficultyLevel.INTERMEDIATE),
                confidence=float(result.get("confidence", 0.7)),
                reasoning=result.get("reasoning", "llm_classified")
            )

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, falling back to heuristics")
            return self._classify_heuristic(text, section_hierarchy)

    def _classify_batch_llm(
        self,
        chunks: List[Tuple[str, Optional[List[str]]]]
    ) -> List[DifficultyResult]:
        """Clasificación en batch usando LLM."""
        results = []

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]

            # Por ahora, procesar uno a uno (podría optimizarse con batch prompts)
            for text, hierarchy in batch:
                results.append(self._classify_llm(text, hierarchy))

        return results


def get_difficulty_label(level: DifficultyLevel) -> str:
    """Obtiene etiqueta legible para un nivel de dificultad."""
    labels = {
        DifficultyLevel.INTRODUCTORY: "Introductorio",
        DifficultyLevel.INTERMEDIATE: "Intermedio",
        DifficultyLevel.ADVANCED: "Avanzado",
        DifficultyLevel.RESEARCH: "Investigación",
    }
    return labels.get(level, "Desconocido")


def difficulty_to_numeric(level: DifficultyLevel) -> int:
    """Convierte nivel de dificultad a valor numérico (1-4)."""
    mapping = {
        DifficultyLevel.INTRODUCTORY: 1,
        DifficultyLevel.INTERMEDIATE: 2,
        DifficultyLevel.ADVANCED: 3,
        DifficultyLevel.RESEARCH: 4,
    }
    return mapping.get(level, 2)
