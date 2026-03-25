"""
LaTeXParser — Conversión LaTeX → SymPy con scoring de confianza.

Estrategia híbrida:
1. (Opcional) LLM normaliza LaTeX desordenado/ambiguo
2. latex2sympy2 intenta parsing determinístico
3. Si falla, usa Lark grammar como fallback
4. Confidence scoring por round-trip: SymPy → LaTeX → comparar con original
"""

import logging
import re
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class LaTeXParser:
    """
    Parser híbrido LaTeX → SymPy.

    Combina parsing determinístico (latex2sympy2) con normalización
    LLM opcional para LaTeX desordenado de libros de texto.
    """

    def __init__(self, use_llm_normalization: bool = True):
        self.use_llm_normalization = use_llm_normalization
        self._latex2sympy_available = self._check_latex2sympy()

    def parse(self, latex: str) -> Tuple[str, float]:
        """
        Parsea una expresión LaTeX a string SymPy.

        Args:
            latex: Expresión en LaTeX (ej: "\\frac{x^2 + 1}{x - 1}")

        Returns:
            (sympy_expression_string, confidence_score)
            confidence_score: 0.0 (fallo) a 1.0 (verificado round-trip)
        """
        # Limpiar LaTeX
        latex_clean = self._clean_latex(latex)

        # Intentar parsing determinístico primero
        result = self._parse_deterministic(latex_clean)
        if result is not None:
            confidence = self._score_confidence(latex_clean, result)
            logger.info(f"LaTeX parseado con latex2sympy2: confianza={confidence:.2f}")
            return result, confidence

        # Si falla y LLM normalization está activada, intentar con LLM
        if self.use_llm_normalization:
            normalized = self._normalize_with_llm(latex_clean)
            if normalized and normalized != latex_clean:
                result = self._parse_deterministic(normalized)
                if result is not None:
                    confidence = self._score_confidence(latex_clean, result) * 0.9  # Penalizar por paso extra
                    logger.info(f"LaTeX parseado tras normalización LLM: confianza={confidence:.2f}")
                    return result, confidence

        # Último recurso: parsing manual con regex
        result = self._parse_manual(latex_clean)
        if result is not None:
            confidence = self._score_confidence(latex_clean, result) * 0.7  # Penalizar por método menos fiable
            logger.info(f"LaTeX parseado con regex manual: confianza={confidence:.2f}")
            return result, confidence

        logger.warning(f"No se pudo parsear LaTeX: {latex[:50]}...")
        return "", 0.0

    def _clean_latex(self, latex: str) -> str:
        """Limpia LaTeX de artefactos comunes."""
        # Quitar delimitadores de entorno math
        latex = re.sub(r'^\$+|\$+$', '', latex.strip())
        latex = re.sub(r'^\\[(\[]|\\[)\]]$', '', latex.strip())
        # Normalizar espacios
        latex = re.sub(r'\s+', ' ', latex).strip()
        # Quitar \displaystyle, \textstyle, etc.
        latex = re.sub(r'\\(displaystyle|textstyle|scriptstyle)\s*', '', latex)
        # Normalizar \left( y \right) a simples paréntesis
        latex = re.sub(r'\\left\s*', '', latex)
        latex = re.sub(r'\\right\s*', '', latex)
        return latex

    def _parse_deterministic(self, latex: str) -> Optional[str]:
        """Intenta parsing con latex2sympy2."""
        if not self._latex2sympy_available:
            return None

        try:
            from latex2sympy2 import latex2sympy
            expr = latex2sympy(latex)
            return str(expr)
        except Exception as e:
            logger.debug(f"latex2sympy2 falló: {e}")
            return None

    def _normalize_with_llm(self, latex: str) -> Optional[str]:
        """Usa LLM para normalizar LaTeX ambiguo."""
        try:
            from src.llm_provider import complete as llm_complete

            response = llm_complete(
                prompt=f"""Normaliza esta expresión LaTeX para que sea parseable.
Corrige espaciado, llaves faltantes, y ambigüedades.
Devuelve SOLO la expresión LaTeX normalizada, sin explicaciones.

Expresión: {latex}""",
                system="Eres un experto en LaTeX. Normaliza expresiones manteniendo su significado matemático.",
                temperature=0.0,
                max_tokens=200,
            )
            normalized = response.content.strip()
            # Limpiar posibles backticks
            normalized = re.sub(r'^```.*?\n|```$', '', normalized, flags=re.MULTILINE).strip()
            normalized = re.sub(r'^\$+|\$+$', '', normalized.strip())
            return normalized if normalized else None
        except Exception as e:
            logger.debug(f"LLM normalization falló: {e}")
            return None

    def _parse_manual(self, latex: str) -> Optional[str]:
        """Parsing manual con regex para patrones comunes."""
        try:
            result = latex

            # \frac{a}{b} → (a)/(b)
            result = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', result)

            # \sqrt{x} → sqrt(x)
            result = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', result)
            result = re.sub(r'\\sqrt\[(\d+)\]\{([^}]*)\}', r'(\2)**(1/\1)', result)

            # Funciones trigonométricas
            for func in ['sin', 'cos', 'tan', 'log', 'ln', 'exp']:
                result = re.sub(rf'\\{func}\s*', f'{func}', result)

            # \pi, \infty, etc.
            result = result.replace(r'\pi', 'pi')
            result = result.replace(r'\infty', 'oo')
            result = result.replace(r'\hbar', 'hbar')

            # Superscripts: x^{2} → x**2, x^2 → x**2
            result = re.sub(r'\^{([^}]*)}', r'**(\1)', result)
            result = re.sub(r'\^(\w)', r'**\1', result)

            # Subscripts: quitar (para SymPy no tienen sentido directo)
            result = re.sub(r'_\{[^}]*\}', '', result)
            result = re.sub(r'_\w', '', result)

            # \cdot → *
            result = result.replace(r'\cdot', '*')
            result = result.replace(r'\times', '*')

            # Multiplicación implícita: 2x → 2*x, 3pi → 3*pi
            result = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', result)
            # Multiplicación implícita por yuxtaposición: "pi r" → "pi*r", "x y" → "x*y"
            result = re.sub(r'([a-zA-Z0-9)]) ([a-zA-Z(])', r'\1*\2', result)

            # Limpiar backslashes restantes
            result = re.sub(r'\\[a-zA-Z]+', '', result)

            # Verificar que es parseable por SymPy
            import sympy as sp
            expr = sp.sympify(result)
            return str(expr)
        except Exception:
            return None

    def _score_confidence(self, original_latex: str, sympy_str: str) -> float:
        """
        Calcula confianza por round-trip: SymPy → LaTeX → comparar con original.

        Score basado en similitud entre el LaTeX original y el LaTeX
        generado a partir de la expresión SymPy parseada.
        """
        try:
            import sympy as sp
            expr = sp.sympify(sympy_str)
            roundtrip_latex = sp.latex(expr)

            # Normalizar ambos para comparar
            orig_norm = self._normalize_for_comparison(original_latex)
            rt_norm = self._normalize_for_comparison(roundtrip_latex)

            if orig_norm == rt_norm:
                return 1.0

            # Similitud basada en tokens
            orig_tokens = set(re.findall(r'[a-zA-Z]+|\d+|[^\s\w]', orig_norm))
            rt_tokens = set(re.findall(r'[a-zA-Z]+|\d+|[^\s\w]', rt_norm))

            if not orig_tokens:
                return 0.5

            intersection = orig_tokens & rt_tokens
            union = orig_tokens | rt_tokens
            jaccard = len(intersection) / len(union) if union else 0.0

            return min(1.0, jaccard + 0.3)  # Base score + jaccard

        except Exception:
            return 0.3

    @staticmethod
    def _normalize_for_comparison(latex: str) -> str:
        """Normaliza LaTeX para comparación (quita espacios, backslashes decorativos)."""
        s = re.sub(r'\s+', '', latex)
        s = s.replace(r'\left', '').replace(r'\right', '')
        s = s.replace(r'\displaystyle', '')
        s = s.replace(r'\,', '').replace(r'\;', '').replace(r'\!', '')
        return s

    @staticmethod
    def _check_latex2sympy() -> bool:
        """Comprueba si latex2sympy2 está disponible."""
        try:
            import latex2sympy2
            return True
        except ImportError:
            logger.info("latex2sympy2 no disponible — usando parsing manual")
            return False
