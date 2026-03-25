"""
FormulaRetriever — Retrieval aumentado con búsqueda por equivalencia simbólica.

Conecta el FormulaGraph (Fase 5) con el pipeline de retrieval:
1. Extrae fórmulas de los chunks recuperados y las indexa
2. Permite buscar por equivalencia simbólica además de por texto
3. Enriquece los resultados de retrieval con relaciones matemáticas

Se integra como post-procesador del retrieval existente, no lo reemplaza.
"""

import logging
import re
from typing import List, Optional, Dict, Any, Tuple

from .formula_graph import FormulaGraph, FormulaNode, RelationType
from .engine import MathEngine

logger = logging.getLogger(__name__)


# Patrones para extraer fórmulas LaTeX de texto
LATEX_INLINE = re.compile(r'\$([^$]+)\$')
LATEX_BLOCK = re.compile(r'\$\$([^$]+)\$\$')
LATEX_ENV = re.compile(r'\\begin\{(?:equation|align|gather)\*?\}(.*?)\\end\{(?:equation|align|gather)\*?\}', re.DOTALL)


class FormulaRetriever:
    """
    Enhancer de retrieval con capacidades de búsqueda simbólica.

    No reemplaza el retrieval existente — lo aumenta:
    - Extrae fórmulas de chunks y las indexa en un FormulaGraph
    - Permite buscar chunks por equivalencia simbólica
    - Identifica relaciones entre fórmulas de diferentes fuentes
    """

    def __init__(self, engine: Optional[MathEngine] = None):
        self.engine = engine or MathEngine()
        self.graph = FormulaGraph(math_engine=self.engine)
        self._chunk_formulas: Dict[str, List[str]] = {}  # chunk_id → formula_node_ids

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Extrae e indexa fórmulas de una lista de chunks.

        Args:
            chunks: Lista de dicts con al menos 'chunk_id' y 'content'

        Returns:
            Número de fórmulas indexadas
        """
        total = 0
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            content = chunk.get("content", "")
            doc_title = chunk.get("doc_title", "")

            formulas = self._extract_formulas(content)
            node_ids = []

            for latex, sympy_str in formulas:
                if not sympy_str:
                    continue

                node_id = self.graph.add_formula(
                    expression=sympy_str,
                    latex=latex,
                    source_chunks=[chunk_id],
                    description=f"From {doc_title}",
                    domain=self._detect_domain(content),
                )
                node_ids.append(node_id)
                total += 1

            if node_ids:
                self._chunk_formulas[chunk_id] = node_ids

        logger.info(f"FormulaRetriever: indexadas {total} fórmulas de {len(chunks)} chunks")
        return total

    def search_by_formula(
        self,
        expression: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Busca chunks que contienen fórmulas equivalentes.

        Args:
            expression: Expresión SymPy a buscar
            top_k: Máximo resultados

        Returns:
            Lista de dicts con chunk_ids y scores de equivalencia
        """
        # Buscar por fingerprint
        matches = self.graph.find_by_fingerprint(expression)

        # Recoger chunk_ids de los matches
        results = []
        seen_chunks = set()

        for node in matches[:top_k]:
            for chunk_id in node.source_chunks:
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    results.append({
                        "chunk_id": chunk_id,
                        "formula_match": node.expression,
                        "formula_latex": node.latex,
                        "equivalence_score": 1.0,
                    })

        return results

    def enrich_results(
        self,
        retrieval_results: List[Any],
    ) -> List[Any]:
        """
        Enriquece resultados de retrieval con metadata de fórmulas.

        Añade información sobre las fórmulas encontradas en cada chunk
        y las relaciones entre ellas.
        """
        for result in retrieval_results:
            chunk_id = getattr(result, 'chunk_id', '') or result.get('chunk_id', '')

            if chunk_id in self._chunk_formulas:
                node_ids = self._chunk_formulas[chunk_id]
                formulas = []

                for nid in node_ids:
                    if nid in self.graph.nodes:
                        node = self.graph.nodes[nid]
                        formulas.append({
                            "expression": node.expression,
                            "latex": node.latex,
                            "fingerprint": node.fingerprint.hash if node.fingerprint else "",
                        })

                # Añadir a metadata
                if hasattr(result, 'metadata'):
                    result.metadata["formulas"] = formulas
                    result.metadata["n_formulas"] = len(formulas)

        return retrieval_results

    def find_related_formulas(
        self,
        expression: str,
    ) -> List[Tuple[FormulaNode, str]]:
        """
        Busca fórmulas relacionadas (no necesariamente equivalentes).

        Busca por estructura similar (mismo número de variables,
        mismas operaciones, etc.).
        """
        fp = self.graph.fp_engine.fingerprint(expression)

        related = []

        # Buscar por propiedades estructurales
        candidates = self.graph.find_by_structure(
            has_trig=fp.has_trig,
            has_exp=fp.has_exp,
        )

        for node in candidates:
            if node.expression != expression:
                related.append((node, "structural_match"))

        return related

    def _extract_formulas(self, text: str) -> List[Tuple[str, str]]:
        """
        Extrae fórmulas LaTeX de texto y las convierte a SymPy.

        Returns:
            Lista de (latex_original, sympy_string)
        """
        from .latex_parser import LaTeXParser
        parser = LaTeXParser(use_llm_normalization=False)

        formulas = []

        # Extraer LaTeX inline y block
        for pattern in [LATEX_BLOCK, LATEX_INLINE]:
            for match in pattern.finditer(text):
                latex = match.group(1).strip()
                if len(latex) < 3:  # Demasiado corto para ser útil
                    continue

                sympy_str, confidence = parser.parse(latex)
                if sympy_str and confidence >= 0.5:
                    formulas.append((latex, sympy_str))

        return formulas

    @staticmethod
    def _detect_domain(content: str) -> str:
        """Detecta el dominio matemático del contenido."""
        content_lower = content.lower()

        domains = {
            "quantum": ["qubit", "quantum", "hilbert", "bra", "ket", "unitary", "hermitian"],
            "linear_algebra": ["matrix", "eigenvalue", "vector", "determinant", "trace"],
            "calculus": ["integral", "derivative", "limit", "series", "convergence"],
            "trigonometry": ["sin", "cos", "tan", "trigonometric", "angle"],
            "algebra": ["group", "ring", "field", "polynomial", "factor"],
        }

        for domain, keywords in domains.items():
            if any(kw in content_lower for kw in keywords):
                return domain

        return "general"

    def get_graph_stats(self) -> Dict[str, Any]:
        """Estadísticas del grafo de fórmulas."""
        return self.graph.to_dict()["stats"]
