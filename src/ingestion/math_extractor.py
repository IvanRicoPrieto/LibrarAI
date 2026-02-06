"""
Extractor de términos matemáticos para búsqueda math-aware.

Extrae y normaliza términos matemáticos de texto con LaTeX para mejorar
la búsqueda semántica. Convierte notación LaTeX a términos buscables.

Ejemplo:
    $\\sum_{i=1}^n$ → ["sumatorio", "summation", "sum"]
    $\\int_0^\\infty$ → ["integral", "integration"]
    $|\\psi\\rangle$ → ["ket", "estado cuántico", "quantum state"]
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MathExtractionResult:
    """Resultado de extracción de términos matemáticos."""
    terms: List[str]  # Términos únicos normalizados
    formulas: List[str]  # Fórmulas LaTeX originales
    symbols: List[str]  # Símbolos matemáticos detectados
    concepts: List[str]  # Conceptos matemáticos inferidos


# Mapeo de comandos LaTeX a términos buscables (ES + EN)
_LATEX_TO_TERMS: Dict[str, List[str]] = {
    # Operadores
    r"\sum": ["sumatorio", "summation", "sum", "suma"],
    r"\prod": ["productorio", "product", "producto"],
    r"\int": ["integral", "integration", "integración"],
    r"\oint": ["integral de contorno", "contour integral", "line integral"],
    r"\lim": ["límite", "limit"],
    r"\partial": ["derivada parcial", "partial derivative"],
    r"\nabla": ["gradiente", "gradient", "nabla", "del operator"],
    r"\Delta": ["laplaciano", "laplacian", "delta"],

    # Álgebra lineal
    r"\det": ["determinante", "determinant"],
    r"\Tr": ["traza", "trace"],
    r"\rank": ["rango", "rank"],
    r"\dim": ["dimensión", "dimension"],
    r"\ker": ["núcleo", "kernel"],
    r"\im": ["imagen", "image"],
    r"\span": ["espacio generado", "span"],

    # Productos y operaciones
    r"\otimes": ["producto tensorial", "tensor product", "kronecker"],
    r"\oplus": ["suma directa", "direct sum"],
    r"\times": ["producto cruz", "cross product"],
    r"\cdot": ["producto escalar", "dot product", "scalar product"],
    r"\wedge": ["producto exterior", "wedge product", "exterior product"],

    # Mecánica cuántica
    r"\bra": ["bra", "dual vector", "vector dual"],
    r"\ket": ["ket", "estado cuántico", "quantum state", "state vector"],
    r"\braket": ["producto interno", "inner product", "bracket"],
    r"\rho": ["matriz de densidad", "density matrix", "operador densidad"],
    r"\psi": ["función de onda", "wave function", "estado", "state"],
    r"\phi": ["fase", "phase", "estado", "state"],
    r"\hbar": ["constante de Planck reducida", "reduced Planck constant"],
    r"\hat": ["operador", "operator"],

    # Espacios
    r"\mathcal{H}": ["espacio de Hilbert", "Hilbert space"],
    r"\mathbb{R}": ["números reales", "real numbers", "reals"],
    r"\mathbb{C}": ["números complejos", "complex numbers"],
    r"\mathbb{Z}": ["enteros", "integers"],
    r"\mathbb{N}": ["naturales", "natural numbers"],

    # Funciones especiales
    r"\sin": ["seno", "sine"],
    r"\cos": ["coseno", "cosine"],
    r"\tan": ["tangente", "tangent"],
    r"\exp": ["exponencial", "exponential"],
    r"\log": ["logaritmo", "logarithm"],
    r"\ln": ["logaritmo natural", "natural logarithm"],
    r"\sqrt": ["raíz cuadrada", "square root"],

    # Relaciones
    r"\equiv": ["equivalente", "equivalent", "congruente"],
    r"\approx": ["aproximadamente", "approximately"],
    r"\propto": ["proporcional", "proportional"],
    r"\sim": ["similar", "distribuido como", "distributed as"],
    r"\subset": ["subconjunto", "subset"],
    r"\subseteq": ["subconjunto", "subset"],
    r"\in": ["pertenece", "belongs to", "element of"],
    r"\forall": ["para todo", "for all"],
    r"\exists": ["existe", "exists"],

    # Matrices y estructuras
    r"\begin{matrix}": ["matriz", "matrix"],
    r"\begin{pmatrix}": ["matriz", "matrix", "parenthesis matrix"],
    r"\begin{bmatrix}": ["matriz", "matrix", "bracket matrix"],
    r"\begin{vmatrix}": ["determinante", "determinant"],

    # Puertas cuánticas
    r"\text{CNOT}": ["CNOT", "controlled NOT", "puerta controlada"],
    r"\text{H}": ["Hadamard", "puerta Hadamard"],
    r"\text{X}": ["Pauli X", "NOT gate", "bit flip"],
    r"\text{Y}": ["Pauli Y"],
    r"\text{Z}": ["Pauli Z", "phase flip"],
    r"\text{T}": ["T gate", "puerta T", "pi/8 gate"],
    r"\text{S}": ["S gate", "puerta S", "phase gate"],
    r"\text{SWAP}": ["SWAP", "intercambio"],
    r"\text{Toffoli}": ["Toffoli", "CCNOT", "puerta Toffoli"],
}

# Patrones para detectar conceptos matemáticos en contexto
_CONCEPT_PATTERNS: Dict[str, List[str]] = {
    r"eigenvalue|autovalor|valor propio": ["eigenvalue", "autovalor", "valor propio"],
    r"eigenvector|autovector|vector propio": ["eigenvector", "autovector", "vector propio"],
    r"hermitian|hermítico|hermitiano": ["hermitiano", "hermitian", "self-adjoint"],
    r"unitary|unitario|unitaria": ["unitario", "unitary"],
    r"orthogonal|ortogonal": ["ortogonal", "orthogonal"],
    r"normalized?|normalizado": ["normalizado", "normalized"],
    r"entangle|entrelaz": ["entrelazamiento", "entanglement", "entangled"],
    r"superposition|superposición": ["superposición", "superposition"],
    r"decoherence|decoherencia": ["decoherencia", "decoherence"],
    r"measurement|medición|medida": ["medición", "measurement"],
    r"observable": ["observable"],
    r"hamiltonian|hamiltoniano": ["hamiltoniano", "hamiltonian"],
    r"lagrangian|lagrangiano": ["lagrangiano", "lagrangian"],
    r"commutator|conmutador": ["conmutador", "commutator"],
    r"probability|probabilidad": ["probabilidad", "probability"],
    r"amplitude|amplitud": ["amplitud", "amplitude"],
    r"phase|fase": ["fase", "phase"],
    r"fidelity|fidelidad": ["fidelidad", "fidelity"],
    r"coherence|coherencia": ["coherencia", "coherence"],
    r"teleportation|teleportación|teletransporte": ["teleportación", "teleportation"],
    r"fourier|Fourier": ["Fourier", "transformada de Fourier", "Fourier transform"],
    r"algorithm|algoritmo": ["algoritmo", "algorithm"],
    r"circuit|circuito": ["circuito", "circuit"],
    r"qubit": ["qubit", "bit cuántico", "quantum bit"],
    r"qutrit": ["qutrit"],
    r"register|registro": ["registro", "register"],
    r"oracle|oráculo": ["oráculo", "oracle"],
    r"ancilla": ["ancilla", "qubit auxiliar"],
}


class MathExtractor:
    """
    Extractor de términos matemáticos para búsqueda mejorada.

    Analiza texto con LaTeX y extrae términos buscables que permiten
    encontrar contenido matemático con queries en lenguaje natural.
    """

    def __init__(self, include_symbols: bool = True, max_terms: int = 50):
        """
        Args:
            include_symbols: Si incluir símbolos crudos además de términos
            max_terms: Máximo de términos a retornar
        """
        self.include_symbols = include_symbols
        self.max_terms = max_terms

        # Compilar patrones (sin escape - los comandos LaTeX ya son literales)
        self._latex_patterns = [
            (cmd, terms)  # No compilamos, usamos búsqueda directa de string
            for cmd, terms in _LATEX_TO_TERMS.items()
        ]
        self._concept_patterns = [
            (re.compile(pattern, re.IGNORECASE), terms)
            for pattern, terms in _CONCEPT_PATTERNS.items()
        ]

    def extract(self, text: str) -> MathExtractionResult:
        """
        Extrae términos matemáticos de un texto.

        Args:
            text: Texto con posible contenido LaTeX

        Returns:
            MathExtractionResult con términos, fórmulas, símbolos y conceptos
        """
        terms: Set[str] = set()
        formulas: List[str] = []
        symbols: Set[str] = set()
        concepts: Set[str] = set()

        # 1. Extraer fórmulas LaTeX
        inline_formulas = re.findall(r"\$([^$]+)\$", text)
        display_formulas = re.findall(r"\$\$([^$]+)\$\$", text)
        formulas = inline_formulas + display_formulas

        # 2. Buscar comandos LaTeX y mapear a términos
        for cmd, pattern_terms in self._latex_patterns:
            if cmd in text:
                terms.update(pattern_terms)

        # 3. Extraer símbolos griegos y matemáticos comunes
        if self.include_symbols:
            greek_symbols = re.findall(
                r"\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|"
                r"lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|"
                r"Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega)",
                text
            )
            symbols.update(greek_symbols)

        # 4. Detectar conceptos matemáticos por contexto
        for pattern, pattern_concepts in self._concept_patterns:
            if pattern.search(text):
                concepts.update(pattern_concepts)

        # 5. Detectar notación bra-ket específica
        if re.search(r"\|[^|>]+\s*[>⟩]|[<⟨]\s*[^|<]+\|", text) or r"\ket" in text or r"\bra" in text:
            concepts.update(["notación de Dirac", "Dirac notation", "bra-ket"])

        # 6. Detectar tipos específicos de problemas/algoritmos cuánticos
        quantum_algorithms = {
            r"shor|factoriza": ["algoritmo de Shor", "Shor's algorithm", "factorización"],
            r"grover|búsqueda": ["algoritmo de Grover", "Grover's algorithm", "búsqueda cuántica"],
            r"deutsch|jozsa": ["Deutsch-Jozsa", "algoritmo de Deutsch"],
            r"simon": ["algoritmo de Simon", "Simon's algorithm"],
            r"bernstein|vazirani": ["Bernstein-Vazirani"],
            r"qft|quantum fourier": ["QFT", "transformada de Fourier cuántica", "quantum Fourier transform"],
            r"vqe|variational": ["VQE", "variational quantum eigensolver"],
            r"qaoa": ["QAOA", "quantum approximate optimization"],
            r"qpe|phase estimation": ["QPE", "estimación de fase", "phase estimation"],
        }

        for pattern, algo_terms in quantum_algorithms.items():
            if re.search(pattern, text, re.IGNORECASE):
                concepts.update(algo_terms)

        # 7. Limitar y ordenar resultados
        all_terms = list(terms | concepts)[:self.max_terms]

        return MathExtractionResult(
            terms=sorted(all_terms),
            formulas=formulas[:20],  # Limitar fórmulas guardadas
            symbols=sorted(symbols),
            concepts=sorted(concepts)
        )

    def extract_batch(self, texts: List[str]) -> List[MathExtractionResult]:
        """Extrae términos de múltiples textos."""
        return [self.extract(text) for text in texts]

    def get_searchable_text(self, text: str) -> str:
        """
        Genera texto enriquecido para embedding con términos matemáticos.

        Añade términos extraídos al final del texto para mejorar
        la búsqueda semántica de contenido matemático.

        Args:
            text: Texto original

        Returns:
            Texto original + términos matemáticos como sufijo
        """
        result = self.extract(text)

        if not result.terms:
            return text

        # Crear sufijo con términos únicos
        unique_terms = list(set(result.terms + result.concepts))[:30]
        suffix = " | Términos: " + ", ".join(unique_terms)

        return text + suffix


def normalize_math_query(query: str) -> str:
    """
    Normaliza una query para búsqueda math-aware.

    Expande términos comunes a sus equivalentes matemáticos.

    Args:
        query: Query del usuario

    Returns:
        Query expandida con términos matemáticos
    """
    expansions = {
        r"\bsumatorio\b": "sumatorio summation sum sigma",
        r"\bintegral\b": "integral integration",
        r"\bderivada\b": "derivada derivative partial",
        r"\bmatriz\b": "matriz matrix matrices",
        r"\bdeterminante\b": "determinante determinant",
        r"\bautovalor\b": "autovalor eigenvalue valor propio",
        r"\bautovector\b": "autovector eigenvector vector propio",
        r"\bentrelazamiento\b": "entrelazamiento entanglement entangled",
        r"\bsuperposición\b": "superposición superposition",
        r"\bqubit\b": "qubit quantum bit",
        r"\bcircuito\b": "circuito circuit quantum circuit",
        r"\bpuerta\b": "puerta gate quantum gate",
        r"\bHadamard\b": "Hadamard H gate",
        r"\bCNOT\b": "CNOT controlled NOT CX",
    }

    result = query
    for pattern, expansion in expansions.items():
        if re.search(pattern, query, re.IGNORECASE):
            result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)

    return result


def formula_to_description(formula: str) -> str:
    """
    Convierte una fórmula LaTeX a descripción textual aproximada.

    Útil para búsqueda semántica cuando el usuario busca por descripción.

    Args:
        formula: Fórmula en LaTeX

    Returns:
        Descripción textual aproximada
    """
    descriptions = []

    # Detectar tipo de expresión
    if r"\sum" in formula:
        if r"_{" in formula and r"}^{" in formula:
            descriptions.append("sumatorio con límites")
        else:
            descriptions.append("sumatorio")

    if r"\int" in formula:
        if r"_0" in formula or r"_{0}" in formula:
            descriptions.append("integral definida")
        else:
            descriptions.append("integral")

    if r"\frac" in formula:
        descriptions.append("fracción")

    if r"\sqrt" in formula:
        descriptions.append("raíz cuadrada")

    if r"\lim" in formula:
        descriptions.append("límite")

    if r"|" in formula and r"\rangle" in formula or r"\ket" in formula:
        descriptions.append("estado cuántico ket")

    if r"\langle" in formula or r"\bra" in formula:
        descriptions.append("bra estado dual")

    if r"\otimes" in formula:
        descriptions.append("producto tensorial")

    if r"H" in formula and (r"\text{H}" in formula or r"\hat{H}" in formula):
        descriptions.append("Hamiltoniano u operador Hadamard")

    if r"\rho" in formula:
        descriptions.append("matriz de densidad")

    if not descriptions:
        return "expresión matemática"

    return ", ".join(descriptions)
