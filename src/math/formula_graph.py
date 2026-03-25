"""
Formula Graph — Knowledge Graph Computacional con fingerprinting simbólico.

Fase 5 del roadmap:
- Fingerprinting simbólico de fórmulas (invariantes bajo renombrado)
- Relaciones matemáticas formales entre ecuaciones
- E-graph con reglas de reescritura inyectadas de fuentes
- Búsqueda por equivalencia simbólica

Cada nodo del grafo es una fórmula con su fingerprint, y las aristas
son transformaciones ejecutables (verificadas con MathArtifacts).
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum

from .engine import MathEngine, MathResult
from .artifacts import MathArtifact, VerificationLevel

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Tipos de relaciones entre fórmulas."""
    DERIVES_FROM = "derives_from"           # A se deduce de B
    EQUIVALENT = "equivalent"               # A ≡ B (simbólicamente)
    SPECIAL_CASE = "special_case"           # A es caso particular de B
    GENERALIZES = "generalizes"             # A generaliza B
    APPROXIMATES = "approximates"           # A ≈ B (numérica/asintótica)
    DEFINES = "defines"                     # A define B
    COMPONENT_OF = "component_of"           # A es parte de B
    CONTRADICTS = "contradicts"             # A contradice B


@dataclass
class FormulaFingerprint:
    """
    Fingerprint invariante de una fórmula.

    El fingerprint es independiente del nombre de las variables:
    sin(x)^2 + cos(x)^2 y sin(y)^2 + cos(y)^2 dan el mismo fingerprint.

    Se basa en la estructura del árbol de expresión:
    - Tipo y cantidad de operaciones (Add, Mul, Pow, sin, cos, etc.)
    - Profundidad del árbol
    - Número de variables libres
    - Grado polinomial (si aplica)
    """
    ops_signature: str = ""       # Signature de operaciones ordenadas
    n_free_vars: int = 0          # Número de variables libres
    depth: int = 0                # Profundidad del árbol
    n_operations: int = 0         # Número total de operaciones
    has_trig: bool = False        # Contiene funciones trigonométricas
    has_exp: bool = False         # Contiene exponenciales
    has_matrix: bool = False      # Contiene matrices
    polynomial_degree: int = -1   # Grado (-1 si no es polinomio)
    hash: str = ""                # Hash final del fingerprint

    def compute_hash(self) -> str:
        content = f"{self.ops_signature}|{self.n_free_vars}|{self.depth}|{self.has_trig}|{self.has_exp}"
        self.hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        return self.hash


@dataclass
class FormulaNode:
    """Nodo del Knowledge Graph — una fórmula con metadata."""
    id: str
    expression: str                # Expresión SymPy como string
    latex: str = ""                # Representación LaTeX
    fingerprint: Optional[FormulaFingerprint] = None
    source_chunks: List[str] = field(default_factory=list)
    description: str = ""          # Descripción en lenguaje natural
    domain: str = ""               # Dominio (quantum, algebra, calculus, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FormulaEdge:
    """Arista ejecutable del Knowledge Graph."""
    source_id: str
    target_id: str
    relation: RelationType
    transform_code: str = ""       # Código SymPy de la transformación
    verified: bool = False
    artifact: Optional[MathArtifact] = None
    description: str = ""
    source_chunks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "relation": self.relation.value,
            "verified": self.verified,
            "description": self.description,
        }


@dataclass
class RewriteRule:
    """Regla de reescritura del e-graph, extraída de fuentes."""
    lhs: str                       # Patrón izquierdo (SymPy)
    rhs: str                       # Patrón derecho (SymPy)
    name: str = ""                 # Nombre descriptivo
    source_chunk: str = ""         # Chunk de donde se extrajo
    bidirectional: bool = True     # Si se puede aplicar en ambas direcciones
    conditions: List[str] = field(default_factory=list)
    verified: bool = False


class FormulaFingerprintEngine:
    """
    Genera fingerprints invariantes para expresiones matemáticas.

    El fingerprint captura la estructura de la expresión sin depender
    de los nombres de las variables, permitiendo buscar fórmulas
    por equivalencia estructural.
    """

    def __init__(self, math_engine: Optional[MathEngine] = None):
        self.engine = math_engine or MathEngine()

    def fingerprint(self, expression: str) -> FormulaFingerprint:
        """Genera fingerprint para una expresión."""
        code = f'''
import sympy as sp
import json

try:
    expr = sp.sympify("{expression}")
    free_vars = list(expr.free_symbols)

    # Usar count_ops para signature de operaciones (sandbox-safe)
    ops_dict = sp.count_ops(expr, visual=True)
    ops_str = str(sp.count_ops(expr, visual=False))

    # Usar repr de la expresión para crear signature estructural
    expr_str = sp.srepr(expr)
    # Extraer nombres de funciones/operaciones del srepr
    import re as _re
    op_names = sorted(_re.findall(r"([A-Z][a-zA-Z]+)\\(", expr_str))
    ops_sig = "|".join(f"{{n}}:{{op_names.count(n)}}" for n in sorted(set(op_names)))

    # Profundidad del árbol
    def tree_depth(e, d=0):
        if e.args:
            return max(tree_depth(a, d+1) for a in e.args)
        return d
    depth = tree_depth(expr)

    # Detección de tipos via isinstance (seguro)
    has_trig = any(isinstance(a, (sp.sin, sp.cos, sp.tan))
                   for a in sp.preorder_traversal(expr))
    has_exp = any(isinstance(a, sp.exp)
                  for a in sp.preorder_traversal(expr))

    # Grado polinomial
    try:
        if free_vars:
            degree = sp.degree(sp.Poly(expr, free_vars[0]))
        else:
            degree = 0
    except Exception:
        degree = -1

    n_ops = sp.count_ops(expr, visual=False)

    result = {{
        "ops_signature": ops_sig,
        "n_free_vars": len(free_vars),
        "depth": depth,
        "n_operations": int(n_ops),
        "has_trig": has_trig,
        "has_exp": has_exp,
        "polynomial_degree": degree,
    }}
except Exception as e:
    result = {{"error": str(e)}}

print("__MATH_RESULT__")
print(json.dumps(result, default=str))
print("__MATH_RESULT_END__")
'''
        math_result = self.engine._execute_math_code(code, "fingerprint", expression)
        data = self.engine._extract_math_result(math_result.stdout) if math_result.success else None

        if data and "error" not in data:
            fp = FormulaFingerprint(
                ops_signature=data.get("ops_signature", ""),
                n_free_vars=data.get("n_free_vars", 0),
                depth=data.get("depth", 0),
                n_operations=data.get("n_operations", 0),
                has_trig=data.get("has_trig", False),
                has_exp=data.get("has_exp", False),
                polynomial_degree=data.get("polynomial_degree", -1),
            )
            fp.compute_hash()
            return fp

        return FormulaFingerprint()

    def are_equivalent(self, expr1: str, expr2: str) -> Tuple[bool, str]:
        """
        Comprueba si dos expresiones son simbólicamente equivalentes.

        Returns:
            (equivalent: bool, method: str)
        """
        code = f'''
import sympy as sp
import json

try:
    e1 = sp.sympify("{expr1}")
    e2 = sp.sympify("{expr2}")
    diff = e1 - e2

    method = "none"
    equivalent = False

    for name, transform in [("expand", sp.expand), ("trigsimp", sp.trigsimp),
                             ("cancel", sp.cancel), ("factor", sp.factor)]:
        try:
            if transform(diff) == 0:
                equivalent = True
                method = name
                break
        except Exception:
            continue

    if not equivalent:
        try:
            if diff.equals(0):
                equivalent = True
                method = "equals"
        except Exception:
            pass

    result = {{"equivalent": equivalent, "method": method}}
except Exception as e:
    result = {{"equivalent": False, "method": "error", "error": str(e)}}

print("__MATH_RESULT__")
print(json.dumps(result, default=str))
print("__MATH_RESULT_END__")
'''
        math_result = self.engine._execute_math_code(code, "equivalence_check", f"{expr1[:30]} == {expr2[:30]}")
        data = self.engine._extract_math_result(math_result.stdout) if math_result.success else None

        if data:
            return data.get("equivalent", False), data.get("method", "error")
        return False, "error"


class FormulaGraph:
    """
    Knowledge Graph Computacional de fórmulas.

    Almacena fórmulas como nodos con fingerprints y aristas ejecutables
    que representan transformaciones verificadas entre ellas.
    """

    def __init__(self, math_engine: Optional[MathEngine] = None):
        self.engine = math_engine or MathEngine()
        self.fp_engine = FormulaFingerprintEngine(self.engine)
        self.nodes: Dict[str, FormulaNode] = {}
        self.edges: List[FormulaEdge] = []
        self.rewrite_rules: List[RewriteRule] = []
        self._fingerprint_index: Dict[str, List[str]] = {}  # hash -> node_ids
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"formula_{self._counter:04d}"

    def add_formula(
        self,
        expression: str,
        latex: str = "",
        source_chunks: Optional[List[str]] = None,
        description: str = "",
        domain: str = "",
    ) -> str:
        """
        Añade una fórmula al grafo.

        Calcula su fingerprint y la indexa para búsqueda por equivalencia.
        Detecta duplicados por fingerprint.

        Returns:
            ID del nodo (nuevo o existente si es duplicado)
        """
        fp = self.fp_engine.fingerprint(expression)

        # Buscar duplicados por fingerprint
        if fp.hash and fp.hash in self._fingerprint_index:
            for existing_id in self._fingerprint_index[fp.hash]:
                existing = self.nodes[existing_id]
                equiv, method = self.fp_engine.are_equivalent(
                    expression, existing.expression
                )
                if equiv:
                    logger.info(f"Fórmula equivalente encontrada: {existing_id} (método: {method})")
                    # Actualizar sources si aportan información nueva
                    if source_chunks:
                        existing.source_chunks.extend(source_chunks)
                    return existing_id

        node_id = self._next_id()
        node = FormulaNode(
            id=node_id,
            expression=expression,
            latex=latex,
            fingerprint=fp,
            source_chunks=source_chunks or [],
            description=description,
            domain=domain,
        )
        self.nodes[node_id] = node

        # Indexar por fingerprint
        if fp.hash:
            self._fingerprint_index.setdefault(fp.hash, []).append(node_id)

        return node_id

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation: RelationType,
        transform_code: str = "",
        description: str = "",
        source_chunks: Optional[List[str]] = None,
        verify: bool = True,
    ) -> FormulaEdge:
        """
        Añade una relación entre dos fórmulas.

        Si verify=True, ejecuta el transform_code para verificar
        que la transformación es correcta.
        """
        edge = FormulaEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            transform_code=transform_code,
            description=description,
            source_chunks=source_chunks or [],
        )

        if verify and transform_code and source_id in self.nodes and target_id in self.nodes:
            result = self.engine.execute_raw(transform_code)
            edge.verified = result.success
            if result.success:
                edge.artifact = MathArtifact(
                    engine="sympy",
                    operation="relation_verification",
                    code=transform_code,
                    result=result.stdout[:200],
                    verification_level=VerificationLevel.SYMBOLIC,
                    verification_passed=True,
                )

        self.edges.append(edge)
        return edge

    def add_rewrite_rule(
        self,
        lhs: str,
        rhs: str,
        name: str = "",
        source_chunk: str = "",
        bidirectional: bool = True,
        verify: bool = True,
    ) -> RewriteRule:
        """
        Añade una regla de reescritura al e-graph.

        Las reglas se extraen de las fuentes (identidades, teoremas)
        y se verifican computacionalmente antes de añadirlas.
        """
        rule = RewriteRule(
            lhs=lhs,
            rhs=rhs,
            name=name,
            source_chunk=source_chunk,
            bidirectional=bidirectional,
        )

        if verify:
            equiv, method = self.fp_engine.are_equivalent(lhs, rhs)
            rule.verified = equiv

        self.rewrite_rules.append(rule)
        return rule

    def find_by_fingerprint(self, expression: str) -> List[FormulaNode]:
        """
        Busca fórmulas con el mismo fingerprint.

        Primero busca por hash exacto, luego verifica equivalencia
        simbólica para eliminar falsos positivos.
        """
        fp = self.fp_engine.fingerprint(expression)

        if not fp.hash or fp.hash not in self._fingerprint_index:
            return []

        candidates = []
        for node_id in self._fingerprint_index[fp.hash]:
            node = self.nodes[node_id]
            equiv, _ = self.fp_engine.are_equivalent(expression, node.expression)
            if equiv:
                candidates.append(node)

        return candidates

    def find_by_structure(
        self,
        n_vars: Optional[int] = None,
        has_trig: Optional[bool] = None,
        has_exp: Optional[bool] = None,
        domain: Optional[str] = None,
    ) -> List[FormulaNode]:
        """Busca fórmulas por propiedades estructurales."""
        results = []

        for node in self.nodes.values():
            fp = node.fingerprint
            if not fp:
                continue

            if n_vars is not None and fp.n_free_vars != n_vars:
                continue
            if has_trig is not None and fp.has_trig != has_trig:
                continue
            if has_exp is not None and fp.has_exp != has_exp:
                continue
            if domain is not None and node.domain != domain:
                continue

            results.append(node)

        return results

    def find_derivation_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> Optional[List[FormulaEdge]]:
        """
        Busca un camino de derivación entre dos fórmulas.

        Usa BFS sobre las aristas del grafo para encontrar
        la secuencia más corta de transformaciones.
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        # BFS
        adj: Dict[str, List[FormulaEdge]] = {}
        for edge in self.edges:
            adj.setdefault(edge.source_id, []).append(edge)
            # Para relaciones bidireccionales
            if edge.relation in (RelationType.EQUIVALENT,):
                reverse = FormulaEdge(
                    source_id=edge.target_id,
                    target_id=edge.source_id,
                    relation=edge.relation,
                    verified=edge.verified,
                    description=f"reverse: {edge.description}",
                )
                adj.setdefault(edge.target_id, []).append(reverse)

        from collections import deque
        queue = deque([(source_id, [])])
        visited = {source_id}

        while queue:
            current, path = queue.popleft()

            if current == target_id:
                return path

            if len(path) >= max_depth:
                continue

            for edge in adj.get(current, []):
                if edge.target_id not in visited:
                    visited.add(edge.target_id)
                    queue.append((edge.target_id, path + [edge]))

        return None

    def apply_rewrite_rules(self, expression: str, max_steps: int = 10) -> List[Tuple[str, RewriteRule]]:
        """
        Aplica reglas de reescritura a una expresión.

        Returns:
            Lista de (nueva_expresión, regla_aplicada) para cada paso
        """
        steps = []
        current = expression
        seen = {expression}

        for _ in range(max_steps):
            applied = False
            for rule in self.rewrite_rules:
                if not rule.verified:
                    continue

                # Intentar aplicar la regla (LHS → RHS)
                equiv_lhs, _ = self.fp_engine.are_equivalent(current, rule.lhs)
                if equiv_lhs and rule.rhs not in seen:
                    steps.append((rule.rhs, rule))
                    current = rule.rhs
                    seen.add(current)
                    applied = True
                    break

                # Intentar en dirección inversa (RHS → LHS)
                if rule.bidirectional:
                    equiv_rhs, _ = self.fp_engine.are_equivalent(current, rule.rhs)
                    if equiv_rhs and rule.lhs not in seen:
                        steps.append((rule.lhs, rule))
                        current = rule.lhs
                        seen.add(current)
                        applied = True
                        break

            if not applied:
                break

        return steps

    def to_dict(self) -> Dict[str, Any]:
        """Serializa el grafo completo."""
        return {
            "nodes": {
                nid: {
                    "expression": n.expression,
                    "latex": n.latex,
                    "fingerprint_hash": n.fingerprint.hash if n.fingerprint else "",
                    "source_chunks": n.source_chunks,
                    "description": n.description,
                    "domain": n.domain,
                }
                for nid, n in self.nodes.items()
            },
            "edges": [e.to_dict() for e in self.edges],
            "rewrite_rules": [
                {
                    "lhs": r.lhs,
                    "rhs": r.rhs,
                    "name": r.name,
                    "verified": r.verified,
                }
                for r in self.rewrite_rules
            ],
            "stats": {
                "n_nodes": len(self.nodes),
                "n_edges": len(self.edges),
                "n_rules": len(self.rewrite_rules),
                "n_verified_edges": sum(1 for e in self.edges if e.verified),
                "n_verified_rules": sum(1 for r in self.rewrite_rules if r.verified),
            },
        }

    def summary(self) -> str:
        """Resumen del grafo."""
        stats = self.to_dict()["stats"]
        return (
            f"FormulaGraph: {stats['n_nodes']} formulas, "
            f"{stats['n_edges']} edges ({stats['n_verified_edges']} verified), "
            f"{stats['n_rules']} rewrite rules ({stats['n_verified_rules']} verified)"
        )
