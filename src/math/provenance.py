"""
Provenance — Trazabilidad W3C PROV para cadenas de razonamiento matemático.

Implementa el modelo PROV-AGENT extendido para IA:
- Entity: Datos (chunks, ecuaciones, resultados, artifacts)
- Activity: Procesos (retrieval, computación, verificación, síntesis)
- Agent: Actores (LLM, SymPy, Wolfram, verificador)

Cada respuesta produce un grafo de provenance que permite
trazar cada afirmación hasta sus fuentes originales.
"""

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional


class EntityType(Enum):
    """Tipos de entidades en el grafo de provenance."""
    SOURCE_CHUNK = "source_chunk"        # Chunk del retrieval
    PARSED_EQUATION = "parsed_equation"  # Ecuación parseada de LaTeX
    COMPUTATION_RESULT = "computation"   # Resultado de cálculo
    MATH_ARTIFACT = "math_artifact"      # Artefacto de verificación
    LLM_RESPONSE = "llm_response"        # Respuesta del LLM
    DERIVATION_STEP = "derivation_step"  # Paso de derivación
    FINAL_RESPONSE = "final_response"    # Respuesta final


class ActivityType(Enum):
    """Tipos de actividades."""
    RETRIEVAL = "retrieval"
    LATEX_PARSING = "latex_parsing"
    COMPUTATION = "computation"
    VERIFICATION = "verification"
    LLM_REASONING = "llm_reasoning"
    SYNTHESIS = "synthesis"
    PLANNING = "planning"


class AgentType(Enum):
    """Tipos de agentes."""
    LLM = "llm"
    SYMPY = "sympy"
    NUMPY = "numpy"
    WOLFRAM = "wolfram"
    QUTIP = "qutip"
    PINT = "pint"
    VERIFIER = "verifier"
    PLANNER = "planner"


@dataclass
class ProvenanceEntity:
    """Entidad en el grafo de provenance."""
    id: str
    entity_type: EntityType
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash and self.content:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class ProvenanceActivity:
    """Actividad en el grafo de provenance."""
    id: str
    activity_type: ActivityType
    description: str = ""
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ended_at: Optional[str] = None
    used: List[str] = field(default_factory=list)        # Entity IDs consumidos
    generated: List[str] = field(default_factory=list)    # Entity IDs producidos
    agent_id: Optional[str] = None                        # Agent que ejecutó
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceAgent:
    """Agente en el grafo de provenance."""
    id: str
    agent_type: AgentType
    name: str = ""
    version: str = ""


class ProvenanceGraph:
    """
    Grafo de provenance W3C PROV para una sesión de razonamiento.

    Registra todas las entidades, actividades y agentes involucrados
    en la generación de una respuesta, permitiendo trazar cada
    afirmación hasta sus fuentes.
    """

    def __init__(self):
        self.entities: Dict[str, ProvenanceEntity] = {}
        self.activities: Dict[str, ProvenanceActivity] = {}
        self.agents: Dict[str, ProvenanceAgent] = {}
        self._derivation_links: List[Dict[str, str]] = []  # wasDerivedFrom
        self._counter = 0

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter:04d}"

    def add_entity(
        self,
        entity_type: EntityType,
        content: str = "",
        metadata: Optional[Dict] = None,
        entity_id: Optional[str] = None,
    ) -> str:
        """Registra una entidad y devuelve su ID."""
        eid = entity_id or self._next_id(entity_type.value)
        self.entities[eid] = ProvenanceEntity(
            id=eid,
            entity_type=entity_type,
            content=content,
            metadata=metadata or {},
        )
        return eid

    def add_activity(
        self,
        activity_type: ActivityType,
        description: str = "",
        used: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Registra una actividad y devuelve su ID."""
        aid = self._next_id(activity_type.value)
        self.activities[aid] = ProvenanceActivity(
            id=aid,
            activity_type=activity_type,
            description=description,
            used=used or [],
            agent_id=agent_id,
            metadata=metadata or {},
        )
        return aid

    def add_agent(
        self,
        agent_type: AgentType,
        name: str = "",
        version: str = "",
    ) -> str:
        """Registra un agente y devuelve su ID."""
        agent_id = self._next_id(agent_type.value)
        self.agents[agent_id] = ProvenanceAgent(
            id=agent_id,
            agent_type=agent_type,
            name=name,
            version=version,
        )
        return agent_id

    def record_generation(
        self,
        activity_id: str,
        entity_id: str,
    ):
        """Registra que una actividad generó una entidad."""
        if activity_id in self.activities:
            self.activities[activity_id].generated.append(entity_id)

    def record_derivation(
        self,
        derived_entity_id: str,
        source_entity_id: str,
    ):
        """Registra que una entidad se derivó de otra."""
        self._derivation_links.append({
            "derived": derived_entity_id,
            "source": source_entity_id,
        })

    def end_activity(self, activity_id: str):
        """Marca una actividad como completada."""
        if activity_id in self.activities:
            self.activities[activity_id].ended_at = datetime.now().isoformat()

    def get_lineage(self, entity_id: str) -> List[Dict[str, Any]]:
        """
        Traza la cadena de provenance completa para una entidad.

        Devuelve la lista de entidades y actividades que contribuyeron
        a la generación de esta entidad, en orden cronológico inverso.
        """
        lineage = []
        visited = set()
        queue = [entity_id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if current in self.entities:
                entity = self.entities[current]
                lineage.append({
                    "type": "entity",
                    "id": entity.id,
                    "entity_type": entity.entity_type.value,
                    "content_hash": entity.content_hash,
                    "timestamp": entity.timestamp,
                })

            # Buscar actividades que generaron esta entidad
            for act in self.activities.values():
                if current in act.generated:
                    lineage.append({
                        "type": "activity",
                        "id": act.id,
                        "activity_type": act.activity_type.value,
                        "agent": act.agent_id,
                        "description": act.description,
                    })
                    queue.extend(act.used)

            # Buscar derivaciones
            for link in self._derivation_links:
                if link["derived"] == current:
                    queue.append(link["source"])

        return lineage

    def to_dict(self) -> Dict[str, Any]:
        """Serializa el grafo completo."""
        return {
            "entities": {
                eid: {
                    "type": e.entity_type.value,
                    "content_hash": e.content_hash,
                    "metadata": e.metadata,
                    "timestamp": e.timestamp,
                }
                for eid, e in self.entities.items()
            },
            "activities": {
                aid: {
                    "type": a.activity_type.value,
                    "description": a.description,
                    "used": a.used,
                    "generated": a.generated,
                    "agent": a.agent_id,
                    "started_at": a.started_at,
                    "ended_at": a.ended_at,
                }
                for aid, a in self.activities.items()
            },
            "agents": {
                aid: {
                    "type": a.agent_type.value,
                    "name": a.name,
                    "version": a.version,
                }
                for aid, a in self.agents.items()
            },
            "derivations": self._derivation_links,
        }

    def summary(self) -> str:
        """Resumen textual del grafo de provenance."""
        return (
            f"ProvenanceGraph: {len(self.entities)} entities, "
            f"{len(self.activities)} activities, "
            f"{len(self.agents)} agents, "
            f"{len(self._derivation_links)} derivations"
        )
