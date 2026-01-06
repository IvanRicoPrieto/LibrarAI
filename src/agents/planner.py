"""
Query Planner - Planifica ejecución de consultas complejas.

Para consultas multi-hop o que requieren varios pasos:
1. Descompone la consulta
2. Planifica el orden de ejecución
3. Gestiona dependencias entre sub-consultas
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import re
import os

logger = logging.getLogger(__name__)


class StepType(Enum):
    """Tipos de pasos en un plan."""
    RETRIEVE = "retrieve"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    COMPARE = "compare"
    SYNTHESIZE = "synthesize"


@dataclass
class PlanStep:
    """Paso individual en el plan de ejecución."""
    step_id: int
    step_type: StepType
    query: str
    depends_on: List[int] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    
    def to_dict(self) -> Dict:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "query": self.query,
            "depends_on": self.depends_on,
            "params": self.params
        }


@dataclass
class ExecutionPlan:
    """Plan de ejecución completo."""
    original_query: str
    steps: List[PlanStep]
    estimated_retrievals: int
    estimated_tokens: int
    reasoning: str
    
    def to_dict(self) -> Dict:
        return {
            "original_query": self.original_query,
            "steps": [s.to_dict() for s in self.steps],
            "estimated_retrievals": self.estimated_retrievals,
            "estimated_tokens": self.estimated_tokens,
            "reasoning": self.reasoning
        }
    
    def get_execution_order(self) -> List[List[PlanStep]]:
        """
        Obtiene pasos agrupados por nivel de dependencia.
        
        Returns:
            Lista de listas, donde cada lista puede ejecutarse en paralelo
        """
        remaining = {s.step_id: s for s in self.steps}
        completed = set()
        levels = []
        
        while remaining:
            # Encontrar pasos ejecutables (dependencias completadas)
            level = []
            for step_id, step in list(remaining.items()):
                if all(dep in completed for dep in step.depends_on):
                    level.append(step)
            
            if not level:
                # Ciclo detectado o error
                logger.warning("No executable steps found, breaking")
                break
            
            levels.append(level)
            
            for step in level:
                completed.add(step.step_id)
                del remaining[step.step_id]
        
        return levels


class QueryPlanner:
    """
    Planificador de consultas complejas.
    
    Descompone consultas en pasos ejecutables y
    determina el orden óptimo de ejecución.
    """
    
    # Patrones de descomposición
    DECOMPOSITION_PATTERNS = {
        "sequential": [
            r"(?:primero|first).*?(?:luego|después|then|after)",
            r"(?:antes de|before).*?(?:después|after)"
        ],
        "comparative": [
            r"(?:diferencia|difference)\s+(?:entre|between)",
            r"(?:comparar?|compare)\s+(\w+)\s+(?:y|and|con|with)\s+(\w+)"
        ],
        "causal": [
            r"(?:por qué|why).*?(?:porque|because|ya que)",
            r"(?:causa|cause).*?(?:efecto|effect)"
        ],
        "conditional": [
            r"(?:si|if).*?(?:entonces|then|qué)",
            r"(?:cuando|when).*?(?:qué|what)"
        ]
    }
    
    def __init__(
        self,
        use_llm_planner: bool = False,
        llm_model: str = "gpt-4o-mini",
        max_steps: int = 5
    ):
        """
        Args:
            use_llm_planner: Si usar LLM para planificación
            llm_model: Modelo para planificación
            max_steps: Máximo número de pasos
        """
        self.use_llm_planner = use_llm_planner
        self.llm_model = llm_model
        self.max_steps = max_steps
        self._llm_client = None
    
    def plan(self, query: str) -> ExecutionPlan:
        """
        Crea plan de ejecución para una consulta.
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Plan de ejecución
        """
        if self.use_llm_planner:
            return self._plan_with_llm(query)
        else:
            return self._plan_with_heuristics(query)
    
    def _plan_with_heuristics(self, query: str) -> ExecutionPlan:
        """Planificación basada en heurísticas."""
        query_lower = query.lower()
        steps = []
        step_id = 0
        
        # Detectar tipo de descomposición
        is_comparative = any(
            re.search(p, query_lower) 
            for p in self.DECOMPOSITION_PATTERNS["comparative"]
        )
        
        is_sequential = any(
            re.search(p, query_lower) 
            for p in self.DECOMPOSITION_PATTERNS["sequential"]
        )
        
        if is_comparative:
            # Extraer elementos a comparar
            entities = self._extract_comparison_entities(query)
            
            if len(entities) >= 2:
                # Paso 1: Retrieval para primera entidad
                steps.append(PlanStep(
                    step_id=step_id,
                    step_type=StepType.RETRIEVE,
                    query=f"información sobre {entities[0]}",
                    params={"entity": entities[0]}
                ))
                step_id += 1
                
                # Paso 2: Retrieval para segunda entidad
                steps.append(PlanStep(
                    step_id=step_id,
                    step_type=StepType.RETRIEVE,
                    query=f"información sobre {entities[1]}",
                    params={"entity": entities[1]}
                ))
                step_id += 1
                
                # Paso 3: Comparar resultados
                steps.append(PlanStep(
                    step_id=step_id,
                    step_type=StepType.COMPARE,
                    query=query,
                    depends_on=[0, 1],
                    params={"entities": entities}
                ))
                step_id += 1
            
        elif is_sequential:
            # Dividir en pasos secuenciales
            sub_queries = self._split_sequential_query(query)
            
            prev_id = None
            for sub_q in sub_queries[:self.max_steps]:
                step = PlanStep(
                    step_id=step_id,
                    step_type=StepType.RETRIEVE,
                    query=sub_q,
                    depends_on=[prev_id] if prev_id is not None else []
                )
                steps.append(step)
                prev_id = step_id
                step_id += 1
            
            # Paso final: síntesis
            steps.append(PlanStep(
                step_id=step_id,
                step_type=StepType.SYNTHESIZE,
                query=query,
                depends_on=list(range(step_id))
            ))
        
        else:
            # Query simple: un solo paso de retrieval
            steps.append(PlanStep(
                step_id=0,
                step_type=StepType.RETRIEVE,
                query=query
            ))
        
        # Estimar recursos
        estimated_retrievals = sum(
            1 for s in steps if s.step_type == StepType.RETRIEVE
        )
        estimated_tokens = estimated_retrievals * 2000  # ~2000 tokens por retrieval
        
        return ExecutionPlan(
            original_query=query,
            steps=steps,
            estimated_retrievals=estimated_retrievals,
            estimated_tokens=estimated_tokens,
            reasoning=f"Plan con {len(steps)} pasos, {estimated_retrievals} retrievals"
        )
    
    def _extract_comparison_entities(self, query: str) -> List[str]:
        """Extrae entidades a comparar de una query."""
        entities = []
        
        # Patrón: "X vs Y", "X y Y", "entre X y Y"
        patterns = [
            r"(?:entre|between)\s+(\w+)\s+(?:y|and)\s+(\w+)",
            r"(\w+)\s+(?:vs\.?|versus)\s+(\w+)",
            r"(?:comparar?|compare)\s+(\w+)\s+(?:con|with|y|and)\s+(\w+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entities = list(match.groups())
                break
        
        return entities
    
    def _split_sequential_query(self, query: str) -> List[str]:
        """Divide query secuencial en partes."""
        # Separadores secuenciales
        separators = [
            r'\s+(?:y\s+)?(?:luego|después|then|after)\s+',
            r'\s+(?:y\s+)?(?:finalmente|finally)\s+',
            r'\s*[;]\s*'
        ]
        
        parts = [query]
        for sep in separators:
            new_parts = []
            for part in parts:
                new_parts.extend(re.split(sep, part, flags=re.IGNORECASE))
            parts = new_parts
        
        return [p.strip() for p in parts if p.strip()]
    
    def _plan_with_llm(self, query: str) -> ExecutionPlan:
        """Planificación usando LLM."""
        self._init_llm()
        
        prompt = f"""Descompón esta consulta en pasos de ejecución.

CONSULTA: {query}

TIPOS DE PASOS:
- RETRIEVE: Buscar información
- FILTER: Filtrar resultados
- AGGREGATE: Combinar información
- COMPARE: Comparar elementos
- SYNTHESIZE: Sintetizar respuesta final

Responde en JSON:
{{
    "steps": [
        {{
            "step_id": 0,
            "step_type": "RETRIEVE",
            "query": "sub-consulta",
            "depends_on": [],
            "params": {{}}
        }}
    ],
    "reasoning": "Explicación del plan"
}}

Máximo {self.max_steps} pasos."""
        
        try:
            response = self._llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            steps = []
            for s in result.get("steps", [])[:self.max_steps]:
                steps.append(PlanStep(
                    step_id=s.get("step_id", len(steps)),
                    step_type=StepType(s.get("step_type", "retrieve").lower()),
                    query=s.get("query", query),
                    depends_on=s.get("depends_on", []),
                    params=s.get("params", {})
                ))
            
            if not steps:
                steps = [PlanStep(step_id=0, step_type=StepType.RETRIEVE, query=query)]
            
            estimated_retrievals = sum(
                1 for s in steps if s.step_type == StepType.RETRIEVE
            )
            
            return ExecutionPlan(
                original_query=query,
                steps=steps,
                estimated_retrievals=estimated_retrievals,
                estimated_tokens=estimated_retrievals * 2000,
                reasoning=result.get("reasoning", "")
            )
            
        except Exception as e:
            logger.warning(f"LLM planning failed: {e}, using heuristics")
            return self._plan_with_heuristics(query)
    
    def _init_llm(self):
        """Inicializa cliente LLM."""
        if self._llm_client is not None:
            return
        
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY no configurada")
        
        self._llm_client = OpenAI(api_key=api_key)
