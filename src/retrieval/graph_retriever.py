"""
Graph Retriever - Búsqueda en grafo de conocimiento.

Permite:
- Navegación por relaciones entre conceptos
- Expansión de contexto vía entidades relacionadas
- Path-finding entre conceptos
"""

from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import logging
import pickle
import re

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Entidad extraída del texto."""
    name: str
    entity_type: str
    chunk_ids: List[str] = field(default_factory=list)
    mentions: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    """Relación entre entidades."""
    source: str
    target: str
    relation_type: str
    weight: float = 1.0
    chunk_id: Optional[str] = None


@dataclass
class GraphSearchResult:
    """Resultado de búsqueda en grafo."""
    entity: str
    entity_type: str
    related_entities: List[Tuple[str, str, str]]  # (entity, relation, type)
    chunk_ids: List[str]
    score: float
    path: Optional[List[str]] = None


class GraphRetriever:
    """
    Retriever basado en grafo de conocimiento.
    
    Usa NetworkX para:
    - Almacenar entidades como nodos
    - Almacenar relaciones como aristas
    - Buscar caminos y vecindarios
    """
    
    # Patrones de extracción de entidades
    ENTITY_PATTERNS = {
        "Algoritmo": [
            r"algoritmo\s+de?\s*(\w+(?:\s+\w+)?)",
            r"(Shor|Grover|Deutsch|Simon|BB84|E91|B92|QKD)",
            r"(\w+(?:-\w+)?)\s+algorithm"
        ],
        "Protocolo": [
            r"protocolo\s+(\w+(?:\s+\w+)?)",
            r"(BB84|E91|B92|SARG04|COW|DPS)",
            r"protocol\s+(\w+)"
        ],
        "Concepto": [
            r"(entrelazamiento|superposición|decoherencia|teleportación)",
            r"(qubit|qutrit|qudit)",
            r"(entanglement|superposition|decoherence)"
        ],
        "Gate": [
            r"puerta\s+(\w+)",
            r"(Hadamard|CNOT|Pauli|Toffoli|SWAP)",
            r"(H|X|Y|Z|T|S)\s+gate"
        ],
        "Autor": [
            r"(?:Bennett|Brassard|Ekert|Shor|Grover|Deutsch|Feynman|Nielsen|Chuang)",
        ]
    }
    
    def __init__(
        self,
        indices_dir: Path,
        ontology_path: Optional[Path] = None
    ):
        """
        Args:
            indices_dir: Directorio con los índices
            ontology_path: Ruta al archivo de ontología YAML
        """
        self.indices_dir = Path(indices_dir)
        self.ontology_path = ontology_path
        
        self._graph = None
        self._chunks_store = None
        self._entity_index: Dict[str, Entity] = {}
    
    def _init_graph(self):
        """Inicializa o carga el grafo."""
        if self._graph is not None:
            return
        
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx no instalado. Ejecuta: pip install networkx"
            )
        
        import pickle
        graph_path = self.indices_dir / "knowledge_graph.gpickle"
        
        if graph_path.exists():
            with open(graph_path, 'rb') as f:
                self._graph = pickle.load(f)
            logger.info(f"Grafo cargado: {self._graph.number_of_nodes()} nodos, "
                       f"{self._graph.number_of_edges()} aristas")
        else:
            self._graph = nx.DiGraph()
            logger.info("Grafo vacío inicializado")
        
        # Construir índice de entidades desde el grafo
        for node, attrs in self._graph.nodes(data=True):
            self._entity_index[node.lower()] = Entity(
                name=node,
                entity_type=attrs.get("type", "Unknown"),
                chunk_ids=attrs.get("chunk_ids", []),
                mentions=attrs.get("mentions", 1)
            )
    
    def _load_chunks_store(self):
        """Carga almacén de chunks."""
        if self._chunks_store is not None:
            return
        
        chunks_path = self.indices_dir / "chunks.pkl"
        if chunks_path.exists():
            with open(chunks_path, 'rb') as f:
                self._chunks_store = pickle.load(f)
    
    def build_graph_from_chunks(self, use_llm: bool = False, sample_rate: float = 0.1):
        """
        Construye grafo extrayendo entidades y relaciones de los chunks.

        Args:
            use_llm: Si usar LLM para extracción (más preciso, más costoso).
                     Cuando True, usa LLMGraphExtractor para extracción alineada a ontología.
            sample_rate: Si use_llm=True, qué proporción de chunks procesar con LLM
                         (el resto usa regex). Valor 0-1.
        """
        import networkx as nx
        import random

        self._load_chunks_store()
        self._graph = nx.DiGraph()

        if not self._chunks_store:
            logger.warning("No hay chunks para construir grafo")
            return

        total_chunks = len(self._chunks_store)
        llm_chunks = set()

        if use_llm and sample_rate > 0:
            # Seleccionar muestra de chunks para procesar con LLM
            n_llm = max(1, int(total_chunks * sample_rate))
            llm_chunks = set(random.sample(list(self._chunks_store.keys()), n_llm))
            logger.info(f"GraphRAG con LLM: {n_llm}/{total_chunks} chunks ({sample_rate*100:.0f}%)")

        # Inicializar LLMGraphExtractor si se usa LLM
        llm_extractor = None
        if use_llm and llm_chunks:
            try:
                from ..ingestion.graph_builder import LLMGraphExtractor
                ontology_path = self.ontology_path
                if ontology_path is None:
                    # Intentar encontrar ontología en config/
                    candidate = self.indices_dir.parent / "config" / "ontology.yaml"
                    if candidate.exists():
                        ontology_path = candidate
                llm_extractor = LLMGraphExtractor(ontology_path=ontology_path)
                logger.info("LLMGraphExtractor inicializado para extracción de grafo")
            except Exception as e:
                logger.warning(f"No se pudo inicializar LLMGraphExtractor: {e}. Fallback a método anterior.")

        # Procesar chunks LLM en batch con LLMGraphExtractor
        if llm_extractor and llm_chunks:
            llm_chunk_list = [
                (cid, self._chunks_store[cid].content)
                for cid in llm_chunks
                if cid in self._chunks_store
            ]
            # Procesar en batches
            batch_size = 5
            for i in range(0, len(llm_chunk_list), batch_size):
                batch = llm_chunk_list[i:i + batch_size]
                try:
                    extraction_results = llm_extractor.extract_batch(batch)
                    for result in extraction_results:
                        # Añadir entidades
                        for ent in result.entities:
                            entity = Entity(
                                name=ent["name"],
                                entity_type=ent["type"]
                            )
                            self._add_entity_to_graph(entity, result.chunk_id)
                        # Añadir triples como relaciones tipadas
                        for triple in result.triples:
                            # Asegurar que source y target existen como nodos
                            src_entity = Entity(name=triple.source_name, entity_type=triple.source_type)
                            tgt_entity = Entity(name=triple.target_name, entity_type=triple.target_type)
                            self._add_entity_to_graph(src_entity, result.chunk_id)
                            self._add_entity_to_graph(tgt_entity, result.chunk_id)
                            self._add_relation(
                                triple.source_name, triple.target_name,
                                triple.relation_type,
                                weight=triple.confidence,
                                chunk_id=result.chunk_id
                            )
                except Exception as e:
                    logger.warning(f"Error en batch LLM de grafo: {e}")

        # Extraer entidades de chunks restantes (regex)
        for chunk_id, chunk in self._chunks_store.items():
            if chunk_id in llm_chunks and llm_extractor:
                continue  # Ya procesado con LLM

            entities = self._extract_entities_regex(chunk.content)

            for entity in entities:
                self._add_entity_to_graph(entity, chunk_id)

            # Crear relaciones CO_OCCURS entre entidades co-ocurrentes
            if len(entities) > 1:
                for i, e1 in enumerate(entities):
                    for e2 in entities[i+1:]:
                        self._add_relation(
                            e1.name, e2.name,
                            "CO_OCCURS",
                            chunk_id=chunk_id
                        )

        # Guardar grafo
        import pickle
        graph_path = self.indices_dir / "knowledge_graph.gpickle"
        with open(graph_path, 'wb') as f:
            pickle.dump(self._graph, f)

        logger.info(f"Grafo construido: {self._graph.number_of_nodes()} nodos, "
                   f"{self._graph.number_of_edges()} aristas")
    
    def _extract_entities(self, text: str, use_llm: bool = False) -> List[Entity]:
        """
        Extrae entidades del texto.
        
        Args:
            text: Texto a analizar
            use_llm: Si usar LLM para extracción (más preciso, más costoso)
        
        Returns:
            Lista de entidades encontradas
        """
        if use_llm:
            return self._extract_entities_llm(text)
        return self._extract_entities_regex(text)
    
    def _extract_entities_regex(self, text: str) -> List[Entity]:
        """Extrae entidades usando patrones regex (rápido, gratuito)."""
        entities = []
        seen = set()
        
        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Obtener el grupo capturado o el match completo
                    name = match.group(1) if match.groups() else match.group(0)
                    name = name.strip()
                    
                    if name.lower() not in seen and len(name) > 1:
                        seen.add(name.lower())
                        entities.append(Entity(
                            name=name,
                            entity_type=entity_type
                        ))
        
        return entities
    
    def _extract_entities_llm(self, text: str) -> List[Entity]:
        """Extrae entidades usando LLM (más preciso para textos complejos)."""
        try:
            import json
            from src.llm_provider import complete as llm_complete

            # Truncar texto si es muy largo
            max_chars = 3000
            if len(text) > max_chars:
                text = text[:max_chars] + "..."

            system_prompt = """Extrae entidades de computación/física cuántica del texto.
Devuelve JSON con formato:
{
  "entities": [
    {"name": "nombre exacto", "type": "tipo", "relations": ["relación con otra entidad"]}
  ]
}

Tipos válidos: Algoritmo, Protocolo, Concepto, Gate, Autor, Teorema, Ecuación
Relaciones válidas: MEJORA, DEPENDE_DE, IMPLEMENTA, EQUIVALENTE_A, DEFINE, DEMUESTRA, USA

Solo incluye entidades claramente identificables del dominio cuántico."""

            response = llm_complete(
                prompt=f"Texto:\n{text}",
                system=system_prompt,
                temperature=0,
                max_tokens=500,
                json_mode=True,
            )

            result = json.loads(response.content)
            entities = []

            for e in result.get("entities", []):
                entities.append(Entity(
                    name=e.get("name", ""),
                    entity_type=e.get("type", "Concepto"),
                    metadata={"relations": e.get("relations", [])}
                ))

            return entities

        except Exception as e:
            logger.warning(f"Error en extracción LLM: {e}. Fallback a regex.")
            return self._extract_entities_regex(text)
    
    def _add_entity_to_graph(self, entity: Entity, chunk_id: str):
        """Añade o actualiza entidad en el grafo."""
        node_id = entity.name.lower()
        
        if self._graph.has_node(node_id):
            # Actualizar existente
            attrs = self._graph.nodes[node_id]
            attrs["mentions"] = attrs.get("mentions", 0) + 1
            chunk_ids = attrs.get("chunk_ids", [])
            if chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)
            attrs["chunk_ids"] = chunk_ids
        else:
            # Añadir nuevo
            self._graph.add_node(
                node_id,
                name=entity.name,
                type=entity.entity_type,
                mentions=1,
                chunk_ids=[chunk_id]
            )
        
        # Actualizar índice
        self._entity_index[node_id] = Entity(
            name=entity.name,
            entity_type=entity.entity_type,
            chunk_ids=self._graph.nodes[node_id].get("chunk_ids", []),
            mentions=self._graph.nodes[node_id].get("mentions", 1)
        )
    
    def _add_relation(
        self,
        source: str,
        target: str,
        relation_type: str,
        weight: float = 1.0,
        chunk_id: Optional[str] = None
    ):
        """Añade relación al grafo."""
        source_id = source.lower()
        target_id = target.lower()
        
        if self._graph.has_edge(source_id, target_id):
            # Incrementar peso
            self._graph[source_id][target_id]["weight"] += weight
        else:
            self._graph.add_edge(
                source_id,
                target_id,
                relation=relation_type,
                weight=weight,
                chunk_id=chunk_id
            )
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        depth: int = 2
    ) -> List[GraphSearchResult]:
        """
        Busca entidades relacionadas con la query.
        
        Args:
            query: Consulta en lenguaje natural
            top_k: Número de resultados
            depth: Profundidad de exploración en el grafo
            
        Returns:
            Entidades relevantes con sus relaciones
        """
        self._init_graph()
        
        # Extraer entidades de la query
        query_entities = self._extract_entities(query)
        
        if not query_entities:
            # Búsqueda por substring
            query_lower = query.lower()
            for entity_name in self._entity_index:
                if query_lower in entity_name or entity_name in query_lower:
                    entity = self._entity_index[entity_name]
                    query_entities.append(entity)
        
        results = []
        seen_entities = set()
        
        for entity in query_entities:
            entity_id = entity.name.lower()
            
            if entity_id not in self._graph:
                continue
            
            if entity_id in seen_entities:
                continue
            seen_entities.add(entity_id)
            
            # Obtener vecinos hasta profundidad especificada
            related = self._get_neighbors(entity_id, depth)
            
            # Obtener chunks asociados
            chunk_ids = self._graph.nodes[entity_id].get("chunk_ids", [])
            
            result = GraphSearchResult(
                entity=entity.name,
                entity_type=entity.entity_type,
                related_entities=related,
                chunk_ids=chunk_ids,
                score=self._graph.nodes[entity_id].get("mentions", 1)
            )
            results.append(result)
        
        # Ordenar por score (mentions)
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _get_neighbors(
        self,
        entity_id: str,
        depth: int
    ) -> List[Tuple[str, str, str]]:
        """Obtiene vecinos hasta cierta profundidad."""
        related = []
        visited = {entity_id}
        current_level = [entity_id]
        
        for _ in range(depth):
            next_level = []
            
            for node in current_level:
                # Sucesores (aristas salientes)
                for successor in self._graph.successors(node):
                    if successor not in visited:
                        visited.add(successor)
                        next_level.append(successor)
                        
                        edge_data = self._graph[node][successor]
                        node_data = self._graph.nodes[successor]
                        
                        related.append((
                            node_data.get("name", successor),
                            edge_data.get("relation", "RELATED"),
                            node_data.get("type", "Unknown")
                        ))
                
                # Predecesores (aristas entrantes)
                for predecessor in self._graph.predecessors(node):
                    if predecessor not in visited:
                        visited.add(predecessor)
                        next_level.append(predecessor)
                        
                        edge_data = self._graph[predecessor][node]
                        node_data = self._graph.nodes[predecessor]
                        
                        related.append((
                            node_data.get("name", predecessor),
                            edge_data.get("relation", "RELATED"),
                            node_data.get("type", "Unknown")
                        ))
            
            current_level = next_level
        
        return related
    
    def find_path(
        self,
        source: str,
        target: str
    ) -> Optional[List[str]]:
        """
        Encuentra camino más corto entre dos entidades.
        
        Args:
            source: Entidad origen
            target: Entidad destino
            
        Returns:
            Lista de entidades en el camino, o None
        """
        self._init_graph()
        
        import networkx as nx
        
        source_id = source.lower()
        target_id = target.lower()
        
        if source_id not in self._graph or target_id not in self._graph:
            return None
        
        try:
            path = nx.shortest_path(
                self._graph.to_undirected(),
                source_id,
                target_id
            )
            
            # Convertir a nombres originales
            return [
                self._graph.nodes[n].get("name", n) 
                for n in path
            ]
        except nx.NetworkXNoPath:
            return None
    
    def get_entity_chunks(self, entity_name: str) -> List[str]:
        """Obtiene chunks donde aparece una entidad."""
        self._init_graph()
        
        entity_id = entity_name.lower()
        
        if entity_id in self._graph:
            return self._graph.nodes[entity_id].get("chunk_ids", [])
        
        return []
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del grafo."""
        self._init_graph()
        
        import networkx as nx
        
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "density": nx.density(self._graph),
            "connected_components": nx.number_weakly_connected_components(self._graph),
            "entity_types": self._count_entity_types()
        }
    
    def _count_entity_types(self) -> Dict[str, int]:
        """Cuenta entidades por tipo."""
        counts = {}
        for node, attrs in self._graph.nodes(data=True):
            entity_type = attrs.get("type", "Unknown")
            counts[entity_type] = counts.get(entity_type, 0) + 1
        return counts
