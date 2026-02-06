"""
LLM Knowledge Graph Extraction — Extracción de grafo con LLM y ontología.

Reemplaza/complementa la extracción regex con extracción LLM alineada
a la ontología definida en config/ontology.yaml. Extrae entidades Y
relaciones tipadas en una sola llamada, con batch processing.

El resultado se integra con el GraphRetriever existente para crear
edges tipados en el grafo NetworkX.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExtractedTriple:
    """Tripleta entidad-relación-entidad extraída."""
    source_name: str
    source_type: str
    relation_type: str
    target_name: str
    target_type: str
    confidence: float
    chunk_id: str


@dataclass
class ExtractionResult:
    """Resultado de extracción para un chunk."""
    chunk_id: str
    entities: List[Dict[str, str]]  # [{"name": ..., "type": ...}, ...]
    triples: List[ExtractedTriple]
    raw_llm_response: str = ""


class OntologyPromptBuilder:
    """
    Construye prompts de extracción alineados con la ontología YAML.

    Lee la ontología y genera las secciones del system prompt con los
    tipos de entidad y relación válidos.
    """

    def __init__(self, ontology_path: Optional[Path] = None):
        self.ontology_path = ontology_path
        self._ontology: Optional[Dict] = None
        self._entity_types: List[str] = []
        self._relation_types: List[str] = []

        if ontology_path and Path(ontology_path).exists():
            self._load_ontology()
        else:
            self._use_defaults()

    def _load_ontology(self):
        """Carga ontología desde YAML."""
        try:
            import yaml

            with open(self.ontology_path, "r") as f:
                self._ontology = yaml.safe_load(f)

            # Extraer tipos de entidad
            entities = self._ontology.get("entities", {})
            self._entity_types = list(entities.keys())

            # Extraer de secciones anidadas (e.g. 'finanzas')
            for key, val in self._ontology.items():
                if isinstance(val, dict) and "entities" in val:
                    self._entity_types.extend(list(val["entities"].keys()))

            # Extraer tipos de relación
            relations = self._ontology.get("relations", {})
            self._relation_types = list(relations.keys())

            for key, val in self._ontology.items():
                if isinstance(val, dict) and "relations" in val:
                    self._relation_types.extend(list(val["relations"].keys()))

            logger.info(
                f"Ontología cargada: {len(self._entity_types)} tipos de entidad, "
                f"{len(self._relation_types)} tipos de relación"
            )

        except Exception as e:
            logger.warning(f"Error cargando ontología: {e}. Usando defaults.")
            self._use_defaults()

    def _use_defaults(self):
        """Tipos por defecto si no hay ontología."""
        self._entity_types = [
            "Algoritmo", "Protocolo", "Concepto", "Gate", "Autor",
            "Teorema", "Ecuación", "Hardware", "Documento",
        ]
        self._relation_types = [
            "MEJORA", "DEPENDE_DE", "USA", "DEFINE", "IMPLEMENTA",
            "DEMUESTRA", "PROPONE", "ES_CASO_DE", "EQUIVALE_A",
            "CO_OCCURS",
        ]

    def build_extraction_prompt(self, text: str) -> Dict[str, str]:
        """
        Construye system prompt + user prompt para extracción.

        Returns:
            Dict con claves 'system' y 'user'.
        """
        entity_list = ", ".join(self._entity_types[:30])
        relation_list = ", ".join(self._relation_types[:25])

        system = (
            "Eres un experto en extracción de conocimiento de textos académicos "
            "de computación cuántica, matemáticas y física.\n\n"
            f"Tipos de entidad válidos: {entity_list}\n"
            f"Tipos de relación válidos: {relation_list}\n\n"
            "Extrae entidades y relaciones del texto. Responde SOLO con JSON "
            "con el siguiente esquema:\n"
            "{\n"
            '  "entities": [{"name": "...", "type": "..."}],\n'
            '  "triples": [\n'
            '    {"source": "...", "source_type": "...", '
            '"relation": "...", "target": "...", "target_type": "...", '
            '"confidence": 0.9}\n'
            "  ]\n"
            "}\n\n"
            "Reglas:\n"
            "- Solo incluir entidades claramente identificables\n"
            "- Usar tipos de la lista proporcionada\n"
            "- confidence entre 0.0 y 1.0\n"
            "- No inventar relaciones que no estén en el texto"
        )

        user = f"Texto a analizar:\n{text}"

        return {"system": system, "user": user}

    @property
    def valid_entity_types(self) -> List[str]:
        return list(self._entity_types)

    @property
    def valid_relation_types(self) -> List[str]:
        return list(self._relation_types)


class LLMGraphExtractor:
    """
    Extrae entidades y relaciones de texto usando LLM.

    Alineado con la ontología del proyecto para producir un grafo
    consistente y tipado.
    """

    def __init__(
        self,
        ontology_path: Optional[Path] = None,
        min_confidence: float = 0.7,
        max_text_chars: int = 3000,
    ):
        """
        Args:
            ontology_path: Ruta al archivo ontology.yaml.
            min_confidence: Confianza mínima para incluir un triple.
            max_text_chars: Máximo de caracteres de texto por llamada.
        """
        self.min_confidence = min_confidence
        self.max_text_chars = max_text_chars
        self.prompt_builder = OntologyPromptBuilder(ontology_path)

    # ------------------------------------------------------------------
    # Extracción individual
    # ------------------------------------------------------------------

    def extract_from_chunk(
        self, chunk_id: str, text: str
    ) -> ExtractionResult:
        """Extrae entidades y triples de un solo chunk."""
        truncated = text[: self.max_text_chars]
        prompts = self.prompt_builder.build_extraction_prompt(truncated)

        try:
            from src.llm_provider import complete as llm_complete

            response = llm_complete(
                prompt=prompts["user"],
                system=prompts["system"],
                temperature=0,
                max_tokens=1000,
                json_mode=True,
            )

            result = self._parse_response(chunk_id, response.content)
            return self.validate_against_ontology(result)

        except Exception as e:
            logger.warning(f"Error extrayendo de chunk {chunk_id}: {e}")
            return ExtractionResult(
                chunk_id=chunk_id, entities=[], triples=[]
            )

    # ------------------------------------------------------------------
    # Extracción por batch
    # ------------------------------------------------------------------

    def extract_batch(
        self, chunks: List[Tuple[str, str]]
    ) -> List[ExtractionResult]:
        """
        Extrae de múltiples chunks. Cada elemento es (chunk_id, text).

        Procesa todos en una sola llamada LLM si caben, o en sub-batches.
        """
        if not chunks:
            return []

        # Si son pocos y caben en el límite, batch único
        total_chars = sum(len(t) for _, t in chunks)
        if len(chunks) <= 5 and total_chars < self.max_text_chars * 3:
            return self._extract_multi_batch(chunks)

        # Si no, procesar uno a uno
        results = []
        for chunk_id, text in chunks:
            result = self.extract_from_chunk(chunk_id, text)
            results.append(result)
        return results

    def _extract_multi_batch(
        self, chunks: List[Tuple[str, str]]
    ) -> List[ExtractionResult]:
        """Procesa múltiples chunks en una llamada."""
        combined = ""
        for idx, (chunk_id, text) in enumerate(chunks):
            truncated = text[: self.max_text_chars]
            combined += f"\n--- Chunk {idx} (id: {chunk_id}) ---\n{truncated}\n"

        system = self.prompt_builder.build_extraction_prompt("")["system"]
        system += (
            "\n\nHay múltiples chunks. Responde con un JSON object donde "
            "las claves son los índices (0, 1, ...) y los valores tienen "
            'el esquema {"entities": [...], "triples": [...]}.'
        )

        try:
            from src.llm_provider import complete as llm_complete

            response = llm_complete(
                prompt=f"Textos a analizar:{combined}",
                system=system,
                temperature=0,
                max_tokens=1000 * len(chunks),
                json_mode=True,
            )

            parsed = json.loads(response.content)
            results = []

            for idx, (chunk_id, _) in enumerate(chunks):
                chunk_data = parsed.get(str(idx), parsed.get(idx, {}))
                if isinstance(chunk_data, dict):
                    result = self._build_result(chunk_id, chunk_data)
                    results.append(self.validate_against_ontology(result))
                else:
                    results.append(
                        ExtractionResult(chunk_id=chunk_id, entities=[], triples=[])
                    )

            return results

        except Exception as e:
            logger.warning(f"Error en batch de extracción: {e}")
            return [
                ExtractionResult(chunk_id=cid, entities=[], triples=[])
                for cid, _ in chunks
            ]

    # ------------------------------------------------------------------
    # Validación contra ontología
    # ------------------------------------------------------------------

    def validate_against_ontology(
        self, result: ExtractionResult
    ) -> ExtractionResult:
        """Filtra tipos inválidos y triples con baja confianza."""
        valid_entity_types = set(self.prompt_builder.valid_entity_types)
        valid_relation_types = set(self.prompt_builder.valid_relation_types)

        # Filtrar entidades con tipos inválidos
        filtered_entities = []
        for e in result.entities:
            if e.get("type") in valid_entity_types:
                filtered_entities.append(e)
            else:
                # Intentar mapear a tipo conocido
                mapped = self._fuzzy_match_type(
                    e.get("type", ""), valid_entity_types
                )
                if mapped:
                    e["type"] = mapped
                    filtered_entities.append(e)

        # Filtrar triples
        filtered_triples = []
        for triple in result.triples:
            if triple.confidence < self.min_confidence:
                continue
            if triple.relation_type not in valid_relation_types:
                mapped = self._fuzzy_match_type(
                    triple.relation_type, valid_relation_types
                )
                if mapped:
                    triple.relation_type = mapped
                else:
                    continue
            filtered_triples.append(triple)

        return ExtractionResult(
            chunk_id=result.chunk_id,
            entities=filtered_entities,
            triples=filtered_triples,
            raw_llm_response=result.raw_llm_response,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_response(
        self, chunk_id: str, raw: str
    ) -> ExtractionResult:
        """Parsea la respuesta JSON del LLM."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return ExtractionResult(
                chunk_id=chunk_id, entities=[], triples=[], raw_llm_response=raw
            )

        return self._build_result(chunk_id, parsed, raw)

    def _build_result(
        self,
        chunk_id: str,
        parsed: Dict,
        raw: str = "",
    ) -> ExtractionResult:
        """Construye ExtractionResult desde un dict parseado."""
        entities = [
            {"name": e.get("name", ""), "type": e.get("type", "Concepto")}
            for e in parsed.get("entities", [])
            if e.get("name")
        ]

        triples = []
        for t in parsed.get("triples", []):
            triples.append(
                ExtractedTriple(
                    source_name=t.get("source", ""),
                    source_type=t.get("source_type", "Concepto"),
                    relation_type=t.get("relation", "CO_OCCURS"),
                    target_name=t.get("target", ""),
                    target_type=t.get("target_type", "Concepto"),
                    confidence=float(t.get("confidence", 0.5)),
                    chunk_id=chunk_id,
                )
            )

        return ExtractionResult(
            chunk_id=chunk_id,
            entities=entities,
            triples=triples,
            raw_llm_response=raw,
        )

    @staticmethod
    def _fuzzy_match_type(
        candidate: str, valid_types: set
    ) -> Optional[str]:
        """Intenta mapear un tipo a uno válido (case-insensitive)."""
        if not candidate:
            return None
        candidate_lower = candidate.lower().strip()
        for vt in valid_types:
            if vt.lower() == candidate_lower:
                return vt
        return None
