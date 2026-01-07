"""
Prompt Builder - Construcción de prompts estructurados para RAG.

Soporta:
- Templates para diferentes tipos de consulta
- Inyección de contexto con metadatos
- Formateo de citas
- Compresión de contexto (opcional)
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

from ..retrieval.fusion import RetrievalResult

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Tipos de consulta detectados."""
    FACTUAL = "factual"           # Pregunta factual directa
    COMPARATIVE = "comparative"    # Comparación entre conceptos
    EXPLANATORY = "explanatory"    # Explicación de concepto
    PROCEDURAL = "procedural"      # Cómo hacer algo
    SYNTHESIS = "synthesis"        # Síntesis de múltiples fuentes
    MATHEMATICAL = "mathematical"  # Derivación/demostración matemática


@dataclass
class PromptTemplate:
    """Template de prompt."""
    name: str
    system_prompt: str
    user_template: str
    context_template: str
    
    def format(
        self,
        query: str,
        contexts: List[str],
        metadata: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        Formatea el template con los datos.
        
        Returns:
            Dict con 'system' y 'user' messages
        """
        # Formatear contextos
        formatted_contexts = []
        for i, ctx in enumerate(contexts, 1):
            formatted = self.context_template.format(
                index=i,
                content=ctx,
                **(metadata or {})
            )
            formatted_contexts.append(formatted)
        
        context_block = "\n\n".join(formatted_contexts)
        
        user_message = self.user_template.format(
            query=query,
            context=context_block
        )
        
        return {
            "system": self.system_prompt,
            "user": user_message
        }


class PromptBuilder:
    """
    Constructor de prompts para RAG.
    
    Características:
    - Detección automática de tipo de query
    - Templates especializados por tipo
    - Formateo de contexto con citas
    """
    
    # Templates predefinidos
    TEMPLATES = {
        QueryType.FACTUAL: PromptTemplate(
            name="factual",
            system_prompt="""Eres un asistente experto en computación cuántica e información cuántica.
Tu objetivo es responder preguntas de forma precisa y concisa, citando las fuentes proporcionadas.

REGLAS:
1. Basa tus respuestas ÚNICAMENTE en el contexto proporcionado
2. Cita las fuentes usando [n] donde n es el número del fragmento
3. Si la información no está en el contexto, indica que no tienes datos suficientes
4. Usa notación LaTeX para fórmulas: $inline$ o $$block$$
5. Sé conciso pero completo""",
            user_template="""CONTEXTO DE LA BIBLIOTECA:
{context}

---

PREGUNTA: {query}

Responde basándote en el contexto anterior, citando las fuentes con [n].""",
            context_template="[{index}] {content}"
        ),
        
        QueryType.EXPLANATORY: PromptTemplate(
            name="explanatory",
            system_prompt="""Eres un profesor experto en computación cuántica.
Tu objetivo es explicar conceptos de forma clara y pedagógica.

REGLAS:
1. Explica el concepto paso a paso
2. Usa analogías cuando sea apropiado
3. Incluye las fórmulas matemáticas relevantes (LaTeX)
4. Cita las fuentes con [n]
5. Proporciona ejemplos del contexto cuando estén disponibles""",
            user_template="""CONTEXTO DE LA BIBLIOTECA:
{context}

---

CONCEPTO A EXPLICAR: {query}

Proporciona una explicación clara y estructurada, citando las fuentes.""",
            context_template="[{index}] Fuente: {content}"
        ),
        
        QueryType.COMPARATIVE: PromptTemplate(
            name="comparative",
            system_prompt="""Eres un analista experto en computación cuántica.
Tu objetivo es comparar conceptos de forma objetiva y estructurada.

REGLAS:
1. Identifica los elementos a comparar
2. Usa una estructura clara (tabla si es apropiado)
3. Destaca similitudes y diferencias
4. Cita las fuentes con [n]
5. Concluye con una síntesis""",
            user_template="""CONTEXTO DE LA BIBLIOTECA:
{context}

---

COMPARACIÓN SOLICITADA: {query}

Realiza una comparación estructurada citando las fuentes.""",
            context_template="[{index}] {content}"
        ),
        
        QueryType.PROCEDURAL: PromptTemplate(
            name="procedural",
            system_prompt="""Eres un instructor de computación cuántica.
Tu objetivo es explicar procedimientos y algoritmos paso a paso.

REGLAS:
1. Enumera los pasos claramente
2. Incluye el circuito cuántico si es relevante
3. Explica la intuición detrás de cada paso
4. Cita las fuentes con [n]
5. Menciona casos especiales o variantes""",
            user_template="""CONTEXTO DE LA BIBLIOTECA:
{context}

---

PROCEDIMIENTO A EXPLICAR: {query}

Describe el procedimiento paso a paso, citando las fuentes.""",
            context_template="[{index}] {content}"
        ),
        
        QueryType.SYNTHESIS: PromptTemplate(
            name="synthesis",
            system_prompt="""Eres un investigador experto en computación cuántica.
Tu objetivo es sintetizar información de múltiples fuentes.

REGLAS:
1. Integra información de todas las fuentes relevantes
2. Identifica consensos y discrepancias
3. Estructura la síntesis de forma coherente
4. Cita cada afirmación con [n]
5. Proporciona una conclusión integradora""",
            user_template="""CONTEXTO DE LA BIBLIOTECA:
{context}

---

TEMA A SINTETIZAR: {query}

Proporciona una síntesis integradora de las fuentes.""",
            context_template="[{index}] De '{doc_title}': {content}"
        ),
        
        QueryType.MATHEMATICAL: PromptTemplate(
            name="mathematical",
            system_prompt="""Eres un físico matemático experto en mecánica cuántica.
Tu objetivo es explicar derivaciones y demostraciones matemáticas.

REGLAS:
1. Presenta las fórmulas con notación LaTeX correcta
2. Explica cada paso de la derivación
3. Menciona las suposiciones y aproximaciones
4. Cita las fuentes con [n]
5. Incluye la notación de Dirac cuando sea apropiado""",
            user_template="""CONTEXTO DE LA BIBLIOTECA:
{context}

---

DERIVACIÓN/DEMOSTRACIÓN: {query}

Desarrolla la matemática paso a paso, citando las fuentes.""",
            context_template="[{index}] {content}"
        )
    }
    
    # Palabras clave para detectar tipo de query
    QUERY_KEYWORDS = {
        QueryType.COMPARATIVE: ["compara", "diferencia", "vs", "versus", "mejor", "ventajas", "desventajas"],
        QueryType.PROCEDURAL: ["cómo", "pasos", "procedimiento", "implementar", "construir", "aplicar"],
        QueryType.MATHEMATICAL: ["demostrar", "derivar", "calcular", "fórmula", "ecuación", "operador"],
        QueryType.SYNTHESIS: ["resumen", "síntesis", "overview", "revisión", "estado del arte"],
        QueryType.EXPLANATORY: ["explica", "qué es", "definición", "concepto", "significa"]
    }
    
    def __init__(self, default_type: QueryType = QueryType.FACTUAL):
        """
        Args:
            default_type: Tipo de query por defecto
        """
        self.default_type = default_type
        self._compressor = None
    
    def _get_compressor(self):
        """Obtiene el compresor de contexto (lazy loading)."""
        if self._compressor is None:
            try:
                from .context_compressor import get_context_compressor, CompressionConfig, CompressionLevel
                config = CompressionConfig(level=CompressionLevel.MEDIUM)
                self._compressor = get_context_compressor(config)
            except ImportError:
                logger.debug("Compresor de contexto no disponible")
                self._compressor = False  # Marker para no reintentar
        return self._compressor if self._compressor else None
    
    def detect_query_type(self, query: str) -> QueryType:
        """
        Detecta el tipo de consulta basándose en palabras clave.
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Tipo de query detectado
        """
        query_lower = query.lower()
        
        for query_type, keywords in self.QUERY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    logger.debug(f"Query type detected: {query_type.value} (keyword: {keyword})")
                    return query_type
        
        return self.default_type
    
    def build_prompt(
        self,
        query: str,
        results: List[RetrievalResult],
        query_type: Optional[QueryType] = None,
        max_context_tokens: int = 4000,
        include_metadata: bool = True,
        compress_context: bool = False
    ) -> Dict[str, str]:
        """
        Construye el prompt completo para el LLM.
        
        Args:
            query: Consulta del usuario
            results: Resultados del retrieval
            query_type: Tipo de query (auto-detectado si None)
            max_context_tokens: Límite de tokens para contexto
            include_metadata: Si incluir metadatos en contexto
            compress_context: Si comprimir contexto para caber en límite
            
        Returns:
            Dict con 'system' y 'user' messages
        """
        # Detectar tipo si no se proporciona
        if query_type is None:
            query_type = self.detect_query_type(query)
        
        # Obtener template
        template = self.TEMPLATES.get(query_type, self.TEMPLATES[QueryType.FACTUAL])
        
        # Preparar contextos
        contexts = []
        total_tokens = 0
        
        for result in results:
            # Estimar tokens (aprox 4 chars/token)
            chunk_tokens = len(result.content) // 4
            
            if total_tokens + chunk_tokens > max_context_tokens:
                break
            
            if include_metadata:
                context = f"[Fuente: {result.doc_title} | {result.header_path}]\n{result.content}"
            else:
                context = result.content
            
            contexts.append(context)
            total_tokens += chunk_tokens
        
        # Comprimir contexto si está habilitado y excede el límite
        compression_stats = None
        if compress_context and total_tokens > max_context_tokens * 0.8:
            compressor = self._get_compressor()
            if compressor:
                contexts, compression_stats = compressor.compress_contexts(
                    contexts, 
                    max_total_tokens=max_context_tokens
                )
                if compression_stats.get("compression_applied"):
                    logger.info(
                        f"Contexto comprimido: {compression_stats['original_tokens']} → "
                        f"{compression_stats['compressed_tokens']} tokens "
                        f"({compression_stats['compression_ratio']:.1%})"
                    )
        
        # Metadata para el template
        metadata = {
            "doc_title": results[0].doc_title if results else "",
            "header_path": results[0].header_path if results else ""
        }
        
        # Formatear
        prompt = template.format(query, contexts, metadata)
        
        # Añadir stats de compresión si aplica
        if compression_stats:
            prompt["_compression_stats"] = compression_stats
        
        logger.info(
            f"Prompt construido: tipo={query_type.value}, "
            f"contextos={len(contexts)}, tokens≈{total_tokens}"
        )
        
        return prompt
    
    def build_followup_prompt(
        self,
        original_query: str,
        original_response: str,
        followup_query: str,
        results: List[RetrievalResult]
    ) -> Dict[str, str]:
        """
        Construye prompt para pregunta de seguimiento.
        
        Args:
            original_query: Pregunta original
            original_response: Respuesta anterior
            followup_query: Nueva pregunta
            results: Nuevos resultados de retrieval
            
        Returns:
            Prompt formateado
        """
        system_prompt = """Eres un asistente experto en computación cuántica.
Estás continuando una conversación previa. Mantén coherencia con lo discutido.

REGLAS:
1. Considera el contexto de la conversación anterior
2. Basa nuevas afirmaciones en el contexto proporcionado
3. Cita fuentes con [n]
4. Si la pregunta de seguimiento requiere información no disponible, indícalo"""
        
        # Preparar contextos
        contexts = [r.content for r in results[:5]]
        context_block = "\n\n".join(
            f"[{i}] {ctx}" for i, ctx in enumerate(contexts, 1)
        )
        
        user_prompt = f"""CONVERSACIÓN ANTERIOR:
Pregunta: {original_query}
Respuesta: {original_response}

---

NUEVO CONTEXTO:
{context_block}

---

PREGUNTA DE SEGUIMIENTO: {followup_query}

Responde considerando la conversación anterior y el nuevo contexto."""
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
