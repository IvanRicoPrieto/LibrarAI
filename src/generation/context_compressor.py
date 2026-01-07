"""
Context Compressor - Compresión de contexto para reducir tokens.

Implementa técnicas de compresión para permitir más contexto
en el mismo presupuesto de tokens:

1. Compresión heurística (sin modelo adicional):
   - Eliminación de redundancias
   - Compactación de espacios en blanco
   - Extracción de oraciones clave
   - Poda de secciones menos relevantes

2. Compresión con LLMLingua (opcional):
   - Compresión semántica usando modelo pequeño
   - Preserva información crítica
   - Reducción 50-70% de tokens

Referencias:
- LLMLingua: https://github.com/microsoft/LLMLingua
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CompressionLevel(Enum):
    """Niveles de compresión disponibles."""
    NONE = "none"           # Sin compresión
    LIGHT = "light"         # Solo limpieza básica (~10-20% reducción)
    MEDIUM = "medium"       # Compresión moderada (~30-40% reducción)
    AGGRESSIVE = "aggressive"  # Compresión agresiva (~50-60% reducción)


@dataclass
class CompressionConfig:
    """Configuración del compresor."""
    level: CompressionLevel = CompressionLevel.MEDIUM
    target_ratio: float = 0.5  # Ratio objetivo de compresión (0.5 = 50% del original)
    preserve_citations: bool = True  # Mantener marcadores de cita [n]
    preserve_math: bool = True  # Mantener fórmulas LaTeX
    preserve_code: bool = True  # Mantener bloques de código
    use_llmlingua: bool = False  # Usar LLMLingua si está disponible
    min_sentence_importance: float = 0.3  # Umbral para filtrar oraciones


@dataclass
class CompressionResult:
    """Resultado de la compresión."""
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    method: str
    preserved_elements: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": self.compression_ratio,
            "method": self.method,
            "preserved_elements": self.preserved_elements
        }


class ContextCompressor:
    """
    Compresor de contexto para RAG.
    
    Reduce el número de tokens manteniendo la información relevante,
    permitiendo incluir más contexto en consultas al LLM.
    """
    
    # Patrones para preservar
    CITATION_PATTERN = re.compile(r'\[\d+\]')
    LATEX_INLINE_PATTERN = re.compile(r'\$[^$]+\$')
    LATEX_BLOCK_PATTERN = re.compile(r'\$\$[^$]+\$\$', re.DOTALL)
    CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```')
    
    # Palabras vacías (stop words) para filtrar
    STOP_WORDS = {
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
        'de', 'del', 'al', 'a', 'en', 'por', 'para', 'con', 'sin',
        'y', 'o', 'pero', 'que', 'como', 'más', 'menos',
        'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas',
        'su', 'sus', 'se', 'le', 'les', 'lo', 'nos', 'me', 'te',
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'and', 'or', 'but', 'not', 'as', 'this', 'that', 'these', 'those'
    }
    
    # Palabras clave importantes en contexto científico
    IMPORTANT_KEYWORDS = {
        # Cuántica
        'qubit', 'quantum', 'cuántico', 'entrelazamiento', 'entanglement',
        'superposición', 'superposition', 'medición', 'measurement',
        'estado', 'state', 'gate', 'puerta', 'circuito', 'circuit',
        'algoritmo', 'algorithm', 'protocolo', 'protocol',
        # Matemáticas
        'teorema', 'theorem', 'lema', 'lemma', 'definición', 'definition',
        'demostración', 'proof', 'corolario', 'corollary',
        'ecuación', 'equation', 'fórmula', 'formula',
        # Técnico
        'clave', 'key', 'seguridad', 'security', 'error', 'corrección',
        'código', 'code', 'canal', 'channel', 'información', 'information'
    }
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        """
        Args:
            config: Configuración del compresor
        """
        self.config = config or CompressionConfig()
        self._llmlingua_model = None
    
    def _estimate_tokens(self, text: str) -> int:
        """Estima el número de tokens (aproximación)."""
        # Regla general: ~4 caracteres por token en inglés/español
        return len(text) // 4
    
    def _extract_preserved_elements(self, text: str) -> Tuple[str, Dict[str, List[str]]]:
        """
        Extrae elementos que deben preservarse y los reemplaza con placeholders.
        
        Returns:
            Tuple (texto con placeholders, diccionario de elementos extraídos)
        """
        preserved = {
            "citations": [],
            "latex_blocks": [],
            "latex_inline": [],
            "code_blocks": []
        }
        
        result = text
        
        # Extraer bloques de código
        if self.config.preserve_code:
            code_blocks = self.CODE_BLOCK_PATTERN.findall(result)
            for i, block in enumerate(code_blocks):
                preserved["code_blocks"].append(block)
                result = result.replace(block, f"__CODE_BLOCK_{i}__", 1)
        
        # Extraer LaTeX bloques
        if self.config.preserve_math:
            latex_blocks = self.LATEX_BLOCK_PATTERN.findall(result)
            for i, block in enumerate(latex_blocks):
                preserved["latex_blocks"].append(block)
                result = result.replace(block, f"__LATEX_BLOCK_{i}__", 1)
            
            # Extraer LaTeX inline
            latex_inline = self.LATEX_INLINE_PATTERN.findall(result)
            for i, inline in enumerate(latex_inline):
                preserved["latex_inline"].append(inline)
                result = result.replace(inline, f"__LATEX_INLINE_{i}__", 1)
        
        # Extraer citas
        if self.config.preserve_citations:
            citations = self.CITATION_PATTERN.findall(result)
            preserved["citations"] = list(set(citations))
        
        return result, preserved
    
    def _restore_preserved_elements(
        self, 
        text: str, 
        preserved: Dict[str, List[str]]
    ) -> str:
        """Restaura los elementos preservados desde placeholders."""
        result = text
        
        # Restaurar código
        for i, block in enumerate(preserved.get("code_blocks", [])):
            result = result.replace(f"__CODE_BLOCK_{i}__", block)
        
        # Restaurar LaTeX bloques
        for i, block in enumerate(preserved.get("latex_blocks", [])):
            result = result.replace(f"__LATEX_BLOCK_{i}__", block)
        
        # Restaurar LaTeX inline
        for i, inline in enumerate(preserved.get("latex_inline", [])):
            result = result.replace(f"__LATEX_INLINE_{i}__", inline)
        
        return result
    
    def _clean_whitespace(self, text: str) -> str:
        """Limpia espacios en blanco redundantes."""
        # Múltiples espacios a uno
        text = re.sub(r' +', ' ', text)
        # Múltiples newlines a dos
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Espacios al inicio/final de líneas
        text = '\n'.join(line.strip() for line in text.split('\n'))
        return text.strip()
    
    def _calculate_sentence_importance(self, sentence: str) -> float:
        """
        Calcula la importancia de una oración basándose en palabras clave.
        
        Returns:
            Score de importancia (0-1)
        """
        words = set(sentence.lower().split())
        
        # Contar palabras clave importantes
        keyword_count = len(words.intersection(self.IMPORTANT_KEYWORDS))
        
        # Penalizar oraciones muy cortas o muy largas
        word_count = len(words)
        length_penalty = 1.0
        if word_count < 5:
            length_penalty = 0.5
        elif word_count > 50:
            length_penalty = 0.8
        
        # Bonus por contener fórmulas o citas
        bonus = 0
        if '$' in sentence:  # Contiene LaTeX
            bonus += 0.3
        if re.search(r'\[\d+\]', sentence):  # Contiene cita
            bonus += 0.2
        
        # Score normalizado
        score = min(1.0, (keyword_count / 5) * length_penalty + bonus)
        
        return score
    
    def _compress_light(self, text: str) -> str:
        """Compresión ligera: solo limpieza."""
        return self._clean_whitespace(text)
    
    def _compress_medium(self, text: str) -> str:
        """
        Compresión media: limpieza + filtrado de oraciones menos importantes.
        """
        text = self._clean_whitespace(text)
        
        # Dividir en oraciones
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filtrar oraciones por importancia
        filtered_sentences = []
        for sentence in sentences:
            importance = self._calculate_sentence_importance(sentence)
            if importance >= self.config.min_sentence_importance:
                filtered_sentences.append(sentence)
        
        return ' '.join(filtered_sentences)
    
    def _compress_aggressive(self, text: str) -> str:
        """
        Compresión agresiva: extractivo + eliminación de redundancias.
        """
        text = self._clean_whitespace(text)
        
        # Dividir en oraciones
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Calcular importancia de cada oración
        scored_sentences = [
            (sentence, self._calculate_sentence_importance(sentence))
            for sentence in sentences
        ]
        
        # Ordenar por importancia
        scored_sentences.sort(key=lambda x: -x[1])
        
        # Tomar las más importantes hasta alcanzar el ratio objetivo
        target_tokens = int(self._estimate_tokens(text) * self.config.target_ratio)
        
        selected = []
        current_tokens = 0
        
        for sentence, score in scored_sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            if current_tokens + sentence_tokens <= target_tokens:
                selected.append(sentence)
                current_tokens += sentence_tokens
        
        # Reordenar cronológicamente (por posición original)
        original_order = {s: i for i, s in enumerate(sentences)}
        selected.sort(key=lambda s: original_order.get(s, float('inf')))
        
        return ' '.join(selected)
    
    def _compress_with_llmlingua(self, text: str) -> str:
        """
        Compresión usando LLMLingua (si está disponible).
        
        Requiere: pip install llmlingua
        """
        try:
            if self._llmlingua_model is None:
                from llmlingua import PromptCompressor
                self._llmlingua_model = PromptCompressor(
                    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                    use_llmlingua2=True
                )
                logger.info("LLMLingua cargado correctamente")
            
            result = self._llmlingua_model.compress_prompt(
                text,
                rate=self.config.target_ratio,
                force_tokens=['\n', '.', '!', '?', ','],
                drop_consecutive=True
            )
            
            return result["compressed_prompt"]
            
        except ImportError:
            logger.warning(
                "LLMLingua no instalado. Usando compresión heurística. "
                "Instala con: pip install llmlingua"
            )
            return self._compress_aggressive(text)
        except Exception as e:
            logger.error(f"Error en LLMLingua: {e}. Usando compresión heurística.")
            return self._compress_aggressive(text)
    
    def compress(self, text: str) -> CompressionResult:
        """
        Comprime el texto según la configuración.
        
        Args:
            text: Texto a comprimir
            
        Returns:
            CompressionResult con texto comprimido y estadísticas
        """
        original_tokens = self._estimate_tokens(text)
        
        # Sin compresión
        if self.config.level == CompressionLevel.NONE:
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                method="none"
            )
        
        # Extraer elementos a preservar
        working_text, preserved = self._extract_preserved_elements(text)
        
        # Aplicar compresión según nivel
        if self.config.use_llmlingua and self.config.level in (
            CompressionLevel.MEDIUM, CompressionLevel.AGGRESSIVE
        ):
            method = "llmlingua"
            compressed = self._compress_with_llmlingua(working_text)
        elif self.config.level == CompressionLevel.LIGHT:
            method = "heuristic_light"
            compressed = self._compress_light(working_text)
        elif self.config.level == CompressionLevel.MEDIUM:
            method = "heuristic_medium"
            compressed = self._compress_medium(working_text)
        else:  # AGGRESSIVE
            method = "heuristic_aggressive"
            compressed = self._compress_aggressive(working_text)
        
        # Restaurar elementos preservados
        compressed = self._restore_preserved_elements(compressed, preserved)
        compressed = self._clean_whitespace(compressed)
        
        compressed_tokens = self._estimate_tokens(compressed)
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        
        preserved_counts = {
            "citations": len(preserved.get("citations", [])),
            "latex_blocks": len(preserved.get("latex_blocks", [])),
            "latex_inline": len(preserved.get("latex_inline", [])),
            "code_blocks": len(preserved.get("code_blocks", []))
        }
        
        logger.info(
            f"Compresión completada: {original_tokens} → {compressed_tokens} tokens "
            f"(ratio: {compression_ratio:.2%}, método: {method})"
        )
        
        return CompressionResult(
            original_text=text,
            compressed_text=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            method=method,
            preserved_elements=preserved_counts
        )
    
    def compress_contexts(
        self, 
        contexts: List[str],
        max_total_tokens: int = 4000
    ) -> Tuple[List[str], Dict[str, any]]:
        """
        Comprime una lista de contextos para caber en el presupuesto de tokens.
        
        Args:
            contexts: Lista de textos de contexto
            max_total_tokens: Máximo de tokens permitidos
            
        Returns:
            Tuple (contextos comprimidos, estadísticas)
        """
        # Calcular tokens actuales
        current_tokens = sum(self._estimate_tokens(c) for c in contexts)
        
        # Si ya cabemos, no comprimir
        if current_tokens <= max_total_tokens:
            return contexts, {
                "original_tokens": current_tokens,
                "compressed_tokens": current_tokens,
                "compression_applied": False
            }
        
        # Calcular ratio necesario
        needed_ratio = max_total_tokens / current_tokens
        
        # Ajustar nivel de compresión según ratio necesario
        if needed_ratio > 0.8:
            self.config.level = CompressionLevel.LIGHT
        elif needed_ratio > 0.5:
            self.config.level = CompressionLevel.MEDIUM
        else:
            self.config.level = CompressionLevel.AGGRESSIVE
        
        self.config.target_ratio = needed_ratio
        
        # Comprimir cada contexto
        compressed_contexts = []
        total_original = 0
        total_compressed = 0
        
        for ctx in contexts:
            result = self.compress(ctx)
            compressed_contexts.append(result.compressed_text)
            total_original += result.original_tokens
            total_compressed += result.compressed_tokens
        
        return compressed_contexts, {
            "original_tokens": total_original,
            "compressed_tokens": total_compressed,
            "compression_ratio": total_compressed / total_original if total_original > 0 else 1.0,
            "compression_applied": True,
            "level": self.config.level.value
        }


# Singleton para reutilización
_compressor_instance: Optional[ContextCompressor] = None


def get_context_compressor(config: Optional[CompressionConfig] = None) -> ContextCompressor:
    """Obtiene instancia singleton del compresor."""
    global _compressor_instance
    if _compressor_instance is None or config is not None:
        _compressor_instance = ContextCompressor(config)
    return _compressor_instance
