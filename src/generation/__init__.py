"""
Módulo de Generation - Síntesis de respuestas con citas.

Exporta el constructor de prompts, sintetizador, gestor de citas
y compresor de contexto.
"""

from .prompt_builder import PromptBuilder, PromptTemplate
from .synthesizer import ResponseSynthesizer, GeneratedResponse
from .citation_injector import CitationInjector, Citation
from .context_compressor import (
    ContextCompressor, 
    CompressionConfig, 
    CompressionLevel, 
    CompressionResult,
    get_context_compressor
)

__all__ = [
    "PromptBuilder",
    "PromptTemplate",
    "ResponseSynthesizer",
    "GeneratedResponse",
    "CitationInjector",
    "Citation",
    "ContextCompressor",
    "CompressionConfig",
    "CompressionLevel",
    "CompressionResult",
    "get_context_compressor"
]
