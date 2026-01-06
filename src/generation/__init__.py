"""
Módulo de Generation - Síntesis de respuestas con citas.

Exporta el constructor de prompts, sintetizador y gestor de citas.
"""

from .prompt_builder import PromptBuilder, PromptTemplate
from .synthesizer import ResponseSynthesizer, GeneratedResponse
from .citation_injector import CitationInjector, Citation

__all__ = [
    "PromptBuilder",
    "PromptTemplate",
    "ResponseSynthesizer",
    "GeneratedResponse",
    "CitationInjector",
    "Citation"
]
