"""
Text Processing - Utilidades de procesamiento de texto para BM25 y búsqueda.

Centraliza la tokenización para que indexer y retriever usen la misma lógica.
"""

import re
import string
from typing import List

# Stopwords en español e inglés (las más comunes)
_STOPWORDS_ES = frozenset({
    "de", "la", "el", "en", "y", "a", "los", "del", "las", "un", "por",
    "con", "no", "una", "su", "para", "es", "al", "lo", "como", "más",
    "pero", "sus", "le", "ya", "o", "este", "sí", "porque", "esta",
    "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta",
    "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos",
    "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos",
    "e", "esto", "mí", "antes", "algunos", "qué", "unos", "yo", "otro",
    "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes",
    "nada", "muchos", "cual", "poco", "ella", "estar", "estas", "algunas",
    "algo", "nosotros", "mi", "mis", "tú", "te", "ti", "tu", "tus",
    "se", "que", "fue", "son", "ser", "ha", "sido", "tiene",
})

_STOPWORDS_EN = frozenset({
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her",
    "she", "or", "an", "will", "my", "one", "all", "would", "there",
    "their", "what", "so", "up", "out", "if", "about", "who", "get",
    "which", "go", "me", "when", "make", "can", "like", "time", "no",
    "just", "him", "know", "take", "people", "into", "year", "your",
    "some", "could", "them", "see", "other", "than", "then", "now",
    "look", "only", "come", "its", "over", "think", "also", "back",
    "after", "use", "two", "how", "our", "way", "even", "new",
    "is", "are", "was", "were", "been", "being", "has", "had",
})

STOPWORDS = _STOPWORDS_ES | _STOPWORDS_EN

# Tabla de traducción para quitar puntuación
_PUNCT_TABLE = str.maketrans("", "", string.punctuation + "¿¡«»—–""''·")


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokeniza texto para indexación/búsqueda BM25.

    - Lowercase
    - Quita puntuación
    - Split por whitespace
    - Filtra stopwords (ES + EN)
    - Filtra tokens < 2 chars

    Args:
        text: Texto a tokenizar

    Returns:
        Lista de tokens limpios
    """
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    tokens = text.split()
    return [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]
