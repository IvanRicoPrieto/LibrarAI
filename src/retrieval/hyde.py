# src/retrieval/hyde.py
"""
HyDE (Hypothetical Document Embeddings) for query expansion.

HyDE improves retrieval by:
1. Generating a hypothetical answer to the query using an LLM
2. Embedding the hypothetical answer instead of (or in addition to) the query
3. Using this embedding to find similar real documents

This resolves vocabulary mismatch between questions and documents,
especially for abstract or conceptual queries.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class HyDEConfig:
    """Configuration for HyDE query expansion."""
    
    # Temperature for generation (lower = more focused)
    temperature: float = 0.3
    
    # Max tokens for hypothetical document
    max_tokens: int = 300
    
    # Whether to combine query embedding with HyDE embedding
    combine_with_query: bool = True
    
    # Weight for original query embedding when combining (0-1)
    query_weight: float = 0.3


class HyDEExpander:
    """
    Generates hypothetical document embeddings for improved retrieval.
    
    Usage:
        hyde = HyDEExpander()
        
        # Get hypothetical document for a query
        hypo_doc = hyde.generate_hypothetical(
            query="What is quantum entanglement?",
            context_hint="physics textbook"
        )
        
        # Use the hypothetical document for embedding instead of query
        embedding = get_embedding(hypo_doc)
    """
    
    def __init__(
        self,
        config: Optional[HyDEConfig] = None,
    ):
        """
        Initialize HyDE expander.

        Args:
            config: HyDE configuration
        """
        self.config = config or HyDEConfig()

    def generate_hypothetical(
        self,
        query: str,
        context_hint: str = "technical documentation",
        domain: str = "quantum computing and physics",
    ) -> str:
        """
        Generate a hypothetical document that would answer the query.

        Args:
            query: The user's question
            context_hint: Type of document to simulate
            domain: Subject domain for the hypothetical content

        Returns:
            Hypothetical document text
        """
        from src.llm_provider import complete as llm_complete

        system_prompt = f"""You are an expert in {domain}.
Your task is to write a short, factual passage that would directly answer the given question.
Write as if you are excerpting from a {context_hint}.
Be technical, precise, and use appropriate terminology.
Do NOT say "I" or address the user. Just write the content directly.
Keep it under 200 words but make it information-dense."""

        user_prompt = f"""Question: {query}

Write a passage from a {context_hint} that would contain the answer to this question:"""

        try:
            response = llm_complete(
                prompt=user_prompt,
                system=system_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            hypothetical = response.content
            logger.debug(f"HyDE generated hypothetical ({len(hypothetical)} chars)")
            return hypothetical

        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}. Using original query.")
            return query
    
    def expand_query(
        self,
        query: str,
        context_hint: str = "technical documentation",
        domain: str = "quantum computing and physics",
    ) -> dict:
        """
        Expand a query using HyDE.
        
        Args:
            query: Original query
            context_hint: Type of document to simulate
            domain: Subject domain
            
        Returns:
            Dict with 'original', 'hypothetical', and 'combined' queries
        """
        hypothetical = self.generate_hypothetical(query, context_hint, domain)
        
        # Combine query and hypothetical for embedding
        if self.config.combine_with_query:
            combined = f"{query}\n\n{hypothetical}"
        else:
            combined = hypothetical
        
        return {
            "original": query,
            "hypothetical": hypothetical,
            "combined": combined,
            "query_weight": self.config.query_weight if self.config.combine_with_query else 0.0,
        }
    
    def get_embedding_texts(
        self,
        query: str,
        context_hint: str = "technical documentation",
        domain: str = "quantum computing and physics",
    ) -> List[tuple[str, float]]:
        """
        Get texts to embed with their weights.
        
        Returns list of (text, weight) tuples for embedding.
        When combine_with_query=True, returns both query and hypothetical.
        
        Args:
            query: Original query
            context_hint: Type of document to simulate
            domain: Subject domain
            
        Returns:
            List of (text, weight) tuples
        """
        hypothetical = self.generate_hypothetical(query, context_hint, domain)
        
        if self.config.combine_with_query:
            return [
                (query, self.config.query_weight),
                (hypothetical, 1.0 - self.config.query_weight),
            ]
        else:
            return [(hypothetical, 1.0)]


# Singleton instance for convenience
_default_hyde: Optional[HyDEExpander] = None


def get_hyde_expander() -> HyDEExpander:
    """Get or create default HyDE expander."""
    global _default_hyde
    if _default_hyde is None:
        _default_hyde = HyDEExpander()
    return _default_hyde
