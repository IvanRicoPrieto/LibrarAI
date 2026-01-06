# src/evaluation/__init__.py
"""
Evaluation pipeline for RAG quality metrics.
Implements RAGAS-style evaluation: faithfulness, relevancy, context precision.
"""

from .metrics import (
    RAGASEvaluator,
    EvaluationResult,
    EvaluationConfig,
)
from .benchmark import (
    BenchmarkRunner,
    BenchmarkSuite,
    TestCase,
)

__all__ = [
    "RAGASEvaluator",
    "EvaluationResult", 
    "EvaluationConfig",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "TestCase",
]
