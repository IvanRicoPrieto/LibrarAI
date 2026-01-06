# src/evaluation/benchmark.py
"""
Benchmark suite for systematic RAG evaluation.

Provides infrastructure for:
- Defining test cases with queries and expected behaviors
- Running benchmarks against the RAG pipeline
- Comparing performance before/after changes
- Generating reports
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .metrics import EvaluationResult, RAGASEvaluator, EvaluationConfig

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """
    A single test case for RAG evaluation.
    
    Attributes:
        query: The question to ask
        ground_truth: Expected answer (optional, for recall)
        category: Test category (e.g., "factual", "conceptual", "relational")
        difficulty: Expected difficulty level
        tags: Additional tags for filtering
        expected_sources: Expected source files/books
    """
    query: str
    ground_truth: Optional[str] = None
    category: str = "general"
    difficulty: str = "medium"
    tags: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    
    # Filled during evaluation
    answer: Optional[str] = None
    contexts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    evaluation: Optional[EvaluationResult] = None
    latency_ms: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "ground_truth": self.ground_truth,
            "category": self.category,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "expected_sources": self.expected_sources,
            "answer": self.answer,
            "num_contexts": len(self.contexts),
            "sources": self.sources,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "latency_ms": self.latency_ms,
        }


@dataclass
class BenchmarkSuite:
    """
    Collection of test cases organized by category.
    
    Usage:
        suite = BenchmarkSuite.load_from_yaml("benchmarks/quantum.yaml")
        suite = BenchmarkSuite.default_quantum_suite()
    """
    name: str
    description: str
    test_cases: list[TestCase] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_case(self, case: TestCase):
        """Add a test case to the suite."""
        self.test_cases.append(case)
    
    def filter_by_category(self, category: str) -> list[TestCase]:
        """Get test cases by category."""
        return [tc for tc in self.test_cases if tc.category == category]
    
    def filter_by_tags(self, tags: list[str]) -> list[TestCase]:
        """Get test cases that have all specified tags."""
        return [tc for tc in self.test_cases if all(t in tc.tags for t in tags)]
    
    @classmethod
    def default_quantum_suite(cls) -> "BenchmarkSuite":
        """Create a default benchmark suite for quantum computing library."""
        suite = cls(
            name="LibrarAI Quantum Benchmark",
            description="Standard test suite for quantum computing RAG evaluation",
        )
        
        # Factual queries (exact retrieval)
        suite.add_case(TestCase(
            query="What is the mathematical definition of quantum entanglement?",
            category="factual",
            difficulty="medium",
            tags=["entanglement", "definition"],
        ))
        
        suite.add_case(TestCase(
            query="What are the DiVincenzo criteria for quantum computing?",
            category="factual",
            difficulty="medium",
            tags=["criteria", "hardware"],
        ))
        
        suite.add_case(TestCase(
            query="How is the Bloch sphere representation defined?",
            category="factual",
            difficulty="easy",
            tags=["bloch", "representation"],
        ))
        
        # Conceptual queries (understanding)
        suite.add_case(TestCase(
            query="Explain the relationship between quantum gates and unitary matrices",
            category="conceptual",
            difficulty="medium",
            tags=["gates", "linear_algebra"],
        ))
        
        suite.add_case(TestCase(
            query="Why is quantum error correction necessary and what are its main approaches?",
            category="conceptual",
            difficulty="hard",
            tags=["error_correction", "qec"],
        ))
        
        suite.add_case(TestCase(
            query="How does quantum teleportation work and why doesn't it violate no-cloning?",
            category="conceptual",
            difficulty="hard",
            tags=["teleportation", "no_cloning"],
        ))
        
        # Relational queries (graph-based)
        suite.add_case(TestCase(
            query="What topics relate Grover's algorithm to quantum speedup?",
            category="relational",
            difficulty="medium",
            tags=["grover", "speedup", "graph"],
        ))
        
        suite.add_case(TestCase(
            query="How are Pauli matrices connected to quantum gates?",
            category="relational", 
            difficulty="easy",
            tags=["pauli", "gates", "graph"],
        ))
        
        # Cross-domain queries
        suite.add_case(TestCase(
            query="How does linear algebra relate to quantum computing?",
            category="cross_domain",
            difficulty="medium",
            tags=["linear_algebra", "foundations"],
        ))
        
        suite.add_case(TestCase(
            query="What information theory concepts are used in quantum error correction?",
            category="cross_domain",
            difficulty="hard",
            tags=["information_theory", "qec"],
        ))
        
        # Edge cases
        suite.add_case(TestCase(
            query="asdfghjkl quantum",  # Nonsense with keyword
            category="edge_case",
            difficulty="special",
            tags=["robustness"],
        ))
        
        suite.add_case(TestCase(
            query="",  # Empty query
            category="edge_case",
            difficulty="special", 
            tags=["robustness"],
        ))
        
        return suite
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "num_cases": len(self.test_cases),
            "test_cases": [tc.to_dict() for tc in self.test_cases],
        }
    
    def save_to_json(self, path: Path | str):
        """Save suite to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_json(cls, path: Path | str) -> "BenchmarkSuite":
        """Load suite from JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        suite = cls(
            name=data["name"],
            description=data["description"],
            created_at=data.get("created_at", datetime.now().isoformat()),
        )
        
        for tc_data in data.get("test_cases", []):
            suite.add_case(TestCase(
                query=tc_data["query"],
                ground_truth=tc_data.get("ground_truth"),
                category=tc_data.get("category", "general"),
                difficulty=tc_data.get("difficulty", "medium"),
                tags=tc_data.get("tags", []),
                expected_sources=tc_data.get("expected_sources", []),
            ))
        
        return suite


class BenchmarkRunner:
    """
    Runs benchmarks against a RAG pipeline and generates reports.
    
    Usage:
        runner = BenchmarkRunner(rag_pipeline)
        results = runner.run_suite(suite)
        runner.generate_report(results, "benchmark_report.md")
    """
    
    def __init__(
        self,
        rag_func: Optional[Callable[[str], dict]] = None,
        evaluator: Optional[RAGASEvaluator] = None,
    ):
        """
        Initialize benchmark runner.
        
        Args:
            rag_func: Function that takes query and returns dict with 'answer' and 'contexts'
            evaluator: RAGAS evaluator instance
        """
        self.rag_func = rag_func
        self.evaluator = evaluator or RAGASEvaluator()
        self._results_history: list[dict] = []
    
    def set_rag_function(self, rag_func: Callable[[str], dict]):
        """Set the RAG function to benchmark."""
        self.rag_func = rag_func
    
    def run_suite(
        self,
        suite: BenchmarkSuite,
        skip_empty_queries: bool = True,
    ) -> BenchmarkSuite:
        """
        Run all test cases in a suite.
        
        Args:
            suite: The benchmark suite to run
            skip_empty_queries: Whether to skip empty/invalid queries
            
        Returns:
            Updated suite with results
        """
        if self.rag_func is None:
            raise ValueError("RAG function not set. Use set_rag_function() first.")
        
        import time
        
        for i, test_case in enumerate(suite.test_cases):
            query = test_case.query.strip()
            
            if skip_empty_queries and not query:
                logger.info(f"Skipping empty query (case {i+1})")
                continue
            
            logger.info(f"Running case {i+1}/{len(suite.test_cases)}: {query[:50]}...")
            
            try:
                # Run RAG pipeline
                start_time = time.time()
                result = self.rag_func(query)
                test_case.latency_ms = (time.time() - start_time) * 1000
                
                # Extract results
                test_case.answer = result.get("answer", "")
                test_case.contexts = result.get("contexts", [])
                test_case.sources = result.get("sources", [])
                
                # Evaluate
                if test_case.answer and test_case.contexts:
                    test_case.evaluation = self.evaluator.evaluate(
                        query=query,
                        answer=test_case.answer,
                        contexts=test_case.contexts,
                        ground_truth=test_case.ground_truth,
                    )
                    
            except Exception as e:
                logger.error(f"Case {i+1} failed: {e}")
                test_case.answer = f"ERROR: {e}"
        
        return suite
    
    def generate_report(
        self,
        suite: BenchmarkSuite,
        output_path: Optional[Path | str] = None,
    ) -> str:
        """
        Generate a markdown report from benchmark results.
        
        Args:
            suite: Completed benchmark suite
            output_path: Optional path to save report
            
        Returns:
            Report as markdown string
        """
        lines = [
            f"# Benchmark Report: {suite.name}",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n**Description:** {suite.description}",
            f"\n**Total Cases:** {len(suite.test_cases)}",
            "\n---\n",
        ]
        
        # Aggregate metrics
        evaluated_cases = [tc for tc in suite.test_cases if tc.evaluation]
        if evaluated_cases:
            results = [tc.evaluation for tc in evaluated_cases]
            aggregated = self.evaluator.aggregate_results(results)
            
            lines.append("## Overall Metrics\n")
            lines.append("| Metric | Mean | Min | Max | Count |")
            lines.append("|--------|------|-----|-----|-------|")
            
            for metric, stats in aggregated.items():
                lines.append(
                    f"| {metric} | {stats['mean']:.3f} | {stats['min']:.3f} | "
                    f"{stats['max']:.3f} | {stats['count']} |"
                )
            
            lines.append("\n---\n")
        
        # Latency stats
        latencies = [tc.latency_ms for tc in suite.test_cases if tc.latency_ms]
        if latencies:
            lines.append("## Latency Statistics\n")
            lines.append(f"- **Mean:** {sum(latencies)/len(latencies):.0f}ms")
            lines.append(f"- **Min:** {min(latencies):.0f}ms")
            lines.append(f"- **Max:** {max(latencies):.0f}ms")
            lines.append("\n---\n")
        
        # Results by category
        categories = set(tc.category for tc in suite.test_cases)
        
        for category in sorted(categories):
            cases = suite.filter_by_category(category)
            lines.append(f"## Category: {category}\n")
            
            for tc in cases:
                query_short = tc.query[:60] + "..." if len(tc.query) > 60 else tc.query
                lines.append(f"### Query: {query_short}\n")
                
                if tc.evaluation:
                    lines.append(f"- **Overall Score:** {tc.evaluation.overall_score:.3f}")
                    if tc.evaluation.faithfulness is not None:
                        lines.append(f"- **Faithfulness:** {tc.evaluation.faithfulness:.3f}")
                    if tc.evaluation.answer_relevancy is not None:
                        lines.append(f"- **Answer Relevancy:** {tc.evaluation.answer_relevancy:.3f}")
                    if tc.evaluation.context_precision is not None:
                        lines.append(f"- **Context Precision:** {tc.evaluation.context_precision:.3f}")
                
                if tc.latency_ms:
                    lines.append(f"- **Latency:** {tc.latency_ms:.0f}ms")
                
                if tc.answer:
                    answer_preview = tc.answer[:200] + "..." if len(tc.answer) > 200 else tc.answer
                    lines.append(f"\n**Answer Preview:**\n> {answer_preview}\n")
                
                lines.append("")
        
        report = "\n".join(lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report
    
    def compare_runs(
        self,
        baseline: BenchmarkSuite,
        current: BenchmarkSuite,
    ) -> dict:
        """
        Compare two benchmark runs.
        
        Args:
            baseline: Previous benchmark results
            current: New benchmark results
            
        Returns:
            Comparison dictionary with deltas
        """
        def get_scores(suite: BenchmarkSuite) -> dict:
            results = [tc.evaluation for tc in suite.test_cases if tc.evaluation]
            if not results:
                return {}
            return self.evaluator.aggregate_results(results)
        
        baseline_scores = get_scores(baseline)
        current_scores = get_scores(current)
        
        comparison = {
            "baseline_name": baseline.name,
            "current_name": current.name,
            "metrics": {},
        }
        
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "overall"]:
            if metric in baseline_scores and metric in current_scores:
                baseline_mean = baseline_scores[metric]["mean"]
                current_mean = current_scores[metric]["mean"]
                delta = current_mean - baseline_mean
                delta_pct = (delta / baseline_mean * 100) if baseline_mean > 0 else 0
                
                comparison["metrics"][metric] = {
                    "baseline": baseline_mean,
                    "current": current_mean,
                    "delta": delta,
                    "delta_percent": delta_pct,
                    "improved": delta > 0,
                }
        
        return comparison
