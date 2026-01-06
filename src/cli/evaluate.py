#!/usr/bin/env python3
# src/cli/evaluate.py
"""
CLI for running RAG evaluation benchmarks.

Usage:
    # Run default benchmark suite
    python -m src.cli.evaluate --suite default
    
    # Run custom benchmark file
    python -m src.cli.evaluate --suite benchmarks/custom.json
    
    # Quick evaluation of a single query
    python -m src.cli.evaluate --query "What is quantum entanglement?"
    
    # Compare with baseline
    python -m src.cli.evaluate --suite default --baseline results/baseline.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.metrics import RAGASEvaluator, EvaluationConfig
from src.evaluation.benchmark import BenchmarkSuite, BenchmarkRunner, TestCase

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def create_rag_function(use_rerank: bool = True, rerank_preset: str = "balanced"):
    """
    Create a RAG function wrapper for benchmarking.
    
    Returns a function that takes a query and returns dict with answer/contexts.
    """
    from src.cli.ask_library import RAGPipeline
    
    pipeline = RAGPipeline(
        use_reranker=use_rerank,
        reranker_preset=rerank_preset,
    )
    
    def rag_func(query: str) -> dict:
        """Run RAG pipeline and return results."""
        result = pipeline.query(query)
        
        # Extract contexts from result
        contexts = []
        sources = []
        
        if hasattr(result, 'contexts'):
            contexts = [ctx.text if hasattr(ctx, 'text') else str(ctx) for ctx in result.contexts]
        elif isinstance(result, dict):
            contexts = result.get('contexts', [])
            
        if hasattr(result, 'sources'):
            sources = result.sources
        elif isinstance(result, dict):
            sources = result.get('sources', [])
        
        answer = ""
        if hasattr(result, 'answer'):
            answer = result.answer
        elif isinstance(result, dict):
            answer = result.get('answer', '')
        elif isinstance(result, str):
            answer = result
        
        return {
            "answer": answer,
            "contexts": contexts,
            "sources": sources,
        }
    
    return rag_func


def run_single_evaluation(
    query: str,
    evaluator: RAGASEvaluator,
    rag_func,
    ground_truth: str = None,
) -> dict:
    """Evaluate a single query."""
    import time
    
    print(f"\n📝 Query: {query}")
    print("-" * 50)
    
    # Run RAG
    start = time.time()
    result = rag_func(query)
    latency = (time.time() - start) * 1000
    
    print(f"⏱️  Latency: {latency:.0f}ms")
    print(f"📄 Contexts retrieved: {len(result['contexts'])}")
    
    # Preview answer
    answer = result["answer"]
    preview = answer[:300] + "..." if len(answer) > 300 else answer
    print(f"\n💬 Answer preview:\n{preview}")
    
    # Evaluate
    print("\n🔍 Evaluating...")
    eval_result = evaluator.evaluate(
        query=query,
        answer=answer,
        contexts=result["contexts"],
        ground_truth=ground_truth,
    )
    
    # Print scores
    print("\n📊 Scores:")
    if eval_result.faithfulness is not None:
        print(f"  • Faithfulness:      {eval_result.faithfulness:.3f}")
    if eval_result.answer_relevancy is not None:
        print(f"  • Answer Relevancy:  {eval_result.answer_relevancy:.3f}")
    if eval_result.context_precision is not None:
        print(f"  • Context Precision: {eval_result.context_precision:.3f}")
    if eval_result.context_recall is not None:
        print(f"  • Context Recall:    {eval_result.context_recall:.3f}")
    print(f"  • Overall:           {eval_result.overall_score:.3f}")
    
    # Print explanations
    if eval_result.explanations:
        print("\n📝 Explanations:")
        for metric, explanation in eval_result.explanations.items():
            print(f"  {metric}: {explanation[:100]}...")
    
    return eval_result.to_dict()


def run_benchmark_suite(
    suite: BenchmarkSuite,
    runner: BenchmarkRunner,
    output_dir: Path,
) -> BenchmarkSuite:
    """Run a full benchmark suite."""
    print(f"\n🚀 Running benchmark: {suite.name}")
    print(f"📋 Test cases: {len(suite.test_cases)}")
    print("-" * 50)
    
    # Run suite
    completed_suite = runner.run_suite(suite)
    
    # Generate report
    report_path = output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report = runner.generate_report(completed_suite, report_path)
    
    # Save results JSON
    results_path = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    completed_suite.save_to_json(results_path)
    
    print(f"\n✅ Benchmark complete!")
    print(f"📄 Report: {report_path}")
    print(f"📊 Results: {results_path}")
    
    # Print summary
    evaluated = [tc for tc in completed_suite.test_cases if tc.evaluation]
    if evaluated:
        results = [tc.evaluation for tc in evaluated]
        aggregated = runner.evaluator.aggregate_results(results)
        
        print("\n📊 Summary Metrics:")
        for metric, stats in aggregated.items():
            print(f"  • {metric}: {stats['mean']:.3f} (min: {stats['min']:.3f}, max: {stats['max']:.3f})")
    
    return completed_suite


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline quality using RAGAS metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default benchmark
  python -m src.cli.evaluate --suite default
  
  # Evaluate single query
  python -m src.cli.evaluate --query "What is quantum entanglement?"
  
  # Use custom benchmark file
  python -m src.cli.evaluate --suite benchmarks/my_tests.json
  
  # Disable reranking for comparison
  python -m src.cli.evaluate --suite default --no-rerank
        """,
    )
    
    # Query options
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query to evaluate",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        help="Expected answer for context recall (with --query)",
    )
    
    # Suite options
    parser.add_argument(
        "--suite", "-s",
        type=str,
        help="Benchmark suite: 'default' or path to JSON file",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Path to baseline results JSON for comparison",
    )
    
    # RAG options
    parser.add_argument(
        "--rerank",
        action="store_true",
        default=True,
        help="Enable re-ranking (default: enabled)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable re-ranking",
    )
    parser.add_argument(
        "--rerank-preset",
        type=str,
        default="balanced",
        choices=["fast", "balanced", "quality", "max_quality"],
        help="Re-ranker preset (default: balanced)",
    )
    
    # Evaluation options
    parser.add_argument(
        "--eval-model",
        type=str,
        default="gpt-4o-mini",
        help="Model for evaluation (default: gpt-4o-mini)",
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="benchmark_results",
        help="Output directory for reports (default: benchmark_results)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Validate args
    if not args.query and not args.suite:
        parser.error("Either --query or --suite is required")
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    use_rerank = args.rerank and not args.no_rerank
    
    # Create evaluator
    config = EvaluationConfig(eval_model=args.eval_model)
    evaluator = RAGASEvaluator(config=config)
    
    # Create RAG function
    print("🔧 Initializing RAG pipeline...")
    rag_func = create_rag_function(
        use_rerank=use_rerank,
        rerank_preset=args.rerank_preset,
    )
    
    if args.query:
        # Single query evaluation
        result = run_single_evaluation(
            query=args.query,
            evaluator=evaluator,
            rag_func=rag_func,
            ground_truth=args.ground_truth,
        )
        
        # Save result
        result_path = output_dir / f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n📄 Result saved: {result_path}")
        
    elif args.suite:
        # Benchmark suite
        if args.suite == "default":
            suite = BenchmarkSuite.default_quantum_suite()
        else:
            suite_path = Path(args.suite)
            if not suite_path.exists():
                print(f"❌ Suite file not found: {suite_path}")
                sys.exit(1)
            suite = BenchmarkSuite.load_from_json(suite_path)
        
        # Create runner
        runner = BenchmarkRunner(rag_func=rag_func, evaluator=evaluator)
        
        # Run benchmark
        completed = run_benchmark_suite(suite, runner, output_dir)
        
        # Compare with baseline if provided
        if args.baseline:
            baseline_path = Path(args.baseline)
            if baseline_path.exists():
                baseline = BenchmarkSuite.load_from_json(baseline_path)
                comparison = runner.compare_runs(baseline, completed)
                
                print("\n📈 Comparison with baseline:")
                for metric, data in comparison["metrics"].items():
                    symbol = "✅" if data["improved"] else "❌"
                    print(f"  {symbol} {metric}: {data['baseline']:.3f} → {data['current']:.3f} "
                          f"({data['delta_percent']:+.1f}%)")
            else:
                print(f"⚠️  Baseline file not found: {baseline_path}")


if __name__ == "__main__":
    main()
