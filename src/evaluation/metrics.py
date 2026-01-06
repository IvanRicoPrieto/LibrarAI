# src/evaluation/metrics.py
"""
RAGAS-style evaluation metrics for RAG quality assessment.

Implements four core metrics:
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevancy: Does the answer address the question?
- Context Precision: Are retrieved chunks actually relevant?
- Context Recall: Does the context cover the expected answer?
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Available evaluation metrics."""
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"


@dataclass
class EvaluationConfig:
    """Configuration for RAG evaluation."""
    
    # LLM for evaluation (uses same as generation by default)
    eval_model: str = "gpt-4o-mini"
    
    # Which metrics to compute
    metrics: list[MetricType] = field(default_factory=lambda: [
        MetricType.FAITHFULNESS,
        MetricType.ANSWER_RELEVANCY,
        MetricType.CONTEXT_PRECISION,
    ])
    
    # Temperature for evaluation LLM (low for consistency)
    temperature: float = 0.0
    
    # Number of samples for statistical significance
    num_samples: int = 1
    
    # Timeout per evaluation
    timeout: float = 60.0


@dataclass 
class EvaluationResult:
    """Result of evaluating a single RAG response."""
    
    # The query that was evaluated
    query: str
    
    # The generated answer
    answer: str
    
    # Retrieved contexts (list of chunk texts)
    contexts: list[str]
    
    # Ground truth answer (optional, for recall)
    ground_truth: Optional[str] = None
    
    # Metric scores (0.0 to 1.0)
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    
    # Detailed explanations from LLM
    explanations: dict[str, str] = field(default_factory=dict)
    
    # Error messages if evaluation failed
    errors: dict[str, str] = field(default_factory=dict)
    
    @property
    def overall_score(self) -> float:
        """Compute weighted average of available metrics."""
        scores = []
        weights = []
        
        if self.faithfulness is not None:
            scores.append(self.faithfulness)
            weights.append(2.0)  # Faithfulness is critical
            
        if self.answer_relevancy is not None:
            scores.append(self.answer_relevancy)
            weights.append(1.5)
            
        if self.context_precision is not None:
            scores.append(self.context_precision)
            weights.append(1.0)
            
        if self.context_recall is not None:
            scores.append(self.context_recall)
            weights.append(1.0)
        
        if not scores:
            return 0.0
            
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "answer": self.answer[:500] + "..." if len(self.answer) > 500 else self.answer,
            "num_contexts": len(self.contexts),
            "ground_truth": self.ground_truth,
            "scores": {
                "faithfulness": self.faithfulness,
                "answer_relevancy": self.answer_relevancy,
                "context_precision": self.context_precision,
                "context_recall": self.context_recall,
                "overall": self.overall_score,
            },
            "explanations": self.explanations,
            "errors": self.errors,
        }


class RAGASEvaluator:
    """
    RAGAS-style evaluator for RAG quality.
    
    Uses an LLM to assess the quality of RAG responses across multiple dimensions.
    Designed to be used during development to measure impact of changes.
    
    Usage:
        evaluator = RAGASEvaluator()
        result = await evaluator.evaluate(
            query="What is quantum entanglement?",
            answer="Quantum entanglement is...",
            contexts=["Context chunk 1...", "Context chunk 2..."],
        )
        print(f"Faithfulness: {result.faithfulness:.2f}")
    """
    
    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        llm_client: Optional[object] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            config: Evaluation configuration
            llm_client: Optional pre-configured LLM client
        """
        self.config = config or EvaluationConfig()
        self._llm_client = llm_client
        self._initialized = False
    
    def _get_llm_client(self):
        """Lazy initialization of LLM client."""
        if self._llm_client is not None:
            return self._llm_client
            
        if not self._initialized:
            try:
                from openai import OpenAI
                self._llm_client = OpenAI()
                self._initialized = True
            except ImportError:
                raise ImportError(
                    "openai package required for evaluation. "
                    "Install with: pip install openai"
                )
        return self._llm_client
    
    def _call_llm(self, prompt: str, system: str = "") -> str:
        """Call LLM for evaluation."""
        client = self._get_llm_client()
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=self.config.eval_model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=1000,
        )
        
        return response.choices[0].message.content
    
    def _evaluate_faithfulness(
        self,
        answer: str,
        contexts: list[str],
    ) -> tuple[float, str]:
        """
        Evaluate if the answer is grounded in the provided contexts.
        
        Returns:
            Tuple of (score, explanation)
        """
        context_text = "\n\n---\n\n".join(contexts[:5])  # Limit context length
        
        prompt = f"""Evaluate the faithfulness of the following answer based on the provided context.
Faithfulness measures whether ALL claims in the answer can be inferred from the context.

CONTEXT:
{context_text}

ANSWER:
{answer}

EVALUATION CRITERIA:
- Score 1.0: Every claim in the answer is directly supported by the context
- Score 0.7-0.9: Most claims are supported, minor extrapolations
- Score 0.4-0.6: Mix of supported and unsupported claims
- Score 0.1-0.3: Most claims are not in the context
- Score 0.0: Answer contradicts or is completely unrelated to context

Respond in JSON format:
{{"score": <float 0.0-1.0>, "explanation": "<brief explanation>"}}"""

        try:
            response = self._call_llm(prompt)
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                return float(data["score"]), data.get("explanation", "")
        except Exception as e:
            logger.warning(f"Faithfulness evaluation failed: {e}")
            return None, str(e)
        
        return None, "Failed to parse LLM response"
    
    def _evaluate_answer_relevancy(
        self,
        query: str,
        answer: str,
    ) -> tuple[float, str]:
        """
        Evaluate if the answer addresses the question.
        
        Returns:
            Tuple of (score, explanation)
        """
        prompt = f"""Evaluate how well the answer addresses the question.
Answer relevancy measures whether the response is actually helpful for the user's query.

QUESTION:
{query}

ANSWER:
{answer}

EVALUATION CRITERIA:
- Score 1.0: Answer directly and completely addresses the question
- Score 0.7-0.9: Answer mostly addresses the question with minor gaps
- Score 0.4-0.6: Answer partially relevant but misses key aspects
- Score 0.1-0.3: Answer tangentially related but doesn't help
- Score 0.0: Answer is completely off-topic

Respond in JSON format:
{{"score": <float 0.0-1.0>, "explanation": "<brief explanation>"}}"""

        try:
            response = self._call_llm(prompt)
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                return float(data["score"]), data.get("explanation", "")
        except Exception as e:
            logger.warning(f"Answer relevancy evaluation failed: {e}")
            return None, str(e)
        
        return None, "Failed to parse LLM response"
    
    def _evaluate_context_precision(
        self,
        query: str,
        contexts: list[str],
    ) -> tuple[float, str]:
        """
        Evaluate if retrieved contexts are relevant to the query.
        
        Returns:
            Tuple of (score, explanation)
        """
        # Evaluate each context chunk
        relevant_count = 0
        evaluations = []
        
        for i, ctx in enumerate(contexts[:10], 1):  # Limit to 10 chunks
            ctx_preview = ctx[:500] + "..." if len(ctx) > 500 else ctx
            
            prompt = f"""Is this context chunk relevant to answering the query?

QUERY: {query}

CONTEXT CHUNK {i}:
{ctx_preview}

Answer with just "RELEVANT" or "NOT_RELEVANT" followed by a brief reason."""

            try:
                response = self._call_llm(prompt)
                is_relevant = "RELEVANT" in response.upper() and "NOT_RELEVANT" not in response.upper()
                if is_relevant:
                    relevant_count += 1
                evaluations.append(f"Chunk {i}: {'Relevant' if is_relevant else 'Not relevant'}")
            except Exception as e:
                evaluations.append(f"Chunk {i}: Error - {e}")
        
        if not contexts:
            return 0.0, "No contexts provided"
        
        score = relevant_count / len(contexts[:10])
        explanation = f"{relevant_count}/{len(contexts[:10])} chunks relevant. " + "; ".join(evaluations[:3])
        
        return score, explanation
    
    def _evaluate_context_recall(
        self,
        contexts: list[str],
        ground_truth: str,
    ) -> tuple[float, str]:
        """
        Evaluate if contexts contain information needed for the ground truth answer.
        
        Returns:
            Tuple of (score, explanation)
        """
        context_text = "\n\n---\n\n".join(contexts[:5])
        
        prompt = f"""Evaluate whether the provided contexts contain the information needed to produce the ground truth answer.

CONTEXTS:
{context_text}

GROUND TRUTH ANSWER:
{ground_truth}

EVALUATION CRITERIA:
- Score 1.0: Contexts contain all information needed for the ground truth
- Score 0.7-0.9: Contexts contain most of the needed information
- Score 0.4-0.6: Contexts contain some relevant information
- Score 0.1-0.3: Contexts contain minimal relevant information
- Score 0.0: Contexts don't contain any relevant information

Respond in JSON format:
{{"score": <float 0.0-1.0>, "explanation": "<brief explanation>"}}"""

        try:
            response = self._call_llm(prompt)
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                return float(data["score"]), data.get("explanation", "")
        except Exception as e:
            logger.warning(f"Context recall evaluation failed: {e}")
            return None, str(e)
        
        return None, "Failed to parse LLM response"
    
    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate a RAG response across all configured metrics.
        
        Args:
            query: The user query
            answer: The generated answer
            contexts: List of retrieved context chunks
            ground_truth: Optional expected answer for recall calculation
            
        Returns:
            EvaluationResult with scores and explanations
        """
        result = EvaluationResult(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )
        
        # Evaluate each configured metric
        for metric in self.config.metrics:
            try:
                if metric == MetricType.FAITHFULNESS:
                    score, explanation = self._evaluate_faithfulness(answer, contexts)
                    result.faithfulness = score
                    if explanation:
                        result.explanations["faithfulness"] = explanation
                        
                elif metric == MetricType.ANSWER_RELEVANCY:
                    score, explanation = self._evaluate_answer_relevancy(query, answer)
                    result.answer_relevancy = score
                    if explanation:
                        result.explanations["answer_relevancy"] = explanation
                        
                elif metric == MetricType.CONTEXT_PRECISION:
                    score, explanation = self._evaluate_context_precision(query, contexts)
                    result.context_precision = score
                    if explanation:
                        result.explanations["context_precision"] = explanation
                        
                elif metric == MetricType.CONTEXT_RECALL:
                    if ground_truth:
                        score, explanation = self._evaluate_context_recall(contexts, ground_truth)
                        result.context_recall = score
                        if explanation:
                            result.explanations["context_recall"] = explanation
                    else:
                        result.errors["context_recall"] = "Ground truth required"
                        
            except Exception as e:
                result.errors[metric.value] = str(e)
                logger.error(f"Metric {metric.value} failed: {e}")
        
        return result
    
    def evaluate_batch(
        self,
        test_cases: list[dict],
    ) -> list[EvaluationResult]:
        """
        Evaluate multiple test cases.
        
        Args:
            test_cases: List of dicts with keys: query, answer, contexts, ground_truth (optional)
            
        Returns:
            List of EvaluationResult objects
        """
        results = []
        
        for i, case in enumerate(test_cases):
            logger.info(f"Evaluating case {i+1}/{len(test_cases)}: {case['query'][:50]}...")
            
            result = self.evaluate(
                query=case["query"],
                answer=case["answer"],
                contexts=case["contexts"],
                ground_truth=case.get("ground_truth"),
            )
            results.append(result)
        
        return results
    
    def aggregate_results(
        self,
        results: list[EvaluationResult],
    ) -> dict:
        """
        Aggregate metrics across multiple evaluations.
        
        Returns:
            Dictionary with mean, min, max for each metric
        """
        metrics = {
            "faithfulness": [],
            "answer_relevancy": [],
            "context_precision": [],
            "context_recall": [],
            "overall": [],
        }
        
        for r in results:
            if r.faithfulness is not None:
                metrics["faithfulness"].append(r.faithfulness)
            if r.answer_relevancy is not None:
                metrics["answer_relevancy"].append(r.answer_relevancy)
            if r.context_precision is not None:
                metrics["context_precision"].append(r.context_precision)
            if r.context_recall is not None:
                metrics["context_recall"].append(r.context_recall)
            metrics["overall"].append(r.overall_score)
        
        aggregated = {}
        for name, values in metrics.items():
            if values:
                aggregated[name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
        
        return aggregated
