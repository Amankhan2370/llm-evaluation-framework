"""
Tests for evaluation framework.
"""
import pytest
from evaluation.hallucination import HallucinationDetector
from evaluation.grounding import GroundingScorer
from evaluation.adversarial import AdversarialTester
from evaluation.runner import EvaluationRunner
from metrics.scoring import MetricsAggregator


def test_hallucination_detection():
    """Test hallucination detection."""
    detector = HallucinationDetector(
        hallucination_threshold=0.7,
        citation_required=False
    )
    
    answer = "The capital of France is Paris."
    context = "Paris is the capital and largest city of France."
    
    result = detector.detect(answer, context)
    
    assert "hallucination_score" in result
    assert "is_hallucination" in result
    assert "checks" in result
    assert isinstance(result["hallucination_score"], float)


def test_grounding_scoring():
    """Test grounding scoring."""
    scorer = GroundingScorer()
    
    answer = "Paris is the capital of France."
    context = "Paris is the capital and largest city of France."
    
    result = scorer.score(answer, context)
    
    assert "grounding_score" in result
    assert "faithfulness_score" in result
    assert "metrics" in result
    assert isinstance(result["grounding_score"], float)


def test_adversarial_variants():
    """Test adversarial variant generation."""
    tester = AdversarialTester(num_variants=3)
    
    original = "What is the capital of France?"
    variants = tester.generate_variants(original)
    
    assert len(variants) == 3
    assert all("prompt" in v for v in variants)
    assert all("variant_type" in v for v in variants)


def test_metrics_aggregation():
    """Test metrics aggregation."""
    aggregator = MetricsAggregator()
    
    results = [
        {
            "hallucination": {"hallucination_score": 0.3},
            "grounding": {"grounding_score": 0.8},
            "overall_score": 0.75
        },
        {
            "hallucination": {"hallucination_score": 0.5},
            "grounding": {"grounding_score": 0.7},
            "overall_score": 0.65
        }
    ]
    
    summary = aggregator.aggregate(results)
    
    assert "total_evaluations" in summary
    assert summary["total_evaluations"] == 2
    assert "hallucination" in summary
    assert "overall" in summary
