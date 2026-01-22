"""
Aggregated metrics and scoring calculations.
"""
from typing import List, Dict, Any
import statistics
import logging

logger = logging.getLogger(__name__)


class MetricsAggregator:
    """
    Aggregates evaluation results into summary metrics.
    """
    
    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate evaluation results into summary statistics.
        
        Args:
            results: List of evaluation result dictionaries
            
        Returns:
            Aggregated metrics dictionary
        """
        if not results:
            return {}
        
        # Extract scores
        hallucination_scores = [
            r.get("hallucination", {}).get("hallucination_score", 0.0)
            for r in results
        ]
        
        grounding_scores = [
            r.get("grounding", {}).get("grounding_score", 0.0)
            for r in results
            if r.get("grounding")
        ]
        
        overall_scores = [r.get("overall_score", 0.0) for r in results]
        
        # Calculate statistics
        metrics = {
            "total_evaluations": len(results),
            "hallucination": self._calculate_stats(hallucination_scores, "hallucination"),
            "grounding": self._calculate_stats(grounding_scores, "grounding") if grounding_scores else {},
            "overall": self._calculate_stats(overall_scores, "overall"),
            "hallucination_rate": self._calculate_rate(
                hallucination_scores,
                threshold=0.7
            ),
            "pass_rate": self._calculate_rate(
                overall_scores,
                threshold=0.6,
                higher_is_better=True
            )
        }
        
        return metrics
    
    def _calculate_stats(self, scores: List[float], metric_name: str) -> Dict[str, float]:
        """Calculate statistical measures for scores."""
        if not scores:
            return {}
        
        return {
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
            "p25": self._percentile(scores, 25),
            "p75": self._percentile(scores, 75),
            "p95": self._percentile(scores, 95),
            "p99": self._percentile(scores, 99)
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _calculate_rate(
        self,
        scores: List[float],
        threshold: float,
        higher_is_better: bool = False
    ) -> float:
        """Calculate rate of scores above/below threshold."""
        if not scores:
            return 0.0
        
        if higher_is_better:
            count = sum(1 for s in scores if s >= threshold)
        else:
            count = sum(1 for s in scores if s <= threshold)
        
        return count / len(scores)
