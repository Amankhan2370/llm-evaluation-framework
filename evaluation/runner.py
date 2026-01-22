"""
Main evaluation runner that orchestrates the evaluation pipeline.
"""
import asyncio
import time
from typing import List, Dict, Any, Optional
from evaluation.hallucination import HallucinationDetector
from evaluation.grounding import GroundingScorer
from evaluation.adversarial import AdversarialTester
import logging

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """
    Orchestrates end-to-end evaluation of LLM outputs.
    """
    
    def __init__(
        self,
        hallucination_detector: HallucinationDetector,
        grounding_scorer: GroundingScorer,
        adversarial_tester: Optional[AdversarialTester] = None
    ):
        self.hallucination_detector = hallucination_detector
        self.grounding_scorer = grounding_scorer
        self.adversarial_tester = adversarial_tester
    
    async def evaluate(
        self,
        prompt: str,
        answer: str,
        context: Optional[str] = None,
        citations: Optional[List[str]] = None,
        retrieval_scores: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Run complete evaluation on a single example.
        
        Returns:
            Complete evaluation results
        """
        start_time = time.time()
        
        results = {
            "prompt": prompt,
            "answer": answer,
            "timestamp": time.time(),
            "evaluation_time_ms": 0
        }
        
        # Hallucination detection
        hallucination_results = self.hallucination_detector.detect(
            answer=answer,
            context=context,
            citations=citations,
            retrieval_scores=retrieval_scores
        )
        results["hallucination"] = hallucination_results
        
        # Grounding scoring
        if context:
            grounding_results = self.grounding_scorer.score(
                answer=answer,
                context=context,
                retrieval_scores=retrieval_scores,
                citations=citations
            )
            results["grounding"] = grounding_results
        
        # Overall score
        results["overall_score"] = self._calculate_overall_score(
            hallucination_results,
            grounding_results if context else None
        )
        
        results["evaluation_time_ms"] = (time.time() - start_time) * 1000
        
        return results
    
    async def evaluate_batch(
        self,
        examples: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of examples with concurrency control.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_with_semaphore(example):
            async with semaphore:
                return await self.evaluate(
                    prompt=example.get("prompt", ""),
                    answer=example.get("answer", ""),
                    context=example.get("context"),
                    citations=example.get("citations"),
                    retrieval_scores=example.get("retrieval_scores")
                )
        
        tasks = [evaluate_with_semaphore(ex) for ex in examples]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def evaluate_with_adversarial(
        self,
        prompt: str,
        answer: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate with adversarial testing.
        """
        if not self.adversarial_tester:
            return await self.evaluate(prompt, answer, context)
        
        # Original evaluation
        original_result = await self.evaluate(prompt, answer, context)
        
        # Generate variants
        variants = self.adversarial_tester.generate_variants(prompt)
        
        # Evaluate variants (simplified - in production, would call LLM for each)
        variant_results = []
        for variant in variants:
            # In real scenario, would generate answer for variant prompt
            variant_result = await self.evaluate(
                prompt=variant["prompt"],
                answer=answer,  # Using same answer for testing
                context=context
            )
            variant_result["variant_type"] = variant["variant_type"]
            variant_results.append(variant_result)
        
        # Analyze failures
        failure_analysis = self.adversarial_tester.analyze_failures(
            original_result,
            variant_results
        )
        
        original_result["adversarial_testing"] = {
            "variants_tested": len(variants),
            "failure_analysis": failure_analysis
        }
        
        return original_result
    
    def _calculate_overall_score(
        self,
        hallucination_results: Dict[str, Any],
        grounding_results: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate overall evaluation score.
        """
        # Hallucination score (lower is better, so invert)
        hallucination_penalty = hallucination_results.get("hallucination_score", 0.0)
        base_score = 1.0 - hallucination_penalty
        
        # Grounding bonus (if available)
        if grounding_results:
            grounding_score = grounding_results.get("grounding_score", 0.0)
            # Weighted combination
            overall = base_score * 0.6 + grounding_score * 0.4
        else:
            overall = base_score
        
        return max(0.0, min(1.0, overall))
