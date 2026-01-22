"""
Adversarial testing and stress-test prompt generation.
Creates variants to test robustness and failure modes.
"""
from typing import List, Dict, Any
import random
import logging

logger = logging.getLogger(__name__)


class AdversarialTester:
    """
    Generates adversarial prompts and stress tests.
    """
    
    def __init__(self, num_variants: int = 5):
        self.num_variants = num_variants
    
    def generate_variants(self, original_prompt: str) -> List[Dict[str, str]]:
        """
        Generate adversarial variants of a prompt.
        
        Returns:
            List of variant dictionaries with 'prompt' and 'variant_type'
        """
        variants = []
        
        # Variant 1: Leading questions
        variants.append({
            "prompt": f"Based on the following, confirm that: {original_prompt}",
            "variant_type": "leading_question"
        })
        
        # Variant 2: Negative framing
        variants.append({
            "prompt": f"Explain why the following is NOT true: {original_prompt}",
            "variant_type": "negative_framing"
        })
        
        # Variant 3: Ambiguous phrasing
        variants.append({
            "prompt": f"Can you tell me something about {original_prompt}?",
            "variant_type": "ambiguous"
        })
        
        # Variant 4: Overly specific request
        variants.append({
            "prompt": f"Provide exact details, numbers, and specific facts about: {original_prompt}",
            "variant_type": "overly_specific"
        })
        
        # Variant 5: Contradictory instruction
        variants.append({
            "prompt": f"{original_prompt} But also consider the opposite perspective.",
            "variant_type": "contradictory"
        })
        
        # Return requested number of variants
        return variants[:self.num_variants]
    
    def generate_stress_tests(self, base_prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Generate stress test scenarios.
        
        Returns:
            List of stress test dictionaries
        """
        stress_tests = []
        
        # Test 1: Empty/null prompt
        stress_tests.append({
            "prompt": "",
            "test_type": "empty_prompt",
            "expected_behavior": "should handle gracefully"
        })
        
        # Test 2: Very long prompt
        long_prompt = " ".join(["test"] * 1000)
        stress_tests.append({
            "prompt": long_prompt,
            "test_type": "long_prompt",
            "expected_behavior": "should truncate or handle appropriately"
        })
        
        # Test 3: Special characters
        stress_tests.append({
            "prompt": "What is the meaning of @#$%^&*()?",
            "test_type": "special_characters",
            "expected_behavior": "should process or reject gracefully"
        })
        
        # Test 4: Multiple questions
        stress_tests.append({
            "prompt": "What is X? What is Y? What is Z? Explain all three.",
            "test_type": "multiple_questions",
            "expected_behavior": "should address all questions"
        })
        
        return stress_tests
    
    def analyze_failures(
        self,
        original_result: Dict[str, Any],
        variant_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze failures across adversarial variants.
        
        Returns:
            Analysis of failure patterns
        """
        analysis = {
            "total_variants": len(variant_results),
            "failed_variants": 0,
            "failure_types": {},
            "robustness_score": 0.0
        }
        
        original_hallucination = original_result.get("hallucination_score", 0.0)
        
        for variant_result in variant_results:
            variant_hallucination = variant_result.get("hallucination_score", 0.0)
            variant_type = variant_result.get("variant_type", "unknown")
            
            # Consider failed if hallucination score increased significantly
            if variant_hallucination > original_hallucination + 0.2:
                analysis["failed_variants"] += 1
                if variant_type not in analysis["failure_types"]:
                    analysis["failure_types"][variant_type] = 0
                analysis["failure_types"][variant_type] += 1
        
        # Robustness score: percentage of variants that didn't fail
        if analysis["total_variants"] > 0:
            analysis["robustness_score"] = (
                1.0 - (analysis["failed_variants"] / analysis["total_variants"])
            )
        
        return analysis
