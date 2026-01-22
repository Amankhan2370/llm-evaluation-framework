"""
Main entry point for LLM evaluation framework.
"""
import asyncio
import logging
from pathlib import Path
from config.settings import settings
from evaluation.hallucination import HallucinationDetector
from evaluation.grounding import GroundingScorer
from evaluation.adversarial import AdversarialTester
from evaluation.runner import EvaluationRunner
from pipelines.batch_eval import BatchEvaluationPipeline

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Main evaluation function."""
    logger.info("Starting LLM Evaluation Framework")
    
    # Initialize components
    hallucination_detector = HallucinationDetector(
        hallucination_threshold=settings.hallucination_threshold,
        citation_required=settings.citation_required,
        unsupported_claim_detection=settings.unsupported_claim_detection
    )
    
    grounding_scorer = GroundingScorer(
        context_overlap_threshold=settings.context_overlap_threshold,
        retrieval_confidence_threshold=settings.retrieval_confidence_threshold
    )
    
    adversarial_tester = None
    if settings.adversarial_enabled:
        adversarial_tester = AdversarialTester(
            num_variants=settings.adversarial_variants
        )
    
    # Create evaluation runner
    runner = EvaluationRunner(
        hallucination_detector=hallucination_detector,
        grounding_scorer=grounding_scorer,
        adversarial_tester=adversarial_tester
    )
    
    # Create batch pipeline
    pipeline = BatchEvaluationPipeline(runner)
    
    # Run evaluation
    output_path = Path(settings.reports_dir) / "evaluation_results.json"
    
    try:
        summary = await pipeline.run(
            dataset_path=settings.eval_dataset_path,
            output_path=str(output_path),
            batch_size=settings.batch_size,
            max_concurrent=settings.max_concurrent_evals
        )
        
        logger.info("Evaluation completed successfully")
        logger.info(f"Summary: {summary}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
