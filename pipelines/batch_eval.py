"""
Batch evaluation pipeline for processing datasets.
"""
import json
import csv
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from evaluation.runner import EvaluationRunner
from metrics.scoring import MetricsAggregator
import logging

logger = logging.getLogger(__name__)


class BatchEvaluationPipeline:
    """
    Pipeline for batch evaluation of datasets.
    """
    
    def __init__(self, evaluation_runner: EvaluationRunner):
        self.runner = evaluation_runner
        self.aggregator = MetricsAggregator()
    
    async def run(
        self,
        dataset_path: str,
        output_path: str,
        batch_size: int = 10,
        max_concurrent: int = 5
    ) -> Dict[str, Any]:
        """
        Run batch evaluation on a dataset.
        
        Args:
            dataset_path: Path to dataset file (JSON or CSV)
            output_path: Path to save results
            batch_size: Batch size for processing
            max_concurrent: Max concurrent evaluations
            
        Returns:
            Summary of evaluation results
        """
        # Load dataset
        examples = self._load_dataset(dataset_path)
        logger.info(f"Loaded {len(examples)} examples from {dataset_path}")
        
        # Process in batches
        all_results = []
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1} ({len(batch)} examples)")
            
            batch_results = await self.runner.evaluate_batch(
                batch,
                max_concurrent=max_concurrent
            )
            all_results.extend(batch_results)
        
        # Aggregate metrics
        summary = self.aggregator.aggregate(all_results)
        
        # Save results
        output = {
            "summary": summary,
            "results": all_results,
            "total_examples": len(examples),
            "evaluated": len(all_results)
        }
        
        self._save_results(output, output_path)
        
        return summary
    
    def _load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load dataset from JSON or CSV."""
        path = Path(dataset_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
                # Handle different JSON structures
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'examples' in data:
                    return data['examples']
                else:
                    raise ValueError("Invalid JSON structure")
        
        elif path.suffix == '.csv':
            examples = []
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    examples.append(row)
            return examples
        
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
