"""
Grounding and faithfulness scoring for LLM outputs.
Measures how well answers are grounded in provided context.
"""
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GroundingScorer:
    """
    Scores grounding quality of answers against context.
    """
    
    def __init__(
        self,
        context_overlap_threshold: float = 0.5,
        retrieval_confidence_threshold: float = 0.7
    ):
        self.context_overlap_threshold = context_overlap_threshold
        self.retrieval_confidence_threshold = retrieval_confidence_threshold
        
        try:
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Failed to load similarity model: {str(e)}")
            self.similarity_model = None
    
    def score(
        self,
        answer: str,
        context: str,
        retrieval_scores: Optional[List[float]] = None,
        citations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Score grounding quality of answer.
        
        Args:
            answer: Generated answer
            context: Retrieved context
            retrieval_scores: Confidence scores for retrieved chunks
            citations: Citations in answer
            
        Returns:
            Dictionary with grounding scores and metrics
        """
        results = {
            "grounding_score": 0.0,
            "faithfulness_score": 0.0,
            "metrics": {},
            "explanation": []
        }
        
        # Metric 1: Answer-Context Alignment
        alignment_score = self._calculate_alignment(answer, context)
        results["metrics"]["answer_context_alignment"] = alignment_score
        results["grounding_score"] += alignment_score * 0.4
        
        # Metric 2: Context Coverage
        coverage_score = self._calculate_coverage(answer, context)
        results["metrics"]["context_coverage"] = coverage_score
        results["grounding_score"] += coverage_score * 0.3
        
        # Metric 3: Retrieval Quality (if available)
        if retrieval_scores:
            retrieval_score = self._score_retrieval_quality(retrieval_scores)
            results["metrics"]["retrieval_quality"] = retrieval_score
            results["grounding_score"] += retrieval_score * 0.2
        
        # Metric 4: Citation Quality
        if citations:
            citation_score = self._score_citation_quality(answer, citations, context)
            results["metrics"]["citation_quality"] = citation_score
            results["grounding_score"] += citation_score * 0.1
        
        # Normalize
        results["grounding_score"] = min(results["grounding_score"], 1.0)
        
        # Faithfulness is similar but focuses on factual accuracy
        results["faithfulness_score"] = alignment_score * 0.6 + coverage_score * 0.4
        
        # Generate explanation
        if results["grounding_score"] < self.context_overlap_threshold:
            results["explanation"].append(
                f"Low grounding score ({results['grounding_score']:.2f}). "
                f"Answer may not be well-grounded in provided context."
            )
        
        return results
    
    def _calculate_alignment(self, answer: str, context: str) -> float:
        """
        Calculate semantic alignment between answer and context.
        Returns score between 0 and 1.
        """
        if not self.similarity_model:
            # Word overlap fallback
            answer_words = set(answer.lower().split())
            context_words = set(context.lower().split())
            if not answer_words:
                return 0.0
            overlap = len(answer_words & context_words) / len(answer_words)
            return overlap
        
        try:
            answer_embedding = self.similarity_model.encode([answer], convert_to_numpy=True)[0]
            context_embedding = self.similarity_model.encode([context], convert_to_numpy=True)[0]
            
            similarity = np.dot(answer_embedding, context_embedding) / (
                np.linalg.norm(answer_embedding) * np.linalg.norm(context_embedding)
            )
            return float(max(0.0, similarity))  # Ensure non-negative
        except Exception as e:
            logger.error(f"Error calculating alignment: {str(e)}")
            return 0.0
    
    def _calculate_coverage(self, answer: str, context: str) -> float:
        """
        Calculate how much of the context is covered by the answer.
        Returns score between 0 and 1.
        """
        if not self.similarity_model:
            # Simple word-based coverage
            answer_words = set(answer.lower().split())
            context_words = set(context.lower().split())
            if not context_words:
                return 0.0
            coverage = len(answer_words & context_words) / len(context_words)
            return coverage
        
        try:
            # Split into chunks for better coverage calculation
            answer_chunks = self._chunk_text(answer, chunk_size=50)
            context_chunks = self._chunk_text(context, chunk_size=50)
            
            if not answer_chunks or not context_chunks:
                return 0.0
            
            answer_embeddings = self.similarity_model.encode(answer_chunks, convert_to_numpy=True)
            context_embeddings = self.similarity_model.encode(context_chunks, convert_to_numpy=True)
            
            # Calculate coverage: how many context chunks are covered by answer
            coverage_scores = []
            for ctx_emb in context_embeddings:
                max_sim = max([
                    np.dot(ctx_emb, ans_emb) / (
                        np.linalg.norm(ctx_emb) * np.linalg.norm(ans_emb)
                    )
                    for ans_emb in answer_embeddings
                ])
                coverage_scores.append(max_sim)
            
            avg_coverage = sum(coverage_scores) / len(coverage_scores)
            return float(max(0.0, avg_coverage))
        except Exception as e:
            logger.error(f"Error calculating coverage: {str(e)}")
            return 0.0
    
    def _score_retrieval_quality(self, retrieval_scores: List[float]) -> float:
        """
        Score quality of retrieval based on confidence scores.
        """
        if not retrieval_scores:
            return 0.0
        
        avg_score = sum(retrieval_scores) / len(retrieval_scores)
        min_score = min(retrieval_scores)
        
        # Weighted score: average (70%) and minimum (30%)
        quality_score = avg_score * 0.7 + min_score * 0.3
        
        # Normalize based on threshold
        if quality_score >= self.retrieval_confidence_threshold:
            return 1.0
        else:
            return quality_score / self.retrieval_confidence_threshold
    
    def _score_citation_quality(self, answer: str, citations: List[str], context: str) -> float:
        """
        Score quality of citations.
        """
        if not citations:
            return 0.0
        
        # Simple check: citations exist and are referenced
        import re
        citation_refs = re.findall(r'\[(\d+)\]', answer)
        
        if len(citation_refs) >= len(citations):
            return 1.0
        else:
            return len(citation_refs) / max(len(citations), 1)
    
    def _chunk_text(self, text: str, chunk_size: int = 50) -> List[str]:
        """Split text into chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
