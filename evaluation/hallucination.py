"""
Hallucination detection logic with explicit, inspectable checks.
Implements answer-context overlap, citation validation, and unsupported claim detection.
"""
import re
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Detects hallucinations in LLM outputs through multiple explicit checks.
    """
    
    def __init__(
        self,
        hallucination_threshold: float = 0.7,
        citation_required: bool = True,
        unsupported_claim_detection: bool = True
    ):
        self.hallucination_threshold = hallucination_threshold
        self.citation_required = citation_required
        self.unsupported_claim_detection = unsupported_claim_detection
        
        # Initialize sentence transformer for semantic similarity
        try:
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Failed to load similarity model: {str(e)}")
            self.similarity_model = None
    
    def detect(
        self,
        answer: str,
        context: Optional[str] = None,
        citations: Optional[List[str]] = None,
        retrieval_scores: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Detect hallucinations in the answer.
        
        Args:
            answer: Generated answer text
            context: Retrieved context (for RAG systems)
            citations: List of citations in answer
            retrieval_scores: Confidence scores for retrieved chunks
            
        Returns:
            Dictionary with hallucination scores and explanations
        """
        results = {
            "hallucination_score": 0.0,
            "is_hallucination": False,
            "checks": {},
            "explanation": []
        }
        
        # Check 1: Answer-Context Overlap (if context provided)
        if context:
            overlap_score = self._check_answer_context_overlap(answer, context)
            results["checks"]["answer_context_overlap"] = {
                "score": overlap_score,
                "threshold": self.hallucination_threshold,
                "passed": overlap_score >= self.hallucination_threshold
            }
            results["hallucination_score"] += (1.0 - overlap_score) * 0.4
            if overlap_score < self.hallucination_threshold:
                results["explanation"].append(
                    f"Low answer-context overlap ({overlap_score:.2f}). "
                    f"Answer may contain information not in context."
                )
        
        # Check 2: Citation Presence and Validity
        if self.citation_required:
            citation_check = self._check_citations(answer, citations)
            results["checks"]["citation_validation"] = citation_check
            if not citation_check["valid"]:
                results["hallucination_score"] += 0.3
                results["explanation"].append(
                    f"Citation validation failed: {citation_check['reason']}"
                )
        
        # Check 3: Unsupported Claim Detection
        if self.unsupported_claim_detection and context:
            unsupported_claims = self._detect_unsupported_claims(answer, context)
            results["checks"]["unsupported_claims"] = {
                "count": len(unsupported_claims),
                "claims": unsupported_claims,
                "passed": len(unsupported_claims) == 0
            }
            if unsupported_claims:
                results["hallucination_score"] += min(len(unsupported_claims) * 0.1, 0.3)
                results["explanation"].append(
                    f"Found {len(unsupported_claims)} unsupported claims in answer."
                )
        
        # Check 4: Retrieval Confidence (for RAG systems)
        if retrieval_scores:
            low_confidence = self._check_retrieval_confidence(retrieval_scores)
            results["checks"]["retrieval_confidence"] = {
                "scores": retrieval_scores,
                "min_confidence": min(retrieval_scores) if retrieval_scores else 0.0,
                "low_confidence_detected": low_confidence
            }
            if low_confidence:
                results["hallucination_score"] += 0.2
                results["explanation"].append(
                    "Low retrieval confidence scores detected."
                )
        
        # Normalize score to [0, 1]
        results["hallucination_score"] = min(results["hallucination_score"], 1.0)
        results["is_hallucination"] = results["hallucination_score"] >= self.hallucination_threshold
        
        return results
    
    def _check_answer_context_overlap(self, answer: str, context: str) -> float:
        """
        Calculate semantic overlap between answer and context.
        Returns score between 0 and 1, where 1 is perfect overlap.
        """
        if not self.similarity_model:
            # Fallback to simple word overlap
            answer_words = set(answer.lower().split())
            context_words = set(context.lower().split())
            if not answer_words:
                return 0.0
            overlap = len(answer_words & context_words) / len(answer_words)
            return overlap
        
        try:
            # Semantic similarity using sentence transformers
            answer_embedding = self.similarity_model.encode([answer], convert_to_numpy=True)
            context_embedding = self.similarity_model.encode([context], convert_to_numpy=True)
            
            # Cosine similarity
            similarity = np.dot(answer_embedding[0], context_embedding[0]) / (
                np.linalg.norm(answer_embedding[0]) * np.linalg.norm(context_embedding[0])
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating overlap: {str(e)}")
            return 0.0
    
    def _check_citations(self, answer: str, citations: Optional[List[str]]) -> Dict[str, Any]:
        """
        Validate citations in the answer.
        Checks for citation markers and validates citation format.
        """
        # Extract citation markers (e.g., [1], [2], (Source 1), etc.)
        citation_patterns = [
            r'\[(\d+)\]',  # [1], [2]
            r'\(([^)]+)\)',  # (Source 1)
            r'\[([^\]]+)\]',  # [Source]
        ]
        
        found_citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, answer)
            found_citations.extend(matches)
        
        result = {
            "valid": True,
            "citations_found": len(found_citations),
            "citations_expected": len(citations) if citations else 0,
            "reason": ""
        }
        
        if self.citation_required:
            if not found_citations:
                result["valid"] = False
                result["reason"] = "No citations found in answer"
            elif citations and len(found_citations) < len(citations):
                result["valid"] = False
                result["reason"] = f"Expected {len(citations)} citations, found {len(found_citations)}"
        
        return result
    
    def _detect_unsupported_claims(self, answer: str, context: str) -> List[str]:
        """
        Detect claims in answer that are not supported by context.
        Returns list of unsupported claim sentences.
        """
        unsupported = []
        
        # Split answer into sentences
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not self.similarity_model:
            # Simple word-based check
            context_words = set(context.lower().split())
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                # If sentence has significant words not in context
                if len(sentence_words) > 5:
                    unique_words = sentence_words - context_words
                    if len(unique_words) / len(sentence_words) > 0.5:
                        unsupported.append(sentence)
            return unsupported
        
        try:
            # Semantic similarity check
            context_embedding = self.similarity_model.encode([context], convert_to_numpy=True)[0]
            
            for sentence in sentences:
                if len(sentence.split()) < 5:  # Skip very short sentences
                    continue
                
                sentence_embedding = self.similarity_model.encode([sentence], convert_to_numpy=True)[0]
                similarity = np.dot(sentence_embedding, context_embedding) / (
                    np.linalg.norm(sentence_embedding) * np.linalg.norm(context_embedding)
                )
                
                # If similarity is very low, claim is likely unsupported
                if similarity < 0.3:
                    unsupported.append(sentence)
        except Exception as e:
            logger.error(f"Error detecting unsupported claims: {str(e)}")
        
        return unsupported
    
    def _check_retrieval_confidence(self, retrieval_scores: List[float], threshold: float = 0.7) -> bool:
        """
        Check if retrieval confidence scores are below threshold.
        Returns True if low confidence detected.
        """
        if not retrieval_scores:
            return True
        
        min_score = min(retrieval_scores)
        avg_score = sum(retrieval_scores) / len(retrieval_scores)
        
        return min_score < threshold or avg_score < threshold
