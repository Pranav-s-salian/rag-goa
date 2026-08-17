"""
Post-generation hallucination/groundedness check using sentence-level NLI.
"""
import logging
import re
from typing import List, Tuple, Optional, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HallucinationChecker:
    """
    Check if generated answer is entailed by retrieved context using sentence-level NLI.
    More robust than paragraph-vs-paragraph comparison.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        entailment_threshold: float = 0.5,
        device: str = "cpu",
        aggregation_method: str = "average"  # 'average' or 'min'
    ):
        """
        Initialize hallucination checker.
        
        Args:
            model_name: NLI model name
            entailment_threshold: Minimum entailment probability to be considered grounded
            device: Device to run model on
            aggregation_method: How to aggregate sentence scores ('average' or 'min')
        """
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.device = device
        self.aggregation_method = aggregation_method
        
        logger.info(f"Loading NLI model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        logger.info(f"NLI model loaded (aggregation: {aggregation_method})")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        Uses simple regex - works for most English and Indic languages.
        """
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?।])\s+', text.strip())
        # Filter out very short sentences (likely fragments)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences
    
    def check(
        self,
        generated_answer: str,
        context_chunks: List[str],
        return_details: bool = False
    ) -> Tuple[bool, Optional[str], Optional[dict]]:
        """
        Check if generated answer is entailed by context using sentence-level NLI.
        
        Strategy:
        1. Split answer into sentences
        2. For each sentence, check entailment against all chunks (take max)
        3. Aggregate sentence scores (average or min)
        4. Compare to threshold
        
        Args:
            generated_answer: Generated answer text (MUST be fresh, not cached)
            context_chunks: List of retrieved context chunks
            return_details: Whether to return detailed scores
        
        Returns:
            Tuple of (is_grounded, rejection_reason, details)
            - is_grounded: True if answer is entailed by context
            - rejection_reason: Explanation if not grounded, None if grounded
            - details: Optional dict with entailment scores
        """
        # Validate inputs
        if not generated_answer or len(generated_answer.strip()) == 0:
            return False, "Generated answer is empty", None
        
        if not context_chunks or len(context_chunks) == 0:
            return False, "No context chunks provided", None
        
        # Split answer into sentences
        answer_sentences = self._split_into_sentences(generated_answer)
        
        if not answer_sentences:
            # Single sentence or no sentence boundaries
            answer_sentences = [generated_answer.strip()]
        
        logger.debug(f"Checking {len(answer_sentences)} sentences against {len(context_chunks)} chunks")
        
        # For each sentence, get max entailment score across all chunks
        sentence_scores = []
        sentence_details = []
        
        for sent_idx, sentence in enumerate(answer_sentences):
            chunk_scores = []
            
            for chunk_idx, chunk in enumerate(context_chunks):
                # NLI: premise=chunk, hypothesis=sentence
                # Check if the chunk entails this sentence
                score = self._compute_entailment(chunk, sentence)
                chunk_scores.append(score)
            
            # Take max score across chunks (best supporting chunk for this sentence)
            max_score = max(chunk_scores) if chunk_scores else 0.0
            sentence_scores.append(max_score)
            
            sentence_details.append({
                'sentence': sentence[:100] + '...' if len(sentence) > 100 else sentence,
                'max_chunk_score': float(max_score),
                'best_chunk_idx': int(np.argmax(chunk_scores)) if chunk_scores else -1
            })
        
        # Aggregate sentence scores
        if self.aggregation_method == "min":
            # Min: all sentences must be grounded (strictest)
            final_score = min(sentence_scores) if sentence_scores else 0.0
        else:
            # Average: sentences are grounded on average (default)
            final_score = np.mean(sentence_scores) if sentence_scores else 0.0
        
        # Check if answer is grounded
        is_grounded = final_score >= self.entailment_threshold
        
        details = None
        if return_details:
            details = {
                'final_score': float(final_score),
                'aggregation_method': self.aggregation_method,
                'num_sentences': len(answer_sentences),
                'sentence_scores': [float(s) for s in sentence_scores],
                'sentence_details': sentence_details,
                'threshold': self.entailment_threshold,
                'answer_preview': generated_answer[:200] + '...' if len(generated_answer) > 200 else generated_answer
            }
        
        if not is_grounded:
            reason = (
                f"Generated answer may contain hallucinations. "
                f"Entailment score: {final_score:.3f} (threshold: {self.entailment_threshold}). "
                f"Checked {len(answer_sentences)} sentences using {self.aggregation_method} aggregation. "
                f"The answer is not sufficiently supported by the retrieved context."
            )
            return False, reason, details
        
        return True, None, details
    
    def _compute_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Compute entailment probability between premise and hypothesis.
        
        Returns:
            Entailment probability (0-1)
        """
        # Tokenize
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Get logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
        
        # Convert to probabilities
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        
        # For NLI models: typically [contradiction, neutral, entailment]
        # Return entailment probability
        if len(probs) == 3:
            entailment_prob = probs[2]  # Index 2 is typically entailment
        elif len(probs) == 2:
            entailment_prob = probs[1]  # Binary: [not_entailment, entailment]
        else:
            logger.warning(f"Unexpected number of classes: {len(probs)}")
            entailment_prob = probs[-1]  # Use last class as fallback
        
        return float(entailment_prob)
    
    def check_and_suggest_fix(
        self,
        generated_answer: str,
        context_chunks: List[str]
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check hallucination and suggest a fix if needed.
        
        Returns:
            Tuple of (is_grounded, rejection_reason, suggested_fix)
        """
        is_grounded, reason, details = self.check(
            generated_answer,
            context_chunks,
            return_details=True
        )
        
        if is_grounded:
            return True, None, None
        
        # Suggest extractive summary as fallback
        # Take the chunk with highest entailment score
        if details and details['scores']:
            best_chunk_idx = np.argmax(details['scores'])
            best_chunk = context_chunks[best_chunk_idx]
            
            suggested_fix = (
                f"Based on the available information: {best_chunk[:200]}... "
                f"I cannot provide a complete answer with high confidence."
            )
        else:
            suggested_fix = None
        
        return False, reason, suggested_fix


# Global instance
_checker_instance: Optional[HallucinationChecker] = None


def get_checker(
    model_name: str = "cross-encoder/nli-deberta-v3-small",
    entailment_threshold: float = 0.5,
    device: str = "cpu",
    force_reload: bool = False
) -> HallucinationChecker:
    """Get or create global hallucination checker instance."""
    global _checker_instance
    
    if _checker_instance is None or force_reload:
        _checker_instance = HallucinationChecker(
            model_name=model_name,
            entailment_threshold=entailment_threshold,
            device=device
        )
    
    return _checker_instance


def check_hallucination(
    generated_answer: str,
    context_chunks: List[str]
) -> Tuple[bool, Optional[str]]:
    """Convenience function for hallucination checking."""
    checker = get_checker()
    is_grounded, reason, _ = checker.check(generated_answer, context_chunks)
    return is_grounded, reason


if __name__ == "__main__":
    # Test hallucination checker
    checker = get_checker()
    
    context = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers to process information."
    ]
    
    # Grounded answer
    grounded_answer = "Machine learning is a subset of AI that learns from data."
    is_grounded, reason, details = checker.check(grounded_answer, context, return_details=True)
    logger.info(f"Grounded test: {is_grounded}, details={details}")
    
    # Hallucinated answer
    hallucinated_answer = "Machine learning was invented in ancient Egypt by the pharaohs."
    is_grounded, reason, details = checker.check(hallucinated_answer, context, return_details=True)
    logger.info(f"Hallucinated test: {is_grounded}, reason={reason}")
