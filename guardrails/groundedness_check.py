"""
Groundedness check: verify query has sufficient semantic similarity to retrieved context.
"""
import logging
from typing import List, Tuple, Optional
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroundednessChecker:
    """
    Check if query is grounded in retrieved context based on embedding similarity.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.3,
        min_chunks_above_threshold: int = 1
    ):
        """
        Initialize groundedness checker.
        
        Args:
            similarity_threshold: Minimum cosine similarity to be considered grounded
            min_chunks_above_threshold: Minimum number of chunks that must exceed threshold
        """
        self.similarity_threshold = similarity_threshold
        self.min_chunks_above_threshold = min_chunks_above_threshold
    
    def check(
        self,
        query_embedding: np.ndarray,
        chunk_embeddings: List[np.ndarray],
        chunk_texts: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[str], float]:
        """
        Check if query is grounded in the retrieved chunks.
        
        Args:
            query_embedding: Query embedding vector
            chunk_embeddings: List of chunk embedding vectors
            chunk_texts: Optional list of chunk texts for logging
        
        Returns:
            Tuple of (is_grounded, rejection_reason, max_similarity)
            - is_grounded: True if query is grounded
            - rejection_reason: Explanation if not grounded, None if grounded
            - max_similarity: Maximum similarity score found
        """
        if len(chunk_embeddings) == 0:
            return False, "No chunks retrieved", 0.0
        
        # Normalize embeddings
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Calculate cosine similarities
        similarities = []
        for chunk_emb in chunk_embeddings:
            chunk_emb = chunk_emb / (np.linalg.norm(chunk_emb) + 1e-8)
            sim = np.dot(query_embedding, chunk_emb)
            similarities.append(float(sim))
        
        max_similarity = max(similarities)
        chunks_above_threshold = sum(1 for s in similarities if s >= self.similarity_threshold)
        
        # Check groundedness
        is_grounded = (
            max_similarity >= self.similarity_threshold and
            chunks_above_threshold >= self.min_chunks_above_threshold
        )
        
        if not is_grounded:
            reason = (
                f"Query not grounded in retrieved context. "
                f"Max similarity: {max_similarity:.3f} (threshold: {self.similarity_threshold}), "
                f"Chunks above threshold: {chunks_above_threshold}/{len(similarities)} "
                f"(required: {self.min_chunks_above_threshold})"
            )
            return False, reason, max_similarity
        
        return True, None, max_similarity
    
    def check_with_scores(
        self,
        query_embedding: np.ndarray,
        retrieval_results: List
    ) -> Tuple[bool, Optional[str], float]:
        """
        Check groundedness using retrieval results (which already have scores).
        
        Args:
            query_embedding: Query embedding vector
            retrieval_results: List of RetrievalResult objects with scores
        
        Returns:
            Tuple of (is_grounded, rejection_reason, max_score)
        """
        if len(retrieval_results) == 0:
            return False, "No results retrieved", 0.0
        
        # For reranked results, use reranker scores directly
        # For FAISS results, scores are negative L2 distances - convert to similarity proxy
        scores = [r.score for r in retrieval_results]
        max_score = max(scores)
        
        # Normalize scores to [0, 1] range if they're negative (FAISS distances)
        if max_score < 0:
            # These are negative L2 distances - closer to 0 is better
            # Convert: -distance -> similarity proxy
            scores = [1.0 / (1.0 + abs(s)) for s in scores]
            max_score = max(scores)
        
        # Check if max score exceeds threshold
        chunks_above_threshold = sum(1 for s in scores if s >= self.similarity_threshold)
        
        is_grounded = (
            max_score >= self.similarity_threshold and
            chunks_above_threshold >= self.min_chunks_above_threshold
        )
        
        if not is_grounded:
            reason = (
                f"Query not sufficiently related to retrieved context. "
                f"Max relevance score: {max_score:.3f} (threshold: {self.similarity_threshold}), "
                f"Relevant chunks: {chunks_above_threshold}/{len(scores)}"
            )
            return False, reason, max_score
        
        return True, None, max_score


# Global instance
_checker_instance: Optional[GroundednessChecker] = None


def get_checker(
    similarity_threshold: float = 0.3,
    min_chunks_above_threshold: int = 1,
    force_reload: bool = False
) -> GroundednessChecker:
    """Get or create global groundedness checker instance."""
    global _checker_instance
    
    if _checker_instance is None or force_reload:
        _checker_instance = GroundednessChecker(
            similarity_threshold=similarity_threshold,
            min_chunks_above_threshold=min_chunks_above_threshold
        )
    
    return _checker_instance


def check_groundedness(
    query_embedding: np.ndarray,
    retrieval_results: List
) -> Tuple[bool, Optional[str], float]:
    """Convenience function for groundedness checking."""
    checker = get_checker()
    return checker.check_with_scores(query_embedding, retrieval_results)


if __name__ == "__main__":
    # Test groundedness checker
    checker = get_checker(similarity_threshold=0.3)
    
    # Simulate embeddings
    query_emb = np.random.randn(384)
    
    # High similarity chunks
    high_sim_chunks = [query_emb + np.random.randn(384) * 0.1 for _ in range(3)]
    is_grounded, reason, max_sim = checker.check(query_emb, high_sim_chunks)
    logger.info(f"High similarity test: {is_grounded}, max_sim={max_sim:.3f}")
    
    # Low similarity chunks
    low_sim_chunks = [np.random.randn(384) for _ in range(3)]
    is_grounded, reason, max_sim = checker.check(query_emb, low_sim_chunks)
    logger.info(f"Low similarity test: {is_grounded}, reason={reason}")
