"""
Input moderation and unsafe content filter.
"""
import logging
import re
from typing import Tuple, Optional
from detoxify import Detoxify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFilter:
    """
    Filter unsafe or inappropriate content from transcribed text.
    """
    
    def __init__(
        self,
        toxicity_threshold: float = 0.7,
        use_detoxify: bool = True
    ):
        """
        Initialize input filter.
        
        Args:
            toxicity_threshold: Score above which to reject input
            use_detoxify: Whether to use Detoxify model (slower but more accurate)
        """
        self.toxicity_threshold = toxicity_threshold
        self.use_detoxify = use_detoxify
        
        # Keyword-based filters (fast)
        self.unsafe_patterns = [
            r'\b(hack|exploit|bypass|jailbreak)\b',
            r'\b(illegal|criminal|fraud)\b',
        ]
        
        # Initialize Detoxify model if enabled
        self.detoxify_model = None
        if use_detoxify:
            try:
                logger.info("Loading Detoxify model...")
                self.detoxify_model = Detoxify('original')
                logger.info("Detoxify model loaded")
            except Exception as e:
                logger.warning(f"Failed to load Detoxify: {e}. Using keyword filter only.")
                self.use_detoxify = False
    
    def check(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if input text is safe.
        
        Args:
            text: Input text to check
        
        Returns:
            Tuple of (is_safe, rejection_reason)
            - is_safe: True if text passes filter
            - rejection_reason: Explanation if rejected, None if safe
        """
        if not text or len(text.strip()) == 0:
            return False, "Empty input"
        
        # Check length
        if len(text) > 5000:
            return False, "Input too long (max 5000 characters)"
        
        # Keyword-based checks (fast)
        text_lower = text.lower()
        for pattern in self.unsafe_patterns:
            if re.search(pattern, text_lower):
                return False, f"Input contains unsafe content (pattern: {pattern})"
        
        # Detoxify model check (more accurate but slower)
        if self.use_detoxify and self.detoxify_model:
            try:
                results = self.detoxify_model.predict(text)
                
                # Check toxicity scores
                for category, score in results.items():
                    if score > self.toxicity_threshold:
                        return False, f"Input rejected: {category} (score: {score:.2f})"
                
            except Exception as e:
                logger.warning(f"Detoxify check failed: {e}")
                # Continue without Detoxify check
        
        return True, None


# Global instance (singleton)
_filter_instance: Optional[InputFilter] = None


def get_filter(
    toxicity_threshold: float = 0.7,
    use_detoxify: bool = True,
    force_reload: bool = False
) -> InputFilter:
    """Get or create global input filter instance."""
    global _filter_instance
    
    if _filter_instance is None or force_reload:
        _filter_instance = InputFilter(
            toxicity_threshold=toxicity_threshold,
            use_detoxify=use_detoxify
        )
    
    return _filter_instance


def check_input(text: str) -> Tuple[bool, Optional[str]]:
    """Convenience function for input checking."""
    filter_instance = get_filter()
    return filter_instance.check(text)


if __name__ == "__main__":
    # Test input filter
    filter_instance = get_filter()
    
    test_cases = [
        "What is machine learning?",
        "How to hack a system?",
        "",
        "This is a normal query about AI and natural language processing.",
    ]
    
    for text in test_cases:
        is_safe, reason = filter_instance.check(text)
        status = "✓ SAFE" if is_safe else "✗ REJECTED"
        logger.info(f"{status}: '{text[:50]}...' - {reason or 'Passed all checks'}")
