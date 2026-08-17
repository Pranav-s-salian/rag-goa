"""
LLM generation using Groq API with context-constrained prompting.
"""
import time
import logging
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from generation operation."""
    answer: str
    source_chunks: List[str]
    model: str
    time_to_first_token_ms: Optional[float]
    total_generation_ms: float
    tokens_generated: Optional[int]


class Generator:
    """
    LLM generator using Groq API for fast inference.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.1,
        max_tokens: int = 200
    ):
        """
        Initialize generator.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            model: Model name (default: llama-3.1-8b-instant for speed)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (default: 200 for concise answers)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be provided or set in environment")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.client = Groq(api_key=self.api_key)
        logger.info(f"Generator initialized with model: {model}")
    
    def generate(
        self,
        query: str,
        context_chunks: List[str],
        max_context_length: int = 2048,
        use_streaming: bool = True,
        query_language: Optional[str] = None,
        top_k_chunks: int = 5
    ) -> GenerationResult:
        """
        Generate answer from query and context.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks (will be sliced to top_k_chunks)
            max_context_length: Maximum characters for context
            use_streaming: Whether to use streaming for lower latency
            query_language: Detected query language ('hindi'/'english') for response matching
            top_k_chunks: Number of top chunks to use (default: 5 for speed)
        
        Returns:
            GenerationResult with answer and metrics
        """
        start_time = time.perf_counter()
        
        # Slice to top-k chunks before building context (performance optimization)
        context_chunks_sliced = context_chunks[:top_k_chunks]
        
        # Build context from chunks
        context = self._build_context(context_chunks_sliced, max_context_length)
        
        # Build prompt with language instruction
        prompt = self._build_prompt(query, context, query_language)
        
        # Generate
        if use_streaming:
            answer, ttft_ms, tokens = self._generate_streaming(prompt)
        else:
            answer, tokens = self._generate_non_streaming(prompt)
            ttft_ms = None
        
        total_ms = (time.perf_counter() - start_time) * 1000
        
        return GenerationResult(
            answer=answer,
            source_chunks=context_chunks_sliced,
            model=self.model,
            time_to_first_token_ms=ttft_ms,
            total_generation_ms=total_ms,
            tokens_generated=tokens
        )
    
    def _build_context(self, chunks: List[str], max_length: int) -> str:
        """Build context string from chunks, truncating if needed."""
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks):
            chunk_header = f"[Context {i+1}]\n{chunk}\n"
            
            if current_length + len(chunk_header) > max_length:
                break
            
            context_parts.append(chunk_header)
            current_length += len(chunk_header)
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str, query_language: Optional[str] = None) -> str:
        """Build context-constrained prompt with language instruction."""
        
        # Language instruction based on detected query language
        language_instruction = ""
        if query_language:
            lang_name = "Hindi" if query_language == "hindi" else "English"
            language_instruction = f"\n6. CRITICAL: You MUST respond in {lang_name}. If the context is in a different language, extract the relevant information and provide your answer in {lang_name}."
        
        system_prompt = f"""You are a helpful AI assistant that provides direct, concise answers from the given context.

IMPORTANT RULES:
1. Answer ONLY using information explicitly stated in the context
2. If the context lacks sufficient information, say "I don't have enough information to answer this question."
3. Do NOT use external knowledge or make assumptions
4. Give a DIRECT, CONCISE answer - do NOT show your thinking process or analysis steps
5. Do NOT include phrases like "Based on Context 1" or "According to the passage" - just state the answer directly{language_instruction}

Your answer should be factual, brief, and directly address the question."""
        
        user_prompt = f"""Context:
{context}

Question: {query}

Direct Answer:"""
        
        return f"{system_prompt}\n\n{user_prompt}"
    
    def _generate_streaming(self, prompt: str) -> Tuple[str, float, int]:
        """Generate with streaming to measure time-to-first-token."""
        first_token_time = None
        start_time = time.perf_counter()
        
        answer_chunks = []
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if first_token_time is None and chunk.choices:
                    first_token_time = (time.perf_counter() - start_time) * 1000
                
                if chunk.choices and chunk.choices[0].delta.content:
                    answer_chunks.append(chunk.choices[0].delta.content)
            
            answer = "".join(answer_chunks)
            tokens = len(answer.split())  # Rough estimate
            
            return answer, first_token_time, tokens
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise
    
    def _generate_non_streaming(self, prompt: str) -> Tuple[str, int]:
        """Generate without streaming."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            
            answer = response.choices[0].message.content
            tokens = response.usage.completion_tokens if response.usage else None
            
            return answer, tokens
            
        except Exception as e:
            logger.error(f"Non-streaming generation failed: {e}")
            raise
    
    def generate_with_retry(
        self,
        query: str,
        context_chunks: List[str],
        max_retries: int = 2,
        stricter_prompt: bool = False
    ) -> GenerationResult:
        """
        Generate with retry logic for hallucination failures.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            max_retries: Maximum number of retries
            stricter_prompt: Whether to use an even stricter prompt
        """
        for attempt in range(max_retries + 1):
            try:
                if stricter_prompt and attempt > 0:
                    # Make prompt stricter on retries
                    logger.info(f"Retry attempt {attempt} with stricter prompt")
                    # Could modify temperature or prompt here
                    self.temperature = max(0.0, self.temperature - 0.05)
                
                result = self.generate(query, context_chunks)
                return result
                
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Generation attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(1)  # Brief backoff
                else:
                    logger.error(f"All generation attempts failed: {e}")
                    raise


# Global instance
_generator_instance: Optional[Generator] = None


def get_generator(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    force_reload: bool = False
) -> Generator:
    """Get or create global generator instance."""
    global _generator_instance
    
    if _generator_instance is None or force_reload:
        model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        _generator_instance = Generator(api_key=api_key, model=model)
    
    return _generator_instance


def generate(
    query: str,
    context_chunks: List[str]
) -> GenerationResult:
    """Convenience function for generation."""
    generator = get_generator()
    return generator.generate(query, context_chunks)


if __name__ == "__main__":
    # Test generator (requires GROQ_API_KEY)
    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY not set. Cannot test generator.")
    else:
        generator = get_generator()
        
        test_query = "What is machine learning?"
        test_context = [
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "Deep learning uses neural networks with multiple layers."
        ]
        
        result = generator.generate(test_query, test_context)
        
        logger.info(f"\nQuery: {test_query}")
        logger.info(f"Answer: {result.answer}")
        logger.info(f"TTFT: {result.time_to_first_token_ms:.2f}ms")
        logger.info(f"Total: {result.total_generation_ms:.2f}ms")
