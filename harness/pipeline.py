"""
End-to-end RAG pipeline orchestrator with staged execution and error handling.
"""
import os
import time
import logging
import asyncio
import re
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from tenacity import retry, stop_after_attempt, wait_exponential
import sys

sys.path.append(str(Path(__file__).parent.parent))

from stt.sarvam_client import get_stt_client, TranscriptionResult
from guardrails.input_filter import get_filter as get_input_filter
from guardrails.groundedness_check import get_checker as get_groundedness_checker
from guardrails.hallucination_check import get_checker as get_hallucination_checker
from embeddings.embed import get_model as get_embedding_model
from retrieval.retriever import Retriever
from generation.generate import get_generator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_query_language(text: str) -> str:
    """
    Detect language of query text using simple heuristic.
    
    Args:
        text: Query text
    
    Returns:
        'hindi' if Devanagari script detected, 'english' otherwise
    """
    # Check for Devanagari Unicode range (U+0900 to U+097F)
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    
    if devanagari_pattern.search(text):
        return 'hindi'
    else:
        return 'english'


# ============================================================================
# HALLUCINATION CHECK CONTROL
# ============================================================================
# TODO: re-enable before final submission — NLI model currently miscalibrated 
# for Hindi text, causing false positives on correct answers (e.g. EMG query 
# scored 0.437 despite a correct answer).
#
# Known issues before re-enabling:
# - NLI model (English-trained cross-encoder/nli-deberta-v3-small) under-scores 
#   valid Hindi answers — needs swap to a multilingual model e.g. 
#   MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (XNLI-trained, includes Hindi)
# - Retry logic re-runs hallucination check even when retry answer is identical 
#   to original (see "Answer changed: False" in logs) — wastes a Groq call + 
#   NLI pass, skip re-checking if retry text == original text
# - Threshold (0.5) was never empirically calibrated — needs testing against 
#   known-good and known-bad Hindi Q&A pairs to find real separation point 
#   before re-enabling
# - Sentence-level NLI implementation is correct but calibration is critical
#   for multilingual use cases
#
HALLUCINATION_CHECK_ENABLED = False
# ============================================================================


# Refusal patterns - if generation produces these, skip hallucination check
REFUSAL_PATTERNS = [
    r"^I don't have enough information",
    r"^I don't have sufficient information",
    r"^I cannot answer",
    r"^I cannot provide",
    r"^I'm unable to",
    r"^I am unable to",
    r"^I don't know",
    r"^I do not know",
    r"^Based on the available information, I cannot",
    r"^The provided context does not contain",
    r"^I cannot find",
    r"^There is no information",
    r"^The context does not include",
]

# Compile patterns for efficiency
REFUSAL_REGEX = re.compile('|'.join(REFUSAL_PATTERNS), re.IGNORECASE)


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    # Index configuration
    strategy_name: str = "fixed"
    models_dir: str = "models"
    
    # Production index settings
    use_production_index: bool = True
    production_bucket: str = None  # Must be set via env var (PRODUCTION_BUCKET)
    index_nprobe: int = 16  # IVF-PQ nprobe (higher = better recall, slower)
    
    # Retrieval configuration
    top_k: int = 10
    enable_rerank: bool = False
    
    # Guardrail configuration
    enable_input_filter: bool = True
    enable_groundedness_check: bool = True
    enable_hallucination_check: bool = True
    groundedness_threshold: float = 0.3
    hallucination_threshold: float = 0.5
    
    # Generation configuration
    max_context_length: int = 2048
    
    # Timeout configuration (seconds)
    stt_timeout: float = 30.0
    retrieval_timeout: float = 5.0
    generation_timeout: float = 15.0


@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    # Output
    answer: Optional[str]
    refused: bool
    refusal_reason: Optional[str]
    source_chunks: Optional[List[str]]
    
    # Latency breakdown (milliseconds)
    stt_ms: float
    retrieval_ms: float
    generation_ms: float
    guardrail_ms: float
    total_ms: float
    
    # Detailed stage info
    stage_details: Dict[str, Any]
    
    # Error info
    error: Optional[str] = None


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline with guardrails and error handling.
    """
    
    def __init__(self, config: PipelineConfig, skip_warmup: bool = False):
        """Initialize pipeline with configuration."""
        self.config = config
        
        # Track retrieval latencies for cold-start detection
        self.retrieval_latency_history = []
        
        logger.info("Initializing RAG pipeline...")
        
        # Initialize components
        self.stt_client = None  # Lazy load
        self.input_filter = get_input_filter() if config.enable_input_filter else None
        self.groundedness_checker = get_groundedness_checker(
            similarity_threshold=config.groundedness_threshold
        ) if config.enable_groundedness_check else None
        
        # Always load hallucination checker (even if disabled) to avoid restart-and-debug
        # cycle when re-enabling. Model loading is kept warm for quick re-activation.
        self.hallucination_checker = get_hallucination_checker(
            entailment_threshold=config.hallucination_threshold
        ) if config.enable_hallucination_check else None
        
        # Initialize embedding model
        self.embedding_model = get_embedding_model(use_onnx=True, use_gpu=False)
        
        # Initialize retriever
        models_dir = Path(config.models_dir)
        self.retriever = Retriever(
            strategy_name=config.strategy_name,
            models_dir=models_dir,
            embedding_model=self.embedding_model,
            enable_rerank=config.enable_rerank,
            use_production_index=config.use_production_index,
            production_bucket=config.production_bucket,
            nprobe=config.index_nprobe
        )
        
        # Initialize generator
        self.generator = get_generator()
        
        logger.info("Pipeline initialized successfully")
        
        # Warm-up to eliminate cold-start latency
        if not skip_warmup:
            self._warmup()
    
    def _warmup(self):
        """
        Warm-up embedding model and FAISS index to eliminate cold-start latency.
        Fires a dummy query through the full retrieval path.
        """
        logger.info("🔥 Warming up embedding + FAISS retrieval path...")
        warmup_start = time.perf_counter()
        
        try:
            # Dummy query
            dummy_query = "What is machine learning?"
            
            # Embed query (warms up embedding model)
            query_embedding = self.embedding_model.embed(dummy_query, prefix="query: ")
            
            # Retrieve (warms up FAISS index)
            results, metrics = self.retriever.retrieve(dummy_query, top_k=5)
            
            warmup_ms = (time.perf_counter() - warmup_start) * 1000
            logger.info(f"✓ Warm-up completed in {warmup_ms:.2f}ms (retrieved {len(results)} chunks)")
            logger.info("  First real query should now see <100ms retrieval latency")
            
        except Exception as e:
            logger.warning(f"⚠ Warm-up failed (non-fatal): {e}")
            logger.warning("  First query may experience cold-start latency")
    
    def _is_refusal_response(self, answer: str) -> bool:
        """
        Check if generated answer is a refusal/inability response.
        These are valid responses that don't need hallucination checking.
        
        Args:
            answer: Generated answer text
        
        Returns:
            True if answer matches refusal pattern
        """
        if not answer:
            return False
        
        # Check against refusal patterns
        return bool(REFUSAL_REGEX.match(answer.strip()))
    
    async def run(
        self,
        query: Optional[str] = None,
        audio_file: Optional[str] = None
    ) -> PipelineResult:
        """
        Run the full pipeline.
        
        Args:
            query: Text query (if no audio)
            audio_file: Path to audio file (if no text query)
        
        Returns:
            PipelineResult with answer and metrics
        """
        pipeline_start = time.perf_counter()
        stage_details = {}
        
        stt_ms = 0.0
        retrieval_ms = 0.0
        generation_ms = 0.0
        guardrail_ms = 0.0
        query_language = None  # Track detected language
        
        try:
            # Stage 1: Speech-to-Text (if audio provided)
            if audio_file:
                logger.info("Stage 1: Transcription")
                query, stt_ms, stt_details = await self._stage_transcribe(audio_file)
                stage_details['transcription'] = stt_details
                
                # Extract language from STT response
                if 'language' in stt_details:
                    stt_lang = stt_details['language']  # e.g., 'en-IN' or 'hi-IN'
                    if stt_lang and stt_lang.startswith('hi'):
                        query_language = 'hindi'
                    elif stt_lang and stt_lang.startswith('en'):
                        query_language = 'english'
                
                logger.info(f"  Transcribed: '{query[:50]}...' ({stt_ms:.2f}ms)")
                logger.info(f"  Detected language: {query_language or 'unknown'}")
            elif not query:
                return self._error_result("No query or audio provided", pipeline_start)
            
            # For text queries, detect language via Devanagari check
            if query and not query_language:
                query_language = detect_query_language(query)
                logger.info(f"  Detected query language (text): {query_language}")
            
            # Stage 2: Input guardrail
            if self.input_filter:
                logger.info("Stage 2: Input filter")
                guard_start = time.perf_counter()
                is_safe, reason = self.input_filter.check(query)
                guard_time = (time.perf_counter() - guard_start) * 1000
                guardrail_ms += guard_time
                
                stage_details['input_filter'] = {'safe': is_safe, 'reason': reason}
                
                if not is_safe:
                    logger.warning(f"  Input rejected: {reason}")
                    return self._refusal_result(
                        f"Input rejected: {reason}",
                        stt_ms, 0, 0, guardrail_ms, pipeline_start
                    )
                logger.info(f"  Input safe ({guard_time:.2f}ms)")
            
            # Stage 3: Retrieval
            logger.info("Stage 3: Retrieval")
            retrieval_results, retrieval_metrics, retrieval_ms = await self._stage_retrieve(query)
            
            # Track retrieval latency for cold-start analysis
            self.retrieval_latency_history.append(retrieval_ms)
            if len(self.retrieval_latency_history) <= 3:
                logger.info(f"  📊 Retrieval #{len(self.retrieval_latency_history)}: {retrieval_ms:.2f}ms")
                if len(self.retrieval_latency_history) == 3:
                    avg_latency = sum(self.retrieval_latency_history) / 3
                    first_vs_avg = self.retrieval_latency_history[0] - avg_latency
                    logger.info(f"  📈 First 3 queries average: {avg_latency:.2f}ms")
                    if first_vs_avg > 100:
                        logger.info(f"  ⚠ Cold-start spike detected: {first_vs_avg:.2f}ms slower on first query")
                    else:
                        logger.info(f"  ✓ No significant cold-start spike (diff: {first_vs_avg:.2f}ms)")
            
            stage_details['retrieval'] = {
                'num_results': len(retrieval_results),
                'metrics': asdict(retrieval_metrics)
            }
            logger.info(f"  Retrieved {len(retrieval_results)} chunks ({retrieval_ms:.2f}ms)")
            
            # Stage 4: Groundedness check
            if self.groundedness_checker:
                logger.info("Stage 4: Groundedness check")
                guard_start = time.perf_counter()
                
                query_embedding = self.embedding_model.embed(query, prefix="query: ")
                is_grounded, reason, max_score = self.groundedness_checker.check_with_scores(
                    query_embedding,
                    retrieval_results
                )
                
                guard_time = (time.perf_counter() - guard_start) * 1000
                guardrail_ms += guard_time
                
                stage_details['groundedness'] = {
                    'grounded': is_grounded,
                    'max_score': max_score,
                    'reason': reason
                }
                
                if not is_grounded:
                    logger.warning(f"  Query not grounded: {reason}")
                    return self._refusal_result(
                        "I don't have enough relevant information to answer this question.",
                        stt_ms, retrieval_ms, 0, guardrail_ms, pipeline_start
                    )
                logger.info(f"  Query grounded (score: {max_score:.3f}, {guard_time:.2f}ms)")
            
            # Stage 5: Generation
            logger.info("Stage 5: Generation")
            answer, generation_ms, gen_details = await self._stage_generate(
                query,
                [r.text for r in retrieval_results],
                query_language=query_language  # Pass detected language
            )
            stage_details['generation'] = gen_details
            logger.info(f"  Generated answer ({generation_ms:.2f}ms)")
            
            # Check if answer is a refusal response
            is_refusal = self._is_refusal_response(answer)
            if is_refusal:
                logger.info("  ℹ Answer is a refusal response - skipping hallucination check")
                stage_details['refusal_detected'] = True
                stage_details['hallucination_check_status'] = 'skipped_refusal'
                
                # Return refusal as valid response (not an error)
                total_ms = (time.perf_counter() - pipeline_start) * 1000
                return PipelineResult(
                    answer=answer,
                    refused=False,  # It's a valid response, just acknowledging lack of info
                    refusal_reason=None,
                    source_chunks=[r.text for r in retrieval_results],
                    stt_ms=stt_ms,
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    guardrail_ms=guardrail_ms,
                    total_ms=total_ms,
                    stage_details=stage_details
                )
            
            # Stage 6: Hallucination check
            if HALLUCINATION_CHECK_ENABLED and self.hallucination_checker:
                logger.info("Stage 6: Hallucination check")
                guard_start = time.perf_counter()
                
                is_grounded, reason, details = self.hallucination_checker.check(
                    answer,
                    [r.text for r in retrieval_results],
                    return_details=True
                )
                
                guard_time = (time.perf_counter() - guard_start) * 1000
                guardrail_ms += guard_time
                
                stage_details['hallucination_check'] = {
                    'grounded': is_grounded,
                    'details': details,
                    'reason': reason
                }
                
                if not is_grounded:
                    logger.warning(f"  Hallucination detected: {reason}")
                    logger.warning(f"  Initial answer (first 100 chars): {answer[:100]}...")
                    logger.warning(f"  Initial entailment score: {details['final_score'] if details else 'N/A'}")
                    
                    # Try one more time with stricter prompt
                    logger.info("  Retrying with stricter prompt...")
                    retry_answer, retry_gen_ms, retry_details = await self._stage_generate(
                        query,
                        [r.text for r in retrieval_results],
                        stricter=True,
                        query_language=query_language  # Pass detected language
                    )
                    generation_ms += retry_gen_ms
                    
                    logger.info(f"  Retry answer (first 100 chars): {retry_answer[:100]}...")
                    logger.info(f"  Answer changed: {retry_answer != answer}")
                    
                    # Check again with NEW answer
                    retry_guard_start = time.perf_counter()
                    is_grounded, reason, details = self.hallucination_checker.check(
                        retry_answer,  # ✓ Use the NEW answer
                        [r.text for r in retrieval_results],
                        return_details=True
                    )
                    retry_guard_time = (time.perf_counter() - retry_guard_start) * 1000
                    guardrail_ms += retry_guard_time
                    
                    logger.info(f"  Retry entailment score: {details['final_score'] if details else 'N/A'}")
                    
                    if not is_grounded:
                        logger.warning("  Hallucination persists after retry")
                        logger.warning(f"  Final score still: {details['final_score'] if details else 'N/A'}")
                        return self._refusal_result(
                            "I cannot provide a confident answer based on the available information.",
                            stt_ms, retrieval_ms, generation_ms, guardrail_ms, pipeline_start
                        )
                    
                    # Update answer to the retry version
                    answer = retry_answer
                    logger.info("  Retry successful - using new answer")
                    stage_details['hallucination_check_retry'] = {
                        'retry_performed': True,
                        'retry_grounded': is_grounded,
                        'retry_score': details['final_score'] if details else None
                    }
                
                logger.info(f"  Answer grounded ({guard_time:.2f}ms)")
                stage_details['hallucination_check_status'] = 'enabled_passed'
            else:
                # Hallucination check disabled - pass answer through directly
                logger.info("Stage 6: Hallucination check SKIPPED (disabled via config flag)")
                logger.info("  ⚠ NLI model miscalibrated for Hindi - needs multilingual model swap")
                stage_details['hallucination_check_status'] = 'disabled'
                stage_details['hallucination_check'] = {
                    'grounded': None,
                    'details': None,
                    'reason': 'Check disabled - see HALLUCINATION_CHECK_ENABLED flag'
                }
            
            # Success!
            total_ms = (time.perf_counter() - pipeline_start) * 1000
            
            logger.info(f"Pipeline completed successfully ({total_ms:.2f}ms)")
            
            return PipelineResult(
                answer=answer,
                refused=False,
                refusal_reason=None,
                source_chunks=[r.text for r in retrieval_results],
                stt_ms=stt_ms,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                guardrail_ms=guardrail_ms,
                total_ms=total_ms,
                stage_details=stage_details
            )
        
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return self._error_result(str(e), pipeline_start)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
    async def _stage_transcribe(self, audio_file: str) -> tuple:
        """Stage 1: Transcribe audio."""
        # Debug logging - trace the audio file path through the pipeline
        logger.info(f"📍 Pipeline _stage_transcribe received: {audio_file}")
        logger.info(f"   File exists in pipeline: {os.path.exists(audio_file)}")
        if os.path.exists(audio_file):
            logger.info(f"   File size in pipeline: {os.path.getsize(audio_file):,} bytes")
        
        if self.stt_client is None:
            from stt.sarvam_client import get_stt_client
            self.stt_client = get_stt_client()
        
        start = time.perf_counter()
        result = self.stt_client.transcribe_with_retry(audio_file)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        details = {
            'confidence': result.confidence,
            'language': result.language,
            'latency_ms': result.latency_ms
        }
        
        return result.text, elapsed_ms, details
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=2))
    async def _stage_retrieve(self, query: str) -> tuple:
        """Stage 3: Retrieve relevant chunks."""
        start = time.perf_counter()
        results, metrics = self.retriever.retrieve(
            query,
            top_k=self.config.top_k
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return results, metrics, elapsed_ms
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=2))
    async def _stage_generate(
        self,
        query: str,
        context_chunks: List[str],
        stricter: bool = False,
        query_language: Optional[str] = None
    ) -> tuple:
        """Stage 5: Generate answer."""
        start = time.perf_counter()
        
        result = self.generator.generate(
            query,
            context_chunks,
            max_context_length=self.config.max_context_length,
            query_language=query_language,
            top_k_chunks=5  # Limit to top 5 chunks for speed
        )
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        details = {
            'model': result.model,
            'ttft_ms': result.time_to_first_token_ms,
            'tokens': result.tokens_generated,
            'stricter': stricter,
            'query_language': query_language,
            'chunks_used': len(result.source_chunks)
        }
        
        return result.answer, elapsed_ms, details
    
    def _refusal_result(
        self,
        reason: str,
        stt_ms: float,
        retrieval_ms: float,
        generation_ms: float,
        guardrail_ms: float,
        pipeline_start: float
    ) -> PipelineResult:
        """Create a refusal result."""
        total_ms = (time.perf_counter() - pipeline_start) * 1000
        
        return PipelineResult(
            answer=None,
            refused=True,
            refusal_reason=reason,
            source_chunks=None,
            stt_ms=stt_ms,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            guardrail_ms=guardrail_ms,
            total_ms=total_ms,
            stage_details={}
        )
    
    def _error_result(self, error: str, pipeline_start: float) -> PipelineResult:
        """Create an error result."""
        total_ms = (time.perf_counter() - pipeline_start) * 1000
        
        return PipelineResult(
            answer=None,
            refused=True,
            refusal_reason=f"Pipeline error: {error}",
            source_chunks=None,
            stt_ms=0,
            retrieval_ms=0,
            generation_ms=0,
            guardrail_ms=0,
            total_ms=total_ms,
            stage_details={},
            error=error
        )


if __name__ == "__main__":
    # Test pipeline
    config = PipelineConfig(
        strategy_name="fixed",
        models_dir=str(Path(__file__).parent.parent / "models")
    )
    
    pipeline = RAGPipeline(config)
    
    async def test():
        result = await pipeline.run(query="What is machine learning?")
        
        print("\n" + "="*60)
        print("PIPELINE RESULT")
        print("="*60)
        print(f"Refused: {result.refused}")
        if result.refused:
            print(f"Reason: {result.refusal_reason}")
        else:
            print(f"Answer: {result.answer}")
        print(f"\nLatency Breakdown:")
        print(f"  STT: {result.stt_ms:.2f}ms")
        print(f"  Retrieval: {result.retrieval_ms:.2f}ms")
        print(f"  Generation: {result.generation_ms:.2f}ms")
        print(f"  Guardrails: {result.guardrail_ms:.2f}ms")
        print(f"  Total: {result.total_ms:.2f}ms")
    
    asyncio.run(test())
