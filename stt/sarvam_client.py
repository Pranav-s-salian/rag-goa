"""
Sarvam AI Speech-to-Text using Realtime WebSocket API.
"""
import time
import logging
import os
import asyncio
from typing import Optional, Tuple
from dataclasses import dataclass
from sarvamai import SarvamAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result from transcription operation."""
    text: str
    confidence: Optional[float]
    language: Optional[str]
    latency_ms: float


class SarvamSTTClient:
    """
    Sarvam AI STT client using saaras:v3 model in Fast mode.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "saaras:v3",
        mode: str = "fast"
    ):
        """
        Initialize Sarvam STT client.
        
        Args:
            api_key: Sarvam API key (defaults to SARVAM_API_KEY env var)
            model: Model name (fixed to saaras:v3)
            mode: Processing mode (fixed to "fast")
        """
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY must be provided or set in environment")
        
        self.model = model
        self.mode = mode
        
        # Initialize Sarvam client (note: parameter is api_subscription_key, not api_key)
        self.client = SarvamAI(api_subscription_key=self.api_key)
        logger.info(f"Sarvam STT client initialized (model: {model}, mode: {mode})")
    
    def transcribe_file(
        self,
        audio_file_path: str,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio from file.
        
        Args:
            audio_file_path: Path to audio file (string path, not file object)
            language: Optional language hint (e.g., 'hi-IN', 'en-IN')
        
        Returns:
            TranscriptionResult with transcript and metrics
        """
        overall_start = time.perf_counter()
        
        try:
            # Debug logging - verify the file before calling Sarvam API
            logger.info(f"🎯 Sarvam API call with file: {audio_file_path}")
            logger.info(f"   File exists: {os.path.exists(audio_file_path)}")
            
            if os.path.exists(audio_file_path):
                file_size = os.path.getsize(audio_file_path)
                logger.info(f"   File size: {file_size:,} bytes")
                
                # Quick WAV validation
                try:
                    import wave
                    with wave.open(audio_file_path, 'rb') as wav:
                        logger.info(f"   WAV format: {wav.getnchannels()}ch, {wav.getframerate()}Hz, {wav.getnframes()} frames")
                except Exception as wav_err:
                    logger.warning(f"   Could not read as WAV: {wav_err}")
            else:
                logger.error(f"   ❌ FILE DOES NOT EXIST!")
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            # Use Sarvam's speech-to-text API
            # Note: speech_to_text is a client object with transcribe() method
            # The file parameter requires an open binary file object, not a path string
            logger.info(f"   Calling Sarvam API (model={self.model}, mode=transcribe)...")
            
            # GRANULAR TIMING: Track each phase separately
            file_open_start = time.perf_counter()
            with open(audio_file_path, 'rb') as audio_file:
                file_open_ms = (time.perf_counter() - file_open_start) * 1000
                
                # Track API call separately (includes connection + upload + processing + response)
                api_call_start = time.perf_counter()
                response = self.client.speech_to_text.transcribe(
                    file=audio_file,  # Binary file object, not path string
                    model=self.model,  # 'saaras:v3'
                    mode="transcribe"  # Mode: transcribe, translate, verbatim, translit, codemix
                    # Note: language_code is auto-detected, not explicitly set
                )
                api_call_ms = (time.perf_counter() - api_call_start) * 1000
            
            overall_ms = (time.perf_counter() - overall_start) * 1000
            
            # Extract fields from response object
            # Response has: transcript, timestamps, language_code, language_probability, request_id
            transcript = response.transcript
            detected_language = response.language_code if hasattr(response, 'language_code') else None
            confidence = response.language_probability if hasattr(response, 'language_probability') else None
            
            logger.info(f"✓ Transcription completed in {overall_ms:.0f}ms")
            logger.info(f"  📊 TIMING BREAKDOWN:")
            logger.info(f"     File open: {file_open_ms:.1f}ms")
            logger.info(f"     API call (connection+upload+processing+response): {api_call_ms:.1f}ms")
            logger.info(f"     Total: {overall_ms:.1f}ms")
            logger.info(f"  Transcript: '{transcript[:100]}{'...' if len(transcript) > 100 else ''}'")
            logger.info(f"  Language: {detected_language}, Confidence: {confidence}")
            
            return TranscriptionResult(
                text=transcript,
                confidence=confidence,
                language=detected_language,
                latency_ms=overall_ms
            )
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            logger.error(f"   File path attempted: {audio_file_path}")
            logger.error(f"   Error type: {type(e).__name__}")
            raise
    
    async def transcribe_stream(
        self,
        audio_stream,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio from stream using WebSocket API.
        
        Args:
            audio_stream: Audio stream (async iterator of audio chunks)
            language: Optional language hint
        
        Returns:
            TranscriptionResult with transcript and metrics
        """
        start_time = time.perf_counter()
        
        try:
            # Note: Actual WebSocket streaming implementation would go here
            # For now, this is a placeholder that shows the structure
            
            # In a real implementation, you would:
            # 1. Establish WebSocket connection
            # 2. Stream audio chunks
            # 3. Receive transcription results in real-time
            # 4. Accumulate final transcript
            
            logger.warning("WebSocket streaming not fully implemented - using file-based fallback")
            
            # Placeholder: accumulate stream to temp file and use file-based transcription
            # In production, implement proper WebSocket streaming
            
            raise NotImplementedError("Streaming transcription requires WebSocket implementation")
            
        except Exception as e:
            logger.error(f"Stream transcription failed: {e}")
            raise
    
    def transcribe_with_retry(
        self,
        audio_file_path: str,
        max_retries: int = 2,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe with retry logic and exponential backoff.
        
        Args:
            audio_file_path: Path to audio file
            max_retries: Maximum number of retries
            language: Optional language hint
        
        Returns:
            TranscriptionResult
        """
        for attempt in range(max_retries + 1):
            try:
                result = self.transcribe_file(audio_file_path, language)
                return result
                
            except Exception as e:
                if attempt < max_retries:
                    backoff_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        f"Transcription attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {backoff_time}s..."
                    )
                    time.sleep(backoff_time)
                else:
                    logger.error(f"All transcription attempts failed: {e}")
                    raise


# Global instance
_stt_client_instance: Optional[SarvamSTTClient] = None


def get_stt_client(
    api_key: Optional[str] = None,
    force_reload: bool = False
) -> SarvamSTTClient:
    """Get or create global STT client instance."""
    global _stt_client_instance
    
    if _stt_client_instance is None or force_reload:
        _stt_client_instance = SarvamSTTClient(api_key=api_key)
    
    return _stt_client_instance


def transcribe(
    audio_file_path: str,
    language: Optional[str] = None
) -> TranscriptionResult:
    """Convenience function for transcription."""
    client = get_stt_client()
    return client.transcribe_with_retry(audio_file_path, language=language)


if __name__ == "__main__":
    # Test STT client (requires SARVAM_API_KEY and audio file)
    if not os.getenv("SARVAM_API_KEY"):
        logger.error("SARVAM_API_KEY not set. Cannot test STT client.")
    else:
        client = get_stt_client()
        
        # Test with a sample audio file (if available)
        test_audio = "test_audio.wav"
        if os.path.exists(test_audio):
            result = client.transcribe_file(test_audio)
            logger.info(f"\nTranscript: {result.text}")
            logger.info(f"Confidence: {result.confidence}")
            logger.info(f"Latency: {result.latency_ms:.2f}ms")
        else:
            logger.warning(f"Test audio file not found: {test_audio}")
