"""
ONNX-optimized embedding using intfloat/multilingual-e5-small.
"""
import time
import logging
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os

# Try to import ONNX runtime (optional for speed optimization)
try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("optimum.onnxruntime not available. Install with: pip install optimum[onnxruntime]")
    logging.warning("Falling back to PyTorch (slightly slower but works fine)")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    ONNX-optimized multilingual embedding model.
    """
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        use_onnx: bool = True,
        use_gpu: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model name
            use_onnx: Whether to use ONNX optimization (if available)
            use_gpu: Whether to use GPU (if available)
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        # Force use_onnx to False if not available
        self.use_onnx = use_onnx and ONNX_AVAILABLE
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "voice-rag")
        
        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Device: {self.device}, ONNX: {self.use_onnx}")
        
        if use_onnx and not ONNX_AVAILABLE:
            logger.warning("ONNX requested but not available. Using PyTorch instead.")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir
        )
        
        # Load model
        if self.use_onnx:
            self._load_onnx_model()
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            self.model.eval()
        
        logger.info("Embedding model loaded successfully")
        
        # Benchmark on initialization
        self._benchmark()
    
    def _load_onnx_model(self):
        """Load or export ONNX model."""
        if not ONNX_AVAILABLE:
            logger.warning("ONNX not available, falling back to PyTorch")
            self.use_onnx = False
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            self.model.eval()
            return
        
        onnx_path = Path(self.cache_dir) / "onnx" / self.model_name.replace("/", "_")
        
        try:
            if onnx_path.exists():
                logger.info(f"Loading existing ONNX model from {onnx_path}")
                self.model = ORTModelForFeatureExtraction.from_pretrained(
                    str(onnx_path),
                    provider="CPUExecutionProvider" if self.device == "cpu" else "CUDAExecutionProvider"
                )
            else:
                logger.info("Exporting model to ONNX format...")
                onnx_path.mkdir(parents=True, exist_ok=True)
                
                # Export to ONNX
                model = ORTModelForFeatureExtraction.from_pretrained(
                    self.model_name,
                    export=True,
                    cache_dir=self.cache_dir,
                    provider="CPUExecutionProvider" if self.device == "cpu" else "CUDAExecutionProvider"
                )
                model.save_pretrained(str(onnx_path))
                self.model = model
                logger.info(f"ONNX model saved to {onnx_path}")
        except Exception as e:
            logger.warning(f"Failed to load ONNX model: {e}. Falling back to PyTorch.")
            self.use_onnx = False
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            self.model.eval()
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embedding."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def embed(self, text: str, prefix: str = "query: ") -> np.ndarray:
        """
        Embed a single text.
        
        Args:
            text: Input text
            prefix: Prefix for the text (e.g., "query: " or "passage: ")
        
        Returns:
            Normalized embedding vector
        """
        return self.embed_batch([text], prefix=prefix)[0]
    
    def embed_batch(
        self,
        texts: List[str],
        prefix: str = "query: ",
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of input texts
            prefix: Prefix for texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress
        
        Returns:
            Array of normalized embedding vectors
        """
        # Add prefix (E5 models require task-specific prefixes)
        prefixed_texts = [f"{prefix}{text}" for text in texts]
        
        all_embeddings = []
        
        for i in range(0, len(prefixed_texts), batch_size):
            batch_texts = prefixed_texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            if not self.use_onnx:
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get embeddings
            with torch.no_grad():
                if self.use_onnx:
                    outputs = self.model(**encoded)
                    embeddings = self._mean_pooling(outputs, encoded['attention_mask'])
                else:
                    outputs = self.model(**encoded)
                    embeddings = self._mean_pooling(outputs, encoded['attention_mask'])
            
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
            
            if show_progress and i % (batch_size * 10) == 0:
                logger.info(f"Embedded {i + len(batch_texts)}/{len(texts)} texts")
        
        return np.vstack(all_embeddings)
    
    def _benchmark(self):
        """Benchmark embedding latency."""
        test_texts = [
            "This is a test sentence for benchmarking.",
            "Machine learning models require efficient inference.",
            "Natural language processing enables semantic understanding."
        ]
        
        # Warmup
        _ = self.embed_batch(test_texts)
        
        # Benchmark single embedding
        start = time.perf_counter()
        _ = self.embed(test_texts[0])
        single_latency = (time.perf_counter() - start) * 1000
        
        # Benchmark batch embedding
        start = time.perf_counter()
        _ = self.embed_batch(test_texts)
        batch_latency = (time.perf_counter() - start) * 1000
        per_text_latency = batch_latency / len(test_texts)
        
        logger.info(f"Embedding benchmark:")
        logger.info(f"  Single text: {single_latency:.2f}ms")
        logger.info(f"  Batch ({len(test_texts)} texts): {batch_latency:.2f}ms ({per_text_latency:.2f}ms per text)")


# Global model instance (singleton pattern)
_model_instance: Optional[EmbeddingModel] = None


def get_model(
    model_name: str = "intfloat/multilingual-e5-small",
    use_onnx: bool = True,
    use_gpu: bool = False,
    force_reload: bool = False
) -> EmbeddingModel:
    """Get or create global embedding model instance."""
    global _model_instance
    
    if _model_instance is None or force_reload:
        _model_instance = EmbeddingModel(
            model_name=model_name,
            use_onnx=use_onnx,
            use_gpu=use_gpu
        )
    
    return _model_instance


def embed(text: str, prefix: str = "query: ") -> np.ndarray:
    """Convenience function for single text embedding."""
    model = get_model()
    return model.embed(text, prefix=prefix)


def embed_batch(
    texts: List[str],
    prefix: str = "query: ",
    batch_size: int = 32
) -> np.ndarray:
    """Convenience function for batch embedding."""
    model = get_model()
    return model.embed_batch(texts, prefix=prefix, batch_size=batch_size)


if __name__ == "__main__":
    # Test embedding
    model = get_model()
    
    test_text = "What is machine learning?"
    embedding = model.embed(test_text)
    logger.info(f"Embedding shape: {embedding.shape}")
    logger.info(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
