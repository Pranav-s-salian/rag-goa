"""
Retrieval module with FAISS search, metadata filtering, and optional reranking.

PRODUCTION INDEX: Uses full 7.79M+ Hindi passage index from HF bucket (IVF-PQ).
This index will grow as more languages are added - do NOT hardcode vector counts.
"""
import os
import time
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from retrieval.bucket_loader import ProductionIndexLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: Dict
    rank: int


@dataclass
class RetrievalMetrics:
    """Timing metrics for retrieval stages."""
    embed_ms: float
    search_ms: float
    filter_ms: float
    rerank_ms: float
    total_ms: float


class Reranker:
    """Cross-encoder reranker for retrieved results."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2", device: str = "cpu"):
        """Initialize reranker model."""
        logger.info(f"Loading reranker model: {model_name}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        logger.info("Reranker loaded")
    
    def rerank(self, query: str, texts: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Rerank texts by relevance to query.
        
        Returns:
            List of (original_index, relevance_score) tuples, sorted by score descending
        """
        # Prepare pairs
        pairs = [[query, text] for text in texts]
        
        # Tokenize
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get scores
            scores = self.model(**inputs).logits.squeeze(-1).cpu().numpy()
        
        # Sort by score descending
        ranked_indices = np.argsort(scores)[::-1]
        results = [(int(idx), float(scores[idx])) for idx in ranked_indices[:top_k]]
        
        return results


class Retriever:
    """
    Retrieval orchestrator: embedding -> FAISS search -> metadata filter -> optional rerank.
    
    PRODUCTION: Uses HF bucket index (7.79M+ Hindi passages, IVF-PQ with nprobe=16).
    """
    
    def __init__(
        self,
        strategy_name: str,
        models_dir: Path,
        embedding_model,
        enable_rerank: bool = False,
        reranker_model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        use_production_index: bool = True,
        production_bucket: str = None,  # Must be provided - no default
        nprobe: int = 16
    ):
        """
        Initialize retriever.
        
        Args:
            strategy_name: Name of chunking strategy (ignored if use_production_index=True)
            models_dir: Directory containing indices
            embedding_model: Embedding model instance
            enable_rerank: Whether to use reranking
            reranker_model_name: Reranker model name
            use_production_index: Use production HF bucket index (default: True)
            production_bucket: HF bucket name for production index
            nprobe: IVF-PQ nprobe setting (only for production index)
        """
        self.strategy_name = strategy_name
        self.embedding_model = embedding_model
        self.enable_rerank = enable_rerank
        self.use_production_index = use_production_index
        
        # Load index and metadata
        if use_production_index:
            logger.info("="*70)
            logger.info("USING PRODUCTION INDEX FROM HF BUCKET")
            logger.info("="*70)
            
            # Load from HF bucket
            loader = ProductionIndexLoader(
                bucket_name=production_bucket,
                nprobe=nprobe
            )
            
            self.index, metadata_list = loader.load_index_and_metadata()
            
            # Convert metadata list to chunks format expected by retriever
            # Each metadata entry has: {"doc_id": ..., "text": ...}
            self.chunks = []
            for idx, meta in enumerate(metadata_list):
                self.chunks.append({
                    'chunk_id': f"chunk_{idx}",
                    'doc_id': meta.get('doc_id', f"doc_{idx}"),
                    'text': meta.get('text', ''),
                    'metadata': {}
                })
            
            logger.info(f"✓ Production index ready: {len(self.chunks):,} passages")
            
        else:
            # Legacy path: load from local files (for archived test indices)
            logger.info(f"Loading ARCHIVED test index for strategy: {strategy_name}")
            logger.info(f"  (This should only be used for chunking strategy comparison)")
            
            index_path = models_dir / f"{strategy_name}_index.faiss"
            metadata_path = models_dir / f"{strategy_name}_metadata.pkl"
            
            if not index_path.exists() or not metadata_path.exists():
                raise FileNotFoundError(
                    f"Index or metadata not found for strategy: {strategy_name}. "
                    f"Check {models_dir}/archive_small_test_indices/ for archived files."
                )
            
            self.index = faiss.read_index(str(index_path))
            
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            self.chunks = self.metadata['chunks']
            logger.info(f"Loaded test index with {len(self.chunks)} chunks")
        
        # Initialize reranker if enabled
        self.reranker = None
        if enable_rerank:
            self.reranker = Reranker(model_name=reranker_model_name)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        initial_k: int = 20,
        metadata_filters: Optional[Dict] = None
    ) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Query text
            top_k: Number of final results to return
            initial_k: Number of candidates to retrieve before reranking
            metadata_filters: Optional metadata filters (e.g., {'language': 'hindi'})
        
        Returns:
            Tuple of (results list, timing metrics)
        """
        metrics_start = time.perf_counter()
        
        # Stage 1: Embed query
        embed_start = time.perf_counter()
        query_embedding = self.embedding_model.embed(query, prefix="query: ")
        embed_ms = (time.perf_counter() - embed_start) * 1000
        
        # Stage 2: FAISS search
        search_start = time.perf_counter()
        search_k = initial_k if self.enable_rerank else top_k
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            search_k
        )
        search_ms = (time.perf_counter() - search_start) * 1000
        
        # Stage 3: Metadata filtering (with safe handling for out-of-range indices)
        filter_start = time.perf_counter()
        candidates = []
        out_of_range_count = 0
        
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            
            # SAFETY CHECK: Ensure index is within valid metadata range
            if idx >= len(self.chunks):
                out_of_range_count += 1
                continue  # Skip vectors without metadata
            
            chunk = self.chunks[idx]
            
            # Apply metadata filters
            if metadata_filters:
                if not self._matches_filters(chunk, metadata_filters):
                    continue
            
            candidates.append({
                'index': int(idx),
                'chunk': chunk,
                'score': float(distance),  # L2 distance from FAISS
            })
        
        # Log if any results were filtered due to missing metadata
        if out_of_range_count > 0:
            logger.debug(
                f"Filtered {out_of_range_count} results due to missing metadata "
                f"(vector IDs >= {len(self.chunks)})"
            )
        
        filter_ms = (time.perf_counter() - filter_start) * 1000
        
        # Stage 4: Optional reranking
        rerank_ms = 0.0
        if self.enable_rerank and self.reranker and len(candidates) > 0:
            rerank_start = time.perf_counter()
            
            # Extract texts for reranking
            texts = [c['chunk']['text'] for c in candidates]
            
            # Rerank
            reranked = self.reranker.rerank(query, texts, top_k=top_k)
            
            # Reorder candidates
            new_candidates = []
            for original_idx, rerank_score in reranked:
                candidate = candidates[original_idx].copy()
                candidate['score'] = rerank_score  # Replace with reranker score
                new_candidates.append(candidate)
            
            candidates = new_candidates
            rerank_ms = (time.perf_counter() - rerank_start) * 1000
        else:
            # No reranking: just take top_k
            # Convert L2 distance to similarity score (negative distance)
            for c in candidates:
                c['score'] = -c['score']
            candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:top_k]
        
        # Build results
        results = []
        for rank, candidate in enumerate(candidates):
            chunk = candidate['chunk']
            
            # Handle hierarchical strategy: use parent text if available
            text = chunk['text']
            if 'parent_text' in chunk.get('metadata', {}):
                parent_text = chunk['metadata']['parent_text']
                if parent_text:
                    text = parent_text  # Use parent for generation context
            
            # Handle sentence window: use context window
            if 'context_text' in chunk.get('metadata', {}):
                context_text = chunk['metadata']['context_text']
                if context_text:
                    text = context_text
            
            results.append(RetrievalResult(
                chunk_id=chunk['chunk_id'],
                doc_id=chunk['doc_id'],
                text=text,
                score=candidate['score'],
                metadata=chunk.get('metadata', {}),
                rank=rank
            ))
        
        # Calculate total time
        total_ms = (time.perf_counter() - metrics_start) * 1000
        
        metrics = RetrievalMetrics(
            embed_ms=embed_ms,
            search_ms=search_ms,
            filter_ms=filter_ms,
            rerank_ms=rerank_ms,
            total_ms=total_ms
        )
        
        return results, metrics
    
    def _matches_filters(self, chunk: Dict, filters: Dict) -> bool:
        """Check if chunk metadata matches all filters."""
        chunk_metadata = chunk.get('metadata', {})
        
        for key, value in filters.items():
            chunk_value = chunk_metadata.get(key)
            
            if isinstance(value, list):
                # List filter: check if chunk value in list
                if chunk_value not in value:
                    return False
            else:
                # Exact match
                if chunk_value != value:
                    return False
        
        return True


if __name__ == "__main__":
    # Test retrieval
    from embeddings.embed import get_model
    
    models_dir = Path(__file__).parent.parent / "models"
    embedding_model = get_model()
    
    # Test with fixed strategy
    retriever = Retriever(
        strategy_name="fixed",
        models_dir=models_dir,
        embedding_model=embedding_model,
        enable_rerank=False
    )
    
    test_query = "What is machine learning?"
    results, metrics = retriever.retrieve(test_query, top_k=5)
    
    logger.info(f"\nQuery: {test_query}")
    logger.info(f"Retrieved {len(results)} results")
    logger.info(f"Metrics: {metrics}")
    
    for i, result in enumerate(results[:3]):
        logger.info(f"\nResult {i+1}:")
        logger.info(f"  Score: {result.score:.4f}")
        logger.info(f"  Text: {result.text[:100]}...")
