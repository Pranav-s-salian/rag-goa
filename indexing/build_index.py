"""
Build FAISS HNSW indices for all chunking strategies.
"""
import json
import time
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from chunking import fixed, semantic, sentence_window, hierarchical, metadata_tagged, Chunk
from embeddings.embed import get_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_index_for_strategy(
    strategy_name: str,
    chunks: List[Chunk],
    embedding_model,
    output_dir: Path,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200
) -> Tuple[faiss.Index, float, Dict]:
    """
    Build and save FAISS HNSW index for a chunking strategy.
    
    Args:
        strategy_name: Name of the chunking strategy
        chunks: List of chunks to index
        embedding_model: Embedding model instance
        output_dir: Directory to save index and metadata
        hnsw_m: HNSW M parameter (links per node)
        hnsw_ef_construction: HNSW ef_construction parameter
    
    Returns:
        Tuple of (index, build_time_seconds, metadata)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Building index for strategy: {strategy_name}")
    logger.info(f"Number of chunks: {len(chunks)}")
    
    start_time = time.perf_counter()
    
    # Extract texts for embedding
    texts = [chunk.text for chunk in chunks]
    
    # Embed all chunks (batch processing for efficiency)
    logger.info("Embedding chunks...")
    embed_start = time.perf_counter()
    embeddings = embedding_model.embed_batch(
        texts,
        prefix="passage: ",
        batch_size=32,
        show_progress=True
    )
    embed_time = time.perf_counter() - embed_start
    logger.info(f"Embedding completed in {embed_time:.2f}s ({embed_time/len(texts)*1000:.2f}ms per chunk)")
    
    # Build FAISS index
    logger.info("Building FAISS HNSW index...")
    dimension = embeddings.shape[1]
    
    # IndexHNSWFlat: HNSW graph structure with flat (exact) distance computation
    index = faiss.IndexHNSWFlat(dimension, hnsw_m)
    index.hnsw.efConstruction = hnsw_ef_construction
    
    # Add vectors to index
    index_start = time.perf_counter()
    index.add(embeddings.astype(np.float32))
    index_time = time.perf_counter() - index_start
    logger.info(f"Index built in {index_time:.2f}s")
    
    total_time = time.perf_counter() - start_time
    
    # Prepare metadata
    metadata = {
        'strategy_name': strategy_name,
        'num_chunks': len(chunks),
        'dimension': dimension,
        'hnsw_m': hnsw_m,
        'hnsw_ef_construction': hnsw_ef_construction,
        'build_time_seconds': total_time,
        'embed_time_seconds': embed_time,
        'index_time_seconds': index_time,
        'chunks': [chunk.dict() for chunk in chunks]
    }
    
    # Save index
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / f"{strategy_name}_index.faiss"
    faiss.write_index(index, str(index_path))
    logger.info(f"Index saved to {index_path}")
    
    # Save metadata and chunks
    metadata_path = output_dir / f"{strategy_name}_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"Metadata saved to {metadata_path}")
    
    # Save human-readable summary
    summary_path = output_dir / f"{strategy_name}_summary.json"
    summary = {
        'strategy_name': strategy_name,
        'num_chunks': len(chunks),
        'dimension': dimension,
        'build_time_seconds': total_time,
        'avg_chunk_length': np.mean([len(c.text) for c in chunks]),
        'sample_chunks': [chunks[i].text[:100] + "..." for i in range(min(3, len(chunks)))]
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Strategy '{strategy_name}' indexing complete!")
    logger.info(f"Total time: {total_time:.2f}s")
    
    return index, total_time, metadata


def main():
    """Build indices for all chunking strategies."""
    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir = Path(__file__).parent.parent / "models"
    
    # Load passages
    passages_file = data_dir / "passages.json"
    if not passages_file.exists():
        logger.error(f"Passages file not found: {passages_file}")
        logger.info("Please run data/download_dataset.py first")
        return
    
    with open(passages_file, 'r', encoding='utf-8') as f:
        passages = json.load(f)
    
    logger.info(f"Loaded {len(passages)} passages from {passages_file}")
    
    # Initialize embedding model
    logger.info("Loading embedding model...")
    embedding_model = get_model(use_onnx=True, use_gpu=False)
    
    # Define strategies
    strategies = {
        'fixed': lambda: fixed.chunk(passages, chunk_size=256, overlap=51),
        'semantic': lambda: semantic.chunk(passages, embed_fn=embedding_model.embed, similarity_threshold=0.75),
        'sentence_window': lambda: sentence_window.chunk(passages, window_size=2),
        'hierarchical': lambda: hierarchical.chunk(passages, child_size=128, parent_size=512),
        'metadata_tagged': lambda: metadata_tagged.chunk(passages, chunk_size=256, overlap=51)
    }
    
    # Build index for each strategy
    results = {}
    
    for strategy_name, chunk_fn in strategies.items():
        try:
            # Generate chunks
            logger.info(f"\nGenerating chunks for strategy: {strategy_name}")
            chunks = chunk_fn()
            logger.info(f"Generated {len(chunks)} chunks")
            
            # Build index
            index, build_time, metadata = build_index_for_strategy(
                strategy_name=strategy_name,
                chunks=chunks,
                embedding_model=embedding_model,
                output_dir=output_dir
            )
            
            results[strategy_name] = {
                'num_chunks': len(chunks),
                'build_time': build_time
            }
            
        except Exception as e:
            logger.error(f"Error building index for {strategy_name}: {e}", exc_info=True)
            results[strategy_name] = {'error': str(e)}
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("INDEX BUILD SUMMARY")
    logger.info("="*60)
    
    for strategy_name, result in results.items():
        if 'error' in result:
            logger.info(f"{strategy_name:20s}: ERROR - {result['error']}")
        else:
            logger.info(
                f"{strategy_name:20s}: {result['num_chunks']:6d} chunks, "
                f"{result['build_time']:6.2f}s build time"
            )
    
    logger.info("\nAll indices built successfully!")
    logger.info(f"Indices saved to: {output_dir}")


if __name__ == "__main__":
    main()
