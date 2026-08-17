"""
Evaluate and compare all chunking strategies using MSMARCO-XI queries.
Compute recall@k and average retrieval latency for each strategy.
"""
import json
import time
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
import faiss
import sys

sys.path.append(str(Path(__file__).parent.parent))

from embeddings.embed import get_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_index_and_metadata(strategy_name: str, models_dir: Path) -> Tuple[faiss.Index, Dict]:
    """Load FAISS index and metadata for a strategy."""
    index_path = models_dir / f"{strategy_name}_index.faiss"
    metadata_path = models_dir / f"{strategy_name}_metadata.pkl"
    
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Index or metadata not found for strategy: {strategy_name}")
    
    index = faiss.read_index(str(index_path))
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return index, metadata


def compute_recall_at_k(
    retrieved_doc_ids: List[str],
    relevant_doc_ids: List[str],
    k: int
) -> float:
    """
    Compute recall@k: fraction of relevant documents retrieved in top-k.
    """
    if not relevant_doc_ids:
        return 0.0
    
    retrieved_set = set(retrieved_doc_ids[:k])
    relevant_set = set(relevant_doc_ids)
    
    intersection = retrieved_set.intersection(relevant_set)
    recall = len(intersection) / len(relevant_set)
    
    return recall


def evaluate_strategy(
    strategy_name: str,
    index: faiss.Index,
    metadata: Dict,
    queries: List[Dict],
    embedding_model,
    k_values: List[int] = [5, 10]
) -> Dict:
    """
    Evaluate a single strategy on the query set.
    
    Returns:
        Dictionary with recall@k scores and latency statistics
    """
    logger.info(f"\nEvaluating strategy: {strategy_name}")
    
    # Extract chunk information
    chunks = metadata['chunks']
    chunk_to_doc = {i: chunk['doc_id'] for i, chunk in enumerate(chunks)}
    
    # Metrics
    recall_scores = {k: [] for k in k_values}
    latencies = []
    
    for i, query in enumerate(queries):
        query_text = query['text']
        relevant_doc_ids = query.get('relevant_passage_ids', [])
        
        # Embed query
        embed_start = time.perf_counter()
        query_embedding = embedding_model.embed(query_text, prefix="query: ")
        embed_time = (time.perf_counter() - embed_start) * 1000
        
        # Search index
        search_start = time.perf_counter()
        max_k = max(k_values)
        distances, indices = index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            max_k
        )
        search_time = (time.perf_counter() - search_start) * 1000
        
        # Total retrieval latency
        total_latency = embed_time + search_time
        latencies.append(total_latency)
        
        # Map retrieved indices to document IDs
        retrieved_chunk_indices = indices[0].tolist()
        retrieved_doc_ids = [
            chunk_to_doc.get(idx, 'unknown')
            for idx in retrieved_chunk_indices
            if idx != -1
        ]
        
        # Compute recall@k for each k
        for k in k_values:
            recall = compute_recall_at_k(retrieved_doc_ids, relevant_doc_ids, k)
            recall_scores[k].append(recall)
        
        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{len(queries)} queries")
    
    # Aggregate results
    results = {
        'strategy_name': strategy_name,
        'num_queries': len(queries),
        'recall': {
            f'recall@{k}': {
                'mean': float(np.mean(recall_scores[k])),
                'std': float(np.std(recall_scores[k])),
                'min': float(np.min(recall_scores[k])),
                'max': float(np.max(recall_scores[k]))
            }
            for k in k_values
        },
        'latency_ms': {
            'mean': float(np.mean(latencies)),
            'p50': float(np.percentile(latencies, 50)),
            'p70': float(np.percentile(latencies, 70)),
            'p95': float(np.percentile(latencies, 95)),
            'p100': float(np.max(latencies)),
            'min': float(np.min(latencies))
        }
    }
    
    logger.info(f"  Mean Recall@5: {results['recall']['recall@5']['mean']:.4f}")
    logger.info(f"  Mean Recall@10: {results['recall']['recall@10']['mean']:.4f}")
    logger.info(f"  P50 Latency: {results['latency_ms']['p50']:.2f}ms")
    logger.info(f"  P95 Latency: {results['latency_ms']['p95']:.2f}ms")
    
    return results


def main():
    """Run evaluation on all strategies."""
    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    models_dir = Path(__file__).parent.parent / "models"
    output_dir = Path(__file__).parent.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load queries
    queries_file = data_dir / "queries.json"
    if not queries_file.exists():
        logger.error(f"Queries file not found: {queries_file}")
        return
    
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    logger.info(f"Loaded {len(queries)} queries from {queries_file}")
    
    # Load embedding model
    logger.info("Loading embedding model...")
    embedding_model = get_model(use_onnx=True, use_gpu=False)
    
    # Strategies to evaluate
    strategies = ['fixed', 'semantic', 'sentence_window', 'hierarchical_child', 'metadata_tagged']
    
    all_results = {}
    
    for strategy_name in strategies:
        try:
            # Load index
            index, metadata = load_index_and_metadata(strategy_name, models_dir)
            
            # Evaluate
            results = evaluate_strategy(
                strategy_name=strategy_name,
                index=index,
                metadata=metadata,
                queries=queries,
                embedding_model=embedding_model,
                k_values=[5, 10]
            )
            
            all_results[strategy_name] = results
            
        except FileNotFoundError as e:
            logger.warning(f"Skipping {strategy_name}: {e}")
        except Exception as e:
            logger.error(f"Error evaluating {strategy_name}: {e}", exc_info=True)
    
    # Save detailed results
    results_file = output_dir / "strategy_comparison.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nDetailed results saved to {results_file}")
    
    # Generate comparison table (Markdown)
    table_file = output_dir / "strategy_comparison_table.md"
    with open(table_file, 'w', encoding='utf-8') as f:
        f.write("# Chunking Strategy Comparison\n\n")
        f.write("## Retrieval Performance Metrics\n\n")
        
        # Table header
        f.write("| Strategy | Recall@5 | Recall@10 | P50 Latency (ms) | P95 Latency (ms) | P100 Latency (ms) |\n")
        f.write("|----------|----------|-----------|------------------|------------------|-------------------|\n")
        
        # Table rows
        for strategy_name, results in all_results.items():
            recall5 = results['recall']['recall@5']['mean']
            recall10 = results['recall']['recall@10']['mean']
            p50 = results['latency_ms']['p50']
            p95 = results['latency_ms']['p95']
            p100 = results['latency_ms']['p100']
            
            f.write(f"| {strategy_name:20s} | {recall5:.4f} | {recall10:.4f} | {p50:8.2f} | {p95:8.2f} | {p100:8.2f} |\n")
        
        f.write("\n## Interpretation\n\n")
        f.write("- **Recall@k**: Fraction of relevant documents retrieved in top-k results (higher is better)\n")
        f.write("- **P50/P95/P100 Latency**: 50th/95th/100th percentile retrieval latency in milliseconds (lower is better)\n")
        f.write("- **Target**: Retrieval latency < 100ms (ideally < 50ms)\n\n")
        
        # Find best strategy
        if all_results:
            best_recall5 = max(all_results.items(), key=lambda x: x[1]['recall']['recall@5']['mean'])
            best_latency = min(all_results.items(), key=lambda x: x[1]['latency_ms']['p50'])
            
            f.write("## Recommendations\n\n")
            f.write(f"- **Best Recall@5**: {best_recall5[0]} ({best_recall5[1]['recall']['recall@5']['mean']:.4f})\n")
            f.write(f"- **Best Latency**: {best_latency[0]} ({best_latency[1]['latency_ms']['p50']:.2f}ms P50)\n")
    
    logger.info(f"Comparison table saved to {table_file}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("STRATEGY COMPARISON SUMMARY")
    logger.info("="*80)
    logger.info(f"{'Strategy':<20} {'Recall@5':>10} {'Recall@10':>10} {'P50 Latency':>15} {'Target Met':>12}")
    logger.info("-"*80)
    
    for strategy_name, results in all_results.items():
        recall5 = results['recall']['recall@5']['mean']
        recall10 = results['recall']['recall@10']['mean']
        p50 = results['latency_ms']['p50']
        target_met = "✓" if p50 < 100 else "✗"
        
        logger.info(f"{strategy_name:<20} {recall5:>10.4f} {recall10:>10.4f} {p50:>12.2f}ms {target_met:>12}")
    
    logger.info("="*80)


if __name__ == "__main__":
    main()
