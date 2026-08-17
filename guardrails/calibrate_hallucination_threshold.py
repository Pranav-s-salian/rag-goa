"""
Calibrate hallucination check threshold based on known-good and unrelated pairs.
Determines where grounded and ungrounded answers actually separate.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
import numpy as np
import matplotlib.pyplot as plt
from guardrails.hallucination_check import HallucinationChecker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Known-good grounded Q&A pairs (answer is entailed by context)
GROUNDED_PAIRS = [
    {
        'context': [
            "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "Deep learning uses neural networks with multiple layers to progressively extract higher-level features from raw input data."
        ],
        'answer': "Machine learning is a subset of AI that allows systems to learn from experience without explicit programming."
    },
    {
        'context': [
            "Natural language processing helps computers understand, interpret, and generate human language in a valuable way.",
            "NLP combines computational linguistics with statistical and machine learning techniques."
        ],
        'answer': "Natural language processing enables computers to understand and interpret human language using computational linguistics and machine learning."
    },
    {
        'context': [
            "Vector databases store and retrieve data based on similarity rather than exact matches, enabling semantic search.",
            "They use embedding vectors to represent complex data and find similar items efficiently."
        ],
        'answer': "Vector databases enable semantic search by storing data as embedding vectors and retrieving based on similarity."
    },
    {
        'context': [
            "Information retrieval is the process of obtaining relevant information from a large repository based on user queries.",
            "Modern IR systems use relevance ranking algorithms to order search results by their likely usefulness."
        ],
        'answer': "Information retrieval obtains relevant information from large repositories using queries and relevance ranking."
    },
    {
        'context': [
            "Transformers are a type of neural network architecture that relies entirely on self-attention mechanisms.",
            "They have become the dominant architecture for NLP tasks since their introduction in 2017."
        ],
        'answer': "Transformers are neural networks that use self-attention and have become dominant in NLP since 2017."
    },
    {
        'context': [
            "BERT is a bidirectional encoder model that reads text in both directions to understand context better.",
            "It is pre-trained on large corpora and can be fine-tuned for specific downstream tasks."
        ],
        'answer': "BERT is a bidirectional encoder that pre-trains on large text and can be fine-tuned for specific tasks."
    },
    {
        'context': [
            "Embeddings are dense vector representations of words or sentences that capture semantic meaning.",
            "Similar concepts are mapped to nearby points in the embedding space."
        ],
        'answer': "Embeddings are vector representations that capture semantic meaning, with similar concepts mapped to nearby points."
    },
    {
        'context': [
            "Transfer learning allows models trained on one task to be adapted for related tasks with less data.",
            "This approach has dramatically reduced the data requirements for many machine learning applications."
        ],
        'answer': "Transfer learning enables models to adapt from one task to related tasks using less data."
    },
    {
        'context': [
            "Attention mechanisms allow models to focus on relevant parts of the input when making predictions.",
            "They compute weighted combinations of input representations based on learned importance scores."
        ],
        'answer': "Attention mechanisms help models focus on relevant input parts by computing weighted combinations based on importance."
    },
    {
        'context': [
            "Retrieval-Augmented Generation combines retrieval systems with language models to ground responses in factual knowledge.",
            "This approach reduces hallucination by providing the model with relevant context before generation."
        ],
        'answer': "RAG combines retrieval with language models to ground responses in facts and reduce hallucination."
    }
]


# Unrelated pairs (answer is NOT entailed by context - hallucination)
UNRELATED_PAIRS = [
    {
        'context': [
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "Deep learning uses neural networks with multiple layers."
        ],
        'answer': "The weather today is sunny and warm with temperatures reaching 25 degrees celsius."
    },
    {
        'context': [
            "Natural language processing helps computers understand human language.",
            "NLP combines computational linguistics with machine learning."
        ],
        'answer': "The Eiffel Tower in Paris was completed in 1889 and stands 330 meters tall."
    },
    {
        'context': [
            "Vector databases store data based on similarity using embeddings.",
            "They enable efficient semantic search."
        ],
        'answer': "Shakespeare wrote Romeo and Juliet in the late 16th century during the Elizabethan era."
    },
    {
        'context': [
            "Information retrieval obtains relevant information from large repositories.",
            "Modern IR systems use relevance ranking algorithms."
        ],
        'answer': "Pizza originated in Naples, Italy and has become one of the world's most popular foods."
    },
    {
        'context': [
            "Transformers are neural networks that use self-attention mechanisms.",
            "They have become dominant in NLP tasks."
        ],
        'answer': "The Pacific Ocean is the largest ocean on Earth, covering more area than all land combined."
    },
    {
        'context': [
            "BERT is a bidirectional encoder model for language understanding.",
            "It can be fine-tuned for downstream tasks."
        ],
        'answer': "The pyramids of Giza were built as tombs for Egyptian pharaohs around 2500 BC."
    },
    {
        'context': [
            "Embeddings are vector representations that capture semantic meaning.",
            "Similar concepts are mapped to nearby points."
        ],
        'answer': "Mount Everest is the highest mountain on Earth at 8,849 meters above sea level."
    },
    {
        'context': [
            "Transfer learning adapts models from one task to related tasks.",
            "This reduces data requirements significantly."
        ],
        'answer': "The Beatles were a British rock band formed in Liverpool in 1960."
    },
    {
        'context': [
            "Attention mechanisms help models focus on relevant input parts.",
            "They compute weighted combinations based on importance."
        ],
        'answer': "The Amazon rainforest produces 20% of the world's oxygen and is home to countless species."
    },
    {
        'context': [
            "RAG combines retrieval with language models for factual responses.",
            "This approach reduces hallucination."
        ],
        'answer': "The speed of light in vacuum is approximately 299,792 kilometers per second."
    }
]


def calibrate_threshold(aggregation_method='average'):
    """
    Run hallucination checker on known pairs and determine optimal threshold.
    
    Args:
        aggregation_method: 'average' or 'min' for sentence score aggregation
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Calibrating Hallucination Check Threshold")
    logger.info(f"Aggregation method: {aggregation_method}")
    logger.info(f"{'='*60}\n")
    
    # Initialize checker with default threshold (will recalibrate)
    checker = HallucinationChecker(
        entailment_threshold=0.5,  # Temporary, will find optimal
        aggregation_method=aggregation_method
    )
    
    # Collect scores for grounded pairs
    logger.info("Testing GROUNDED pairs (should have HIGH scores)...")
    grounded_scores = []
    
    for i, pair in enumerate(GROUNDED_PAIRS):
        is_grounded, reason, details = checker.check(
            generated_answer=pair['answer'],
            context_chunks=pair['context'],
            return_details=True
        )
        score = details['final_score']
        grounded_scores.append(score)
        logger.info(f"  Grounded {i+1}: score={score:.3f} | {pair['answer'][:60]}...")
    
    # Collect scores for unrelated pairs
    logger.info("\nTesting UNRELATED pairs (should have LOW scores)...")
    unrelated_scores = []
    
    for i, pair in enumerate(UNRELATED_PAIRS):
        is_grounded, reason, details = checker.check(
            generated_answer=pair['answer'],
            context_chunks=pair['context'],
            return_details=True
        )
        score = details['final_score']
        unrelated_scores.append(score)
        logger.info(f"  Unrelated {i+1}: score={score:.3f} | {pair['answer'][:60]}...")
    
    # Analyze distributions
    logger.info(f"\n{'='*60}")
    logger.info("Score Distributions")
    logger.info(f"{'='*60}")
    
    logger.info(f"\nGROUNDED pairs (n={len(grounded_scores)}):")
    logger.info(f"  Mean:   {np.mean(grounded_scores):.3f}")
    logger.info(f"  Median: {np.median(grounded_scores):.3f}")
    logger.info(f"  Min:    {np.min(grounded_scores):.3f}")
    logger.info(f"  Max:    {np.max(grounded_scores):.3f}")
    logger.info(f"  Std:    {np.std(grounded_scores):.3f}")
    
    logger.info(f"\nUNRELATED pairs (n={len(unrelated_scores)}):")
    logger.info(f"  Mean:   {np.mean(unrelated_scores):.3f}")
    logger.info(f"  Median: {np.median(unrelated_scores):.3f}")
    logger.info(f"  Min:    {np.min(unrelated_scores):.3f}")
    logger.info(f"  Max:    {np.max(unrelated_scores):.3f}")
    logger.info(f"  Std:    {np.std(unrelated_scores):.3f}")
    
    # Find optimal threshold (midpoint between distributions)
    grounded_min = np.min(grounded_scores)
    unrelated_max = np.max(unrelated_scores)
    
    # Optimal threshold: midpoint between clusters
    optimal_threshold = (grounded_min + unrelated_max) / 2
    
    # Alternative: use mean of medians
    alternative_threshold = (np.median(grounded_scores) + np.median(unrelated_scores)) / 2
    
    logger.info(f"\n{'='*60}")
    logger.info("Recommended Thresholds")
    logger.info(f"{'='*60}")
    logger.info(f"Conservative (midpoint min/max): {optimal_threshold:.3f}")
    logger.info(f"Balanced (midpoint medians):    {alternative_threshold:.3f}")
    logger.info(f"Current default:                 0.500")
    
    # Calculate separation
    gap = grounded_min - unrelated_max
    logger.info(f"\nCluster separation:")
    logger.info(f"  Gap: {gap:.3f} ({gap/optimal_threshold*100:.1f}% of threshold)")
    if gap > 0:
        logger.info(f"  ✓ Clean separation (grounded_min > unrelated_max)")
    else:
        logger.info(f"  ⚠ Overlap detected (grounded_min < unrelated_max)")
    
    # Test different thresholds
    logger.info(f"\n{'='*60}")
    logger.info("Threshold Performance")
    logger.info(f"{'='*60}")
    
    test_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, optimal_threshold, alternative_threshold]
    test_thresholds = sorted(set([round(t, 3) for t in test_thresholds]))
    
    logger.info(f"{'Threshold':<12} {'Grounded OK':<15} {'Unrelated OK':<15} {'Accuracy'}")
    logger.info("-" * 60)
    
    for threshold in test_thresholds:
        grounded_correct = sum(1 for s in grounded_scores if s >= threshold)
        unrelated_correct = sum(1 for s in unrelated_scores if s < threshold)
        total_correct = grounded_correct + unrelated_correct
        total = len(grounded_scores) + len(unrelated_scores)
        accuracy = total_correct / total
        
        logger.info(
            f"{threshold:<12.3f} {grounded_correct}/{len(grounded_scores):<14} "
            f"{unrelated_correct}/{len(unrelated_scores):<14} {accuracy:.1%}"
        )
    
    # Plot distributions
    try:
        plt.figure(figsize=(10, 6))
        
        plt.hist(grounded_scores, bins=15, alpha=0.6, label='Grounded', color='green')
        plt.hist(unrelated_scores, bins=15, alpha=0.6, label='Unrelated', color='red')
        
        plt.axvline(optimal_threshold, color='blue', linestyle='--', label=f'Optimal: {optimal_threshold:.3f}')
        plt.axvline(0.5, color='gray', linestyle=':', label='Default: 0.500')
        
        plt.xlabel('Entailment Score')
        plt.ylabel('Frequency')
        plt.title(f'Hallucination Score Distributions ({aggregation_method} aggregation)')
        plt.legend()
        plt.grid(alpha=0.3)
        
        output_path = Path(__file__).parent / f'hallucination_calibration_{aggregation_method}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"\nPlot saved to: {output_path}")
        
    except Exception as e:
        logger.warning(f"Could not generate plot: {e}")
    
    # Final recommendation
    logger.info(f"\n{'='*60}")
    logger.info("RECOMMENDATION")
    logger.info(f"{'='*60}")
    logger.info(f"Set entailment_threshold = {optimal_threshold:.3f}")
    logger.info(f"This provides the best separation between grounded and ungrounded answers.")
    logger.info(f"\nUpdate in guardrails/hallucination_check.py:")
    logger.info(f"  entailment_threshold={optimal_threshold:.3f}")
    logger.info(f"Or in .env:")
    logger.info(f"  HALLUCINATION_THRESHOLD={optimal_threshold:.3f}")
    logger.info(f"{'='*60}\n")
    
    return {
        'optimal_threshold': optimal_threshold,
        'alternative_threshold': alternative_threshold,
        'grounded_scores': grounded_scores,
        'unrelated_scores': unrelated_scores,
        'aggregation_method': aggregation_method
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calibrate hallucination check threshold')
    parser.add_argument(
        '--aggregation',
        type=str,
        default='average',
        choices=['average', 'min'],
        help='Sentence score aggregation method (default: average)'
    )
    
    args = parser.parse_args()
    
    results = calibrate_threshold(aggregation_method=args.aggregation)
    
    # Also test the other aggregation method for comparison
    if args.aggregation == 'average':
        logger.info("\n\nNow testing with 'min' aggregation for comparison...\n")
        results_min = calibrate_threshold(aggregation_method='min')
