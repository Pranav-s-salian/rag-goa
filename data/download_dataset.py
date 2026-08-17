"""
Download and prepare IndicMSMARCO dataset for indexing and evaluation.

IndicMSMARCO Structure:
- Dataset: ai4bharat/IndicMSMARCO
- Size: 13.1MB total (manageable, no streaming needed)
- Schema: Flat structure with query_id, query, passage, relevance_score
- No nested unpacking required

Usage:
  python data/download_dataset.py
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_dataset
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_indic_msmarco(
    output_dir: Path,
    language_code: str = "hi",
    relevance_threshold: float = 1.0
) -> Dict[str, int]:
    """
    Download IndicMSMARCO dataset and save passages and queries.
    
    Schema: Flat structure with query_id, query, passage, relevance_score
    - No nested unpacking needed
    - Dataset is 13.1MB total (small, no streaming needed)
    
    Args:
        output_dir: Directory to save processed data
        language_code: Language ISO code (hi=Hindi, bn=Bengali, te=Telugu, etc.)
        relevance_threshold: Minimum relevance_score to consider passage relevant (default: 1.0)
    
    Returns:
        Dictionary with counts of passages and queries saved
        
    Raises:
        Exception: If dataset loading fails (no silent fallback)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading IndicMSMARCO dataset (language: {language_code})...")
    logger.info("Dataset size: 13.1MB (loading fully, no streaming needed)")
    
    # Load full dataset (it's only 13.1MB)
    try:
        dataset = load_dataset(
            "ai4bharat/IndicMSMARCO",
            language_code,
            split="train"
        )
        logger.info(f"Dataset loaded successfully: {len(dataset)} examples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error("\nPossible causes:")
        logger.error(f"1. Invalid language code: '{language_code}' (use: hi, bn, te, ta, mr, etc.)")
        logger.error("2. Network connection issues")
        logger.error("3. HuggingFace datasets library needs update: pip install --upgrade datasets")
        raise RuntimeError(f"Dataset loading failed for language '{language_code}': {e}") from e
    
    # Process flat schema: query_id, query, passage, relevance_score
    passages_dict = {}  # passage_text_hash -> passage data
    query_passages_map = defaultdict(list)  # query_id -> list of (passage_id, relevance_score)
    queries_dict = {}  # query_id -> query data
    passage_id_counter = 0
    
    logger.info("Processing examples...")
    for idx, example in enumerate(dataset):
        if idx % 1000 == 0 and idx > 0:
            logger.info(f"  Processed {idx}/{len(dataset)} examples...")
        
        # Extract fields from flat schema
        query_id = example.get('query_id', f'q_{idx}')
        query_text = example.get('query', '').strip()
        passage_text = example.get('passage', '').strip()
        relevance_score = example.get('relevance_score', 0)
        
        # Skip if missing critical data
        if not query_text or not passage_text:
            continue
        
        if len(passage_text) < 20:  # Skip very short passages
            continue
        
        # Store passage (deduplicate by text hash)
        passage_hash = hash(passage_text)
        if passage_hash not in passages_dict:
            passage_id = f"passage_{passage_id_counter}"
            passage_id_counter += 1
            
            passages_dict[passage_hash] = {
                'id': passage_id,
                'text': passage_text,
                'language': language_code,
                'source': 'indic-msmarco',
                'original_idx': idx
            }
        else:
            passage_id = passages_dict[passage_hash]['id']
        
        # Store query (may appear multiple times with different passages)
        if query_id not in queries_dict:
            queries_dict[query_id] = {
                'id': query_id,
                'text': query_text,
                'language': language_code,
                'original_idx': idx
            }
        
        # Map query to passage with relevance score
        query_passages_map[query_id].append({
            'passage_id': passage_id,
            'relevance_score': float(relevance_score)
        })
    
    logger.info("Processing complete. Building final structures...")
    
    # Convert passages dict to list
    passages = list(passages_dict.values())
    
    # Build queries with relevant passage IDs (based on relevance threshold)
    queries = []
    for query_id, query_data in queries_dict.items():
        # Get passages for this query
        passage_mappings = query_passages_map[query_id]
        
        # Filter by relevance threshold and extract IDs
        relevant_passage_ids = [
            pm['passage_id'] 
            for pm in passage_mappings 
            if pm['relevance_score'] >= relevance_threshold
        ]
        
        # Store all passage mappings for potential future use
        all_passage_scores = {
            pm['passage_id']: pm['relevance_score']
            for pm in passage_mappings
        }
        
        queries.append({
            'id': query_data['id'],
            'text': query_data['text'],
            'relevant_passage_ids': relevant_passage_ids,  # Ground truth for recall@k
            'passage_scores': all_passage_scores,  # All scores for advanced metrics
            'language': query_data['language'],
            'original_idx': query_data['original_idx']
        })
    
    # Sort queries by ID for consistency
    queries.sort(key=lambda x: x['id'])
    
    logger.info(f"\nExtraction complete:")
    logger.info(f"  - Total examples processed: {len(dataset)}")
    logger.info(f"  - Unique passages: {len(passages)}")
    logger.info(f"  - Unique queries: {len(queries)}")
    
    # Calculate statistics
    queries_with_relevant = sum(1 for q in queries if q['relevant_passage_ids'])
    avg_relevant = sum(len(q['relevant_passage_ids']) for q in queries) / len(queries) if queries else 0
    
    logger.info(f"  - Queries with relevant passages (score >= {relevance_threshold}): {queries_with_relevant}/{len(queries)}")
    logger.info(f"  - Avg relevant passages per query: {avg_relevant:.2f}")
    
    # Validate we got meaningful data
    if len(passages) == 0:
        raise ValueError(
            f"No passages extracted! Check dataset structure or language code '{language_code}'"
        )
    
    if len(queries) == 0:
        raise ValueError(
            f"No queries extracted! Check dataset structure or language code '{language_code}'"
        )
    
    # Save passages
    passages_file = output_dir / "passages.json"
    with open(passages_file, 'w', encoding='utf-8') as f:
        json.dump(passages, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved {len(passages)} passages to {passages_file}")
    
    # Save queries
    queries_file = output_dir / "queries.json"
    with open(queries_file, 'w', encoding='utf-8') as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(queries)} queries to {queries_file}")
    
    # Save statistics
    stats = {
        'dataset': 'ai4bharat/IndicMSMARCO',
        'num_passages': len(passages),
        'num_queries': len(queries),
        'language_code': language_code,
        'total_examples': len(dataset),
        'relevance_threshold': relevance_threshold,
        'queries_with_relevant_passages': queries_with_relevant,
        'avg_relevant_passages_per_query': avg_relevant,
        'avg_passage_length': sum(len(p['text']) for p in passages) / len(passages),
        'avg_query_length': sum(len(q['text']) for q in queries) / len(queries)
    }
    
    stats_file = output_dir / "dataset_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {stats_file}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download IndicMSMARCO dataset')
    parser.add_argument(
        '--language',
        type=str,
        default='hi',
        help='Language ISO code: hi=Hindi, bn=Bengali, te=Telugu, etc. (default: hi)'
    )
    parser.add_argument(
        '--relevance-threshold',
        type=float,
        default=1.0,
        help='Minimum relevance_score to consider passage relevant (default: 1.0)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: data/processed/)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "processed"
    
    try:
        stats = download_indic_msmarco(
            output_dir=output_dir,
            language_code=args.language,
            relevance_threshold=args.relevance_threshold
        )
        
        logger.info("\n" + "="*60)
        logger.info("DATASET PREPARATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Dataset: {stats['dataset']}")
        logger.info(f"Language: {stats['language_code']}")
        logger.info(f"Passages: {stats['num_passages']}")
        logger.info(f"Queries: {stats['num_queries']}")
        logger.info(f"Total examples: {stats['total_examples']}")
        logger.info(f"Relevance threshold: {stats['relevance_threshold']}")
        logger.info(f"Queries with relevant passages: {stats['queries_with_relevant_passages']}")
        logger.info(f"Avg relevant passages/query: {stats['avg_relevant_passages_per_query']:.2f}")
        logger.info(f"Avg passage length: {stats['avg_passage_length']:.1f} chars")
        logger.info(f"Avg query length: {stats['avg_query_length']:.1f} chars")
        logger.info("="*60)
        
    except Exception as e:
        logger.error("\n" + "="*60)
        logger.error("DATASET LOADING FAILED")
        logger.error("="*60)
        logger.error(f"Error: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Check internet connection")
        logger.error("2. Verify language code (use 'hi' not 'hindi')")
        logger.error("3. Check HuggingFace datasets library: pip install --upgrade datasets")
        logger.error("4. Try different language: --language bn")
        logger.error("5. Available languages: hi, bn, te, ta, mr, gu, kn, ml, or, pa")
        logger.error("="*60)
        import sys
        sys.exit(1)
