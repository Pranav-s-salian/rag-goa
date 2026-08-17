"""
Quick test script to verify dataset loader works correctly.
"""
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.download_dataset import download_indic_msmarco


def test_dataset_loader():
    """Test dataset loader with IndicMSMARCO."""
    print("="*60)
    print("Testing IndicMSMARCO Dataset Loader")
    print("="*60)
    
    # Test output directory
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    try:
        print("\n1. Loading IndicMSMARCO dataset (full, 13.1MB)...")
        stats = download_indic_msmarco(
            output_dir=output_dir,
            language_code='hi',
            relevance_threshold=1.0
        )
        
        print("\n2. Verifying output files...")
        
        # Check passages
        passages_file = output_dir / "passages.json"
        assert passages_file.exists(), "passages.json not created"
        
        with open(passages_file, 'r', encoding='utf-8') as f:
            passages = json.load(f)
        
        print(f"   ✓ Passages file: {len(passages)} passages")
        
        # Check queries
        queries_file = output_dir / "queries.json"
        assert queries_file.exists(), "queries.json not created"
        
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        
        print(f"   ✓ Queries file: {len(queries)} queries")
        
        # Check stats
        stats_file = output_dir / "dataset_stats.json"
        assert stats_file.exists(), "dataset_stats.json not created"
        print(f"   ✓ Stats file created")
        
        print("\n3. Validating data structure...")
        
        # Check passage structure
        if passages:
            p = passages[0]
            assert 'id' in p, "Passage missing 'id'"
            assert 'text' in p, "Passage missing 'text'"
            assert 'language' in p, "Passage missing 'language'"
            assert 'source' in p, "Passage missing 'source'"
            print(f"   ✓ Passage structure valid")
            print(f"     Sample: {p['id'][:20]}... ({len(p['text'])} chars)")
        
        # Check query structure
        if queries:
            q = queries[0]
            assert 'id' in q, "Query missing 'id'"
            assert 'text' in q, "Query missing 'text'"
            assert 'relevant_passage_ids' in q, "Query missing 'relevant_passage_ids'"
            assert 'language' in q, "Query missing 'language'"
            print(f"   ✓ Query structure valid")
            print(f"     Sample: {q['text'][:50]}...")
            print(f"     Relevant passages: {len(q['relevant_passage_ids'])}")
        
        print("\n4. Checking ground-truth coverage...")
        queries_with_gt = sum(1 for q in queries if q['relevant_passage_ids'])
        coverage = queries_with_gt / len(queries) * 100 if queries else 0
        print(f"   ✓ Queries with ground-truth: {queries_with_gt}/{len(queries)} ({coverage:.1f}%)")
        
        print("\n5. Statistics summary...")
        print(f"   - Passages: {stats['num_passages']}")
        print(f"   - Queries: {stats['num_queries']}")
        print(f"   - Language: {stats['language_code']}")
        print(f"   - Avg relevant passages/query: {stats['avg_relevant_passages_per_query']:.2f}")
        print(f"   - Avg passage length: {stats['avg_passage_length']:.1f} chars")
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        print("\nDataset loader is working correctly.")
        print("You can now run: python data/download_dataset.py")
        print("="*60)
        
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ Test failed!")
        print("="*60)
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Install/upgrade datasets: pip install --upgrade datasets")
        print("3. Verify language code is correct (use 'hi' not 'hindi')")
        print("4. Try: huggingface-cli login")
        print("="*60)
        raise


if __name__ == "__main__":
    try:
        test_dataset_loader()
        sys.exit(0)
    except Exception:
        sys.exit(1)
