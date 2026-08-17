# IndicMSMARCO Dataset Setup

## Quick Start

```bash
# Download full dataset (default, ~10-20 seconds)
python data/download_dataset.py

# Custom configuration
python data/download_dataset.py --language bn --relevance-threshold 2.0
```

## Dataset Information

**Source**: [ai4bharat/IndicMSMARCO](https://huggingface.co/datasets/ai4bharat/IndicMSMARCO)

**Dataset Size**: 13.1MB total

**No streaming needed** - Small enough to load fully in one go!

## Schema

**Flat structure** (no nested unpacking):
```python
{
  'query_id': str,           # Unique query identifier
  'query': str,              # Search query text
  'passage': str,            # Passage text
  'relevance_score': float   # Ground-truth relevance (0-3 typically)
}
```

Simple and clean!

## Supported Languages

Use **ISO 639-1 codes**:

| Language | ISO Code | Example Usage |
|----------|----------|---------------|
| Hindi | `hi` | `--language hi` (default) |
| Bengali | `bn` | `--language bn` |
| Telugu | `te` | `--language te` |
| Tamil | `ta` | `--language ta` |
| Marathi | `mr` | `--language mr` |
| Gujarati | `gu` | `--language gu` |
| Kannada | `kn` | `--language kn` |
| Malayalam | `ml` | `--language ml` |
| Oriya | `or` | `--language or` |
| Punjabi | `pa` | `--language pa` |

**Important**: Use `hi` not `hindi`, `bn` not `bengali`, etc.

## Dataset Structure

### Input (MSMARCO-XI)

Each example contains:
```python
{
  'query': str,                          # The search query
  'passages': {
    'Translated_passages': [str, ...],   # List of Hindi passages
    'is_selected': [bool, ...]           # Ground-truth relevance labels
  }
}
```

### Output (After Processing)

**`passages.json`**:
```json
[
  {
    "id": "passage_0",
    "text": "मशीन लर्निंग...",
    "language": "hi",
    "source": "msmarco-xi",
    "example_idx": 0,
    "passage_idx_in_example": 0
  }
]
```

**`queries.json`**:
```json
[
  {
    "id": "query_0",
    "text": "मशीन लर्निंग क्या है?",
    "relevant_passage_ids": ["passage_0", "passage_5"],
    "language": "hi",
    "example_idx": 0
  }
]
```

**Key field**: `relevant_passage_ids` contains ground-truth relevant passages for recall@k evaluation.

**`dataset_stats.json`**:
```json
{
  "num_passages": 4523,
  "num_queries": 5000,
  "language_code": "hi",
  "max_examples_processed": 5000,
  "avg_relevant_passages_per_query": 2.34,
  "queries_with_relevant_passages": 4891,
  "avg_passage_length": 234.5
}
```

## Usage Examples

### Default (Hindi, 5000 examples)
```bash
python data/download_dataset.py
```

### More examples (10k)
```bash
python data/download_dataset.py --max-examples 10000
```

### Different language (Bengali)
```bash
python data/download_dataset.py --language bn
```

### Custom output directory
```bash
python data/download_dataset.py --output-dir /path/to/output
```

### All options
```bash
python data/download_dataset.py \
  --max-examples 10000 \
  --language te \
  --output-dir custom_data/
```

## Expected Output

```
INFO - Loading MSMARCO-XI dataset (language: hi)...
INFO - Using streaming mode to limit to 5000 examples (dataset is 11.45M/55.6GB)
INFO - Dataset stream opened successfully
INFO - Processed 500/5000 examples...
INFO - Processed 1000/5000 examples...
...
INFO - Extraction complete:
INFO -   - Processed 5000 examples
INFO -   - Extracted 4523 unique passages
INFO -   - Extracted 5000 queries
INFO -   - Avg relevant passages per query: 2.34
INFO - Saved 4523 passages to data/processed/passages.json
INFO - Saved 5000 queries to data/processed/queries.json
INFO - Saved statistics to data/processed/dataset_stats.json

============================================================
DATASET PREPARATION COMPLETE
============================================================
Passages: 4523
Queries: 5000
Language: hi
Queries with ground-truth: 4891
Avg relevant passages/query: 2.34
============================================================
```

## Typical Stats (5000 examples)

- **Passages**: 4,000-5,000 unique passages
- **Queries**: ~5,000 queries
- **Relevant passages per query**: 2-3 on average
- **Queries with ground-truth**: 95-98%
- **Average passage length**: 200-300 characters
- **Download time**: 30-120 seconds (depends on connection)
- **Disk usage**: ~10-20 MB

## Ground-Truth Relevance

The `is_selected` field from MSMARCO-XI indicates which passages are relevant to each query. This is preserved in:

```python
query['relevant_passage_ids']  # List of IDs of relevant passages
```

**Used for**:
- `indexing/eval_strategies.py` - Compute recall@k
- Benchmark different chunking strategies
- Validate retrieval accuracy

**Example**:
```python
# Query: "What is machine learning?"
# Ground truth: passages 0, 5, 12 are relevant
# Retrieved: passages 0, 3, 5, 7, 9

# Recall@5 = 2/3 = 0.667 (found 2 out of 3 relevant passages)
```

## Troubleshooting

### Error: "Dataset not found"

**Problem**: Wrong language code

**Solution**: Use ISO codes (`hi`, `bn`, `te`) not full names (`hindi`, `bengali`, `telugu`)

```bash
# Wrong
python data/download_dataset.py --language hindi

# Correct
python data/download_dataset.py --language hi
```

### Error: "Failed to extract data: 0 passages, 0 queries"

**Problem**: Dataset structure changed or wrong config

**Solutions**:
1. Update datasets library: `pip install --upgrade datasets`
2. Check language code is valid
3. Check internet connection
4. Try different language: `--language bn`

### Error: Connection timeout

**Problem**: Slow internet or HuggingFace down

**Solution**: 
1. Reduce examples: `--max-examples 1000`
2. Retry: Run command again
3. Check HuggingFace status: https://status.huggingface.co/

### Output files empty or too small

**Problem**: Passages filtered out (too short)

**Current filter**: Passages must be ≥20 characters

**Solution**: Check `dataset_stats.json` for actual counts

### Want full dataset (not recommended)

**Warning**: Full dataset is 55.6GB and will take hours

```bash
# This will take a VERY long time
python data/download_dataset.py --max-examples 11450000
```

## File Sizes

| Examples | Passages | Queries | JSON Size | Download Time |
|----------|----------|---------|-----------|---------------|
| 1,000 | ~900 | ~1,000 | ~2 MB | 10-20s |
| 5,000 | ~4,500 | ~5,000 | ~10 MB | 30-60s |
| 10,000 | ~9,000 | ~10,000 | ~20 MB | 60-120s |
| 50,000 | ~45,000 | ~50,000 | ~100 MB | 5-10 min |
| 100,000 | ~90,000 | ~100,000 | ~200 MB | 10-20 min |

**Recommendation**: Start with 5,000 (default), increase if needed for better recall.

## Integration with Pipeline

After running this script:

1. **Build indices**: `python indexing/build_index.py`
   - Reads `data/processed/passages.json`
   - Creates FAISS indices

2. **Evaluate strategies**: `python indexing/eval_strategies.py`
   - Reads `data/processed/queries.json`
   - Uses `relevant_passage_ids` for recall@k

3. **Run benchmarks**: `python eval/run_latency_bench.py`
   - Uses queries for end-to-end testing

## Advanced: Inspect Data

```python
import json

# Load passages
with open('data/processed/passages.json', 'r', encoding='utf-8') as f:
    passages = json.load(f)

print(f"Total passages: {len(passages)}")
print(f"Sample passage: {passages[0]}")

# Load queries
with open('data/processed/queries.json', 'r', encoding='utf-8') as f:
    queries = json.load(f)

print(f"Total queries: {len(queries)}")
print(f"Sample query: {queries[0]}")

# Check ground-truth coverage
queries_with_gt = sum(1 for q in queries if q['relevant_passage_ids'])
print(f"Queries with ground-truth: {queries_with_gt}/{len(queries)}")
```

## Notes

1. **Streaming mode** means we don't download the entire 55.6GB dataset
2. **Deduplication** happens via text hashing (same text = same passage)
3. **Ground-truth preserved** in `relevant_passage_ids` for evaluation
4. **No fallback** to synthetic data - real errors raise exceptions
5. **Validates output** - ensures we got meaningful data before saving

## References

- **Dataset**: https://huggingface.co/datasets/ai4bharat/MSMARCO-XI
- **Paper**: [MSMARCO-XI: Multilingual Information Retrieval](https://arxiv.org/abs/2308.04176)
- **Languages**: Based on AI4Bharat IndicTrans
