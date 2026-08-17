"""
HuggingFace bucket loader for production Hindi index.

Loads the full 7.79M+ Hindi passage index (IVF-PQ) from HF bucket,
with proper metadata ordering and alignment verification.

CRITICAL: This index will grow as more languages are added.
Do NOT hardcode vector counts or batch file counts.

OPTIMIZATION: Caches reconstructed metadata to avoid processing 156 batch
files on every startup. After first load, uses single cached metadata file.
"""
import os
import json
import logging
import re
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import faiss
from huggingface_hub import list_bucket_tree
from dotenv import load_dotenv

# Load environment variables at module level
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionIndexLoader:
    """
    Loader for production index from HuggingFace bucket.
    
    Handles:
    - Bucket download with caching
    - Metadata reconstruction in correct numeric order
    - IVF-PQ index nprobe configuration
    - Alignment verification
    """
    
    def __init__(
        self,
        bucket_name: str,
        cache_dir: Optional[Path] = None,
        hf_token: Optional[str] = None,
        nprobe: int = 16
    ):
        """
        Initialize bucket loader.
        
        Args:
            bucket_name: HF bucket name (e.g., 'username/voice-rag-index')
            cache_dir: Local cache directory (default: ./index_cache/hindi_full/)
            hf_token: HF access token (or from HF_TOKEN env var)
            nprobe: IVF-PQ nprobe setting for query (default: 16)
        """
        self.bucket_name = bucket_name
        self.cache_dir = cache_dir or Path("./index_cache/hindi_full")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load HF_TOKEN from environment if not provided
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
        # Verify token is loaded
        if not self.hf_token:
            raise ValueError(
                "HF_TOKEN required for bucket access. "
                "Set HF_TOKEN environment variable in .env file or pass hf_token parameter."
            )
        
        # Log token presence (not value) for diagnostics
        token_length = len(self.hf_token)
        if token_length > 0:
            logger.info(f"✓ HF_TOKEN loaded (length: {token_length} chars)")
        else:
            raise ValueError("HF_TOKEN is empty - check .env file")
        
        self.nprobe = nprobe
        
        # Log resolved bucket ID for visibility
        logger.info(f"ProductionIndexLoader initialized")
        logger.info(f"  ★ BUCKET_REPO_ID: {bucket_name}")  # Prominently log for diagnostics
        logger.info(f"  Cache dir: {self.cache_dir}")
        logger.info(f"  IVF-PQ nprobe: {nprobe}")
    
    def load_index_and_metadata(
        self,
        force_redownload: bool = False,
        force_rebuild_metadata: bool = False
    ) -> Tuple[faiss.IndexIVFPQ, List[Dict]]:
        """
        Load FAISS index and metadata from bucket.
        
        Args:
            force_redownload: Force re-download even if cached files exist
            force_rebuild_metadata: Force rebuild metadata cache even if it exists
        
        Returns:
            Tuple of (faiss_index, metadata_list)
            
        Raises:
            ValueError: If index and metadata counts don't match
        """
        logger.info("="*70)
        logger.info("LOADING PRODUCTION INDEX FROM HF BUCKET")
        logger.info("="*70)
        
        # Step 1: Load FAISS index (from cache if available)
        index_path = self._download_index(force_redownload)
        
        logger.info(f"\nLoading FAISS index from {index_path}...")
        index = faiss.read_index(str(index_path))
        
        # Verify it's IVF-PQ
        if not isinstance(index, faiss.IndexIVFPQ):
            logger.warning(f"⚠ Index type is {type(index)}, expected IndexIVFPQ")
        
        logger.info(f"✓ Index loaded: {index.ntotal:,} vectors")
        logger.info(f"  Index type: {type(index).__name__}")
        
        # Configure nprobe for IVF-PQ
        if isinstance(index, faiss.IndexIVFPQ):
            index.nprobe = self.nprobe
            logger.info(f"  Set nprobe = {self.nprobe} (IVF-PQ query parameter)")
        
        # Step 2: Load metadata (from cache if available, otherwise reconstruct)
        metadata_list = self._load_or_reconstruct_metadata(force_redownload, force_rebuild_metadata)
        
        logger.info(f"✓ Metadata loaded: {len(metadata_list):,} passages")
        
        # Step 3: ALIGNMENT CHECK (with safe handling for partial metadata)
        if index.ntotal != len(metadata_list):
            gap = abs(index.ntotal - len(metadata_list))
            gap_pct = (gap / index.ntotal) * 100
            
            if index.ntotal > len(metadata_list):
                logger.warning("="*70)
                logger.warning("⚠ ALIGNMENT MISMATCH DETECTED")
                logger.warning("="*70)
                logger.warning(
                    f"Index has {index.ntotal:,} vectors but metadata has {len(metadata_list):,} entries.\n"
                    f"Gap: {gap:,} vectors without metadata ({gap_pct:.2f}% of index)\n"
                    f"\n"
                    f"SAFE MODE ENABLED:\n"
                    f"  - System will use all {index.ntotal:,} vectors for retrieval\n"
                    f"  - Results with vector IDs >= {len(metadata_list):,} will be filtered out\n"
                    f"  - Only vectors with valid metadata will be returned\n"
                    f"  - This prevents serving misaligned/missing metadata\n"
                    f"\n"
                    f"ROOT CAUSE: Bucket file 'metadata_batch_155.jsonl' is incomplete\n"
                    f"  Expected: 58,494 lines, Actual: 19,498 lines\n"
                    f"  To fix permanently: Upload complete file to bucket\n"
                )
                logger.warning("="*70)
            else:
                # Metadata > index (unexpected)
                logger.error(
                    f"❌ UNEXPECTED: Metadata ({len(metadata_list):,}) > Index ({index.ntotal:,})\n"
                    f"This should not happen - refusing to load"
                )
                raise ValueError(f"Metadata count exceeds index vectors by {gap:,}")
        else:
            logger.info(f"✓ Perfect alignment: {index.ntotal:,} vectors == {len(metadata_list):,} metadata entries")
        
        # Log embedding prefix convention check
        self._log_embedding_prefix_check()
        
        logger.info("="*70)
        logger.info("✓ PRODUCTION INDEX LOADED SUCCESSFULLY")
        logger.info("="*70 + "\n")
        
        return index, metadata_list
    
    def _download_index(self, force: bool = False) -> Path:
        """Download index.faiss from bucket using Buckets API."""
        index_path = self.cache_dir / "index.faiss"
        
        if index_path.exists() and not force:
            logger.info(f"✓ Index found in cache: {index_path}")
            logger.info(f"  Size: {index_path.stat().st_size / 1024 / 1024:.1f} MB")
            return index_path
        
        logger.info(f"Downloading index.faiss from bucket...")
        logger.info(f"  Using Buckets API: {self.bucket_name}")
        
        try:
            # Use Buckets API (not dataset repo API)
            # Import here to show it's the Buckets-specific function
            from huggingface_hub import download_bucket_files
            
            # Download single file from bucket
            # Note: download_bucket_files expects files parameter as a list of tuples
            download_bucket_files(
                bucket_id=self.bucket_name,
                files=[("index.faiss", str(index_path))],
                token=self.hf_token
            )
            
            # Function downloads directly to specified path
            downloaded_path = str(index_path)
            
            logger.info(f"✓ Downloaded via Buckets API: {downloaded_path}")
            logger.info(f"  Size: {Path(downloaded_path).stat().st_size / 1024 / 1024:.1f} MB")
            
            return Path(downloaded_path)
        
        except Exception as e:
            logger.error(f"❌ Failed to download index via Buckets API: {e}")
            logger.error(f"  Bucket: {self.bucket_name}")
            logger.error(f"  Token length: {len(self.hf_token) if self.hf_token else 0}")
            logger.error(f"  Expected URL: https://huggingface.co/buckets/{self.bucket_name}")
            raise
    
    def _load_or_reconstruct_metadata(
        self,
        force_redownload: bool = False,
        force_rebuild: bool = False
    ) -> List[Dict]:
        """
        Load metadata from cache or reconstruct from batch files.
        
        OPTIMIZATION: Caches the reconstructed metadata to a single file (metadata.pkl)
        to avoid processing 156 JSONL files on every startup.
        
        Args:
            force_redownload: Force re-download batch files
            force_rebuild: Force rebuild metadata cache even if it exists
        
        Returns:
            List of metadata dicts
        """
        metadata_cache_path = self.cache_dir / "metadata.pkl"
        
        # Try to load from cache first (unless forced to rebuild)
        if metadata_cache_path.exists() and not force_rebuild:
            logger.info(f"\n✓ Cached metadata found: {metadata_cache_path}")
            logger.info(f"  Skipping download/reconstruction of 156 batch files")
            logger.info(f"  Loading cached metadata...")
            
            try:
                import time
                load_start = time.perf_counter()
                
                with open(metadata_cache_path, 'rb') as f:
                    metadata_list = pickle.load(f)
                
                load_time = (time.perf_counter() - load_start) * 1000
                
                logger.info(f"✓ Metadata loaded from cache in {load_time:.0f}ms")
                logger.info(f"  Entries: {len(metadata_list):,}")
                logger.info(f"  Cache size: {metadata_cache_path.stat().st_size / 1024 / 1024:.1f} MB")
                
                return metadata_list
            
            except Exception as e:
                logger.warning(f"⚠ Failed to load cached metadata: {e}")
                logger.warning(f"  Falling back to reconstruction from batch files...")
        
        # Cache not available or invalid - reconstruct from batch files
        logger.info(f"\nCached metadata not found or invalid")
        logger.info(f"  Reconstructing from batch files (this will be cached for future startups)...")
        
        # Download batch files
        metadata_paths = self._download_metadata_batches(force_redownload)
        
        # Reconstruct metadata
        logger.info(f"\nReconstructing metadata from {len(metadata_paths)} batch files...")
        metadata_list = self._reconstruct_metadata(metadata_paths)
        
        # Save to cache for future startups
        logger.info(f"\nCaching reconstructed metadata for future startups...")
        try:
            import time
            cache_start = time.perf_counter()
            
            with open(metadata_cache_path, 'wb') as f:
                pickle.dump(metadata_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            cache_time = (time.perf_counter() - cache_start) * 1000
            cache_size = metadata_cache_path.stat().st_size / 1024 / 1024
            
            logger.info(f"✓ Metadata cached successfully in {cache_time:.0f}ms")
            logger.info(f"  Location: {metadata_cache_path}")
            logger.info(f"  Size: {cache_size:.1f} MB")
            logger.info(f"  Next startup will skip batch file processing")
        
        except Exception as e:
            logger.warning(f"⚠ Failed to cache metadata: {e}")
            logger.warning(f"  Metadata will be reconstructed on next startup")
        
        return metadata_list
    
    def _download_metadata_batches(self, force: bool = False) -> List[Path]:
        """
        Download all metadata_batch_*.jsonl files from bucket using Buckets API.
        
        Returns:
            List of paths to metadata batch files (unsorted)
        """
        logger.info(f"Downloading metadata batch files...")
        
        try:
            # List all files in bucket using Buckets API
            # list_bucket_tree returns BucketFile objects, extract filenames
            bucket_tree = list_bucket_tree(
                bucket_id=self.bucket_name,
                token=self.hf_token
            )
            
            # Extract filenames from BucketFile objects
            all_files = [item.path for item in bucket_tree if hasattr(item, 'path')]
            
            # Filter metadata batch files
            metadata_files = [
                f for f in all_files 
                if f.startswith("metadata_batch_") and f.endswith(".jsonl")
            ]
            
            if not metadata_files:
                raise ValueError(
                    f"No metadata_batch_*.jsonl files found in bucket {self.bucket_name}. "
                    f"Verify bucket name and contents at: "
                    f"https://huggingface.co/buckets/{self.bucket_name}"
                )
            
            logger.info(f"  Found {len(metadata_files)} metadata batch files in bucket")
            logger.info(f"  Using Buckets API for downloads...")
            
            # Import Buckets API function
            from huggingface_hub import download_bucket_files
            
            # Download each file
            downloaded_paths = []
            for i, filename in enumerate(metadata_files):
                local_path = self.cache_dir / filename
                
                if local_path.exists() and not force:
                    # Skip if already cached
                    downloaded_paths.append(local_path)
                    if i < 3 or i >= len(metadata_files) - 2:  # Log first/last few
                        logger.info(f"  ✓ Cached: {filename}")
                    elif i == 3:
                        logger.info(f"  ... ({len(metadata_files) - 5} more files cached) ...")
                    continue
                
                # Download using Buckets API
                try:
                    # Note: download_bucket_files expects files parameter as a list of tuples
                    download_bucket_files(
                        bucket_id=self.bucket_name,
                        files=[(filename, str(local_path))],
                        token=self.hf_token
                    )
                    
                    # Function downloads directly to specified path
                    downloaded_paths.append(local_path)
                    
                    if i < 3 or i >= len(metadata_files) - 2:
                        logger.info(f"  ✓ Downloaded: {filename}")
                    elif i == 3:
                        logger.info(f"  ... (downloading {len(metadata_files) - 5} more files) ...")
                
                except Exception as e:
                    logger.error(f"  ✗ Failed to download {filename}: {e}")
                    raise
            
            logger.info(f"✓ Downloaded/verified {len(downloaded_paths)} metadata batch files via Buckets API")
            
            return downloaded_paths
        
        except Exception as e:
            logger.error(f"❌ Failed to download metadata batches: {e}")
            logger.error(f"  Bucket: {self.bucket_name}")
            logger.error(f"  Token present: {bool(self.hf_token)}")
            logger.error(f"  Verify bucket at: https://huggingface.co/buckets/{self.bucket_name}")
            raise
    
    def _reconstruct_metadata(self, metadata_paths: List[Path]) -> List[Dict]:
        """
        Reconstruct metadata list in correct numeric batch order.
        
        CRITICAL: FAISS assigns vector IDs sequentially (0, 1, 2, ...) during insertion.
        Metadata must be concatenated in the SAME order the vectors were added.
        
        This means sorting batch files by numeric suffix, NOT alphabetical order:
          CORRECT: batch_0, batch_1, batch_2, ..., batch_10, batch_11, ...
          WRONG: batch_0, batch_1, batch_10, batch_11, batch_2, ...
        
        Args:
            metadata_paths: List of paths to metadata_batch_N.jsonl files
        
        Returns:
            List of metadata dicts in correct order
        """
        # Extract batch numbers and sort numerically
        batch_pattern = re.compile(r'metadata_batch_(\d+)\.jsonl')
        
        batches = []
        for path in metadata_paths:
            match = batch_pattern.search(path.name)
            if not match:
                logger.warning(f"Skipping file with unexpected name: {path.name}")
                continue
            
            batch_num = int(match.group(1))
            batches.append((batch_num, path))
        
        # Sort by batch number (NUMERIC, not lexicographic!)
        batches.sort(key=lambda x: x[0])
        
        logger.info(f"  Batch order: {[b[0] for b in batches[:10]]}{'...' if len(batches) > 10 else ''}")
        logger.info(f"  (sorted numerically by batch number, not alphabetically)")
        
        # Concatenate all records in order
        metadata_list = []
        
        for batch_num, path in batches:
            logger.info(f"  Loading batch {batch_num}: {path.name}")
            
            with open(path, 'r', encoding='utf-8') as f:
                batch_records = [json.loads(line) for line in f]
                metadata_list.extend(batch_records)
        
        return metadata_list
    
    def _log_embedding_prefix_check(self):
        """
        Log embedding prefix convention check.
        
        e5 models are trained with "query: " and "passage: " prefixes.
        Mismatch between corpus embedding and query embedding prefixes
        will degrade retrieval quality.
        """
        logger.info("\n" + "="*70)
        logger.info("EMBEDDING PREFIX CONVENTION CHECK")
        logger.info("="*70)
        logger.info(
            "⚠ IMPORTANT: Verify embedding prefix consistency:\n"
            "  - Kaggle corpus embedding: Did it use 'passage: ' prefix?\n"
            "  - Query embedding (embeddings/embed.py): Does it use 'query: ' prefix?\n"
            "  - e5 models require BOTH prefixes for optimal retrieval\n"
            "  - Any mismatch will silently degrade retrieval quality\n"
            "\n"
            "  Current query embedding code uses: prefix='query: '\n"
            "  If Kaggle corpus was embedded WITHOUT 'passage: ' prefix,\n"
            "  retrieval quality will be degraded (but won't crash).\n"
            "\n"
            "  Decision: Accept current setup or re-embed corpus/queries?\n"
            "  (This is logged for visibility, not auto-fixed)"
        )
        logger.info("="*70 + "\n")


def cleanup_old_test_indices(models_dir: Path):
    """
    Archive old small test indices to avoid confusion.
    
    The small 1801-chunk indices were used for:
    1. Initial pipeline testing
    2. Chunking strategy comparison evaluation
    
    They must be preserved for the chunking strategy comparison artifact,
    but moved out of the way so they're not accidentally used.
    
    Args:
        models_dir: Path to models directory
    """
    logger.info("="*70)
    logger.info("ARCHIVING OLD TEST INDICES")
    logger.info("="*70)
    
    archive_dir = models_dir / "archive_small_test_indices"
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Patterns for test index files
    test_patterns = [
        "*_index.faiss",
        "*_metadata.pkl",
        "*_summary.json"
    ]
    
    archived_count = 0
    
    for pattern in test_patterns:
        for file_path in models_dir.glob(pattern):
            if file_path.parent == archive_dir:
                continue  # Already archived
            
            # Move to archive
            dest_path = archive_dir / file_path.name
            
            if dest_path.exists():
                logger.info(f"  Skipping (already archived): {file_path.name}")
                continue
            
            logger.info(f"  Archiving: {file_path.name} → {archive_dir.name}/")
            file_path.rename(dest_path)
            archived_count += 1
    
    logger.info(f"\n✓ Archived {archived_count} test index files")
    logger.info(f"  Location: {archive_dir}")
    logger.info(f"  Note: These are preserved for chunking strategy comparison artifact")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    # Test loader
    import sys
    
    # Load bucket name from environment variable (no hardcoded fallback)
    bucket_name = os.getenv("PRODUCTION_BUCKET")
    if not bucket_name:
        print("ERROR: PRODUCTION_BUCKET environment variable not set")
        print("Set it in .env file: PRODUCTION_BUCKET=cybercoot/voice-rag-index")
        sys.exit(1)
    
    loader = ProductionIndexLoader(
        bucket_name=bucket_name,
        nprobe=16
    )
    
    try:
        index, metadata = loader.load_index_and_metadata(force_redownload=False)
        
        print(f"\n✓ Successfully loaded production index")
        print(f"  Vectors: {index.ntotal:,}")
        print(f"  Metadata entries: {len(metadata):,}")
        print(f"  Index type: {type(index).__name__}")
        
        if len(metadata) > 0:
            print(f"\n  Sample metadata entry:")
            print(f"    {metadata[0]}")
    
    except Exception as e:
        print(f"\n✗ Failed to load index: {e}")
        sys.exit(1)
