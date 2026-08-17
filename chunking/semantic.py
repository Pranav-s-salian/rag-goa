"""
Semantic chunking: split on embedding-similarity breakpoints between sentences.
"""
from typing import List, Dict
import re
import numpy as np
from . import Chunk


def chunk(
    passages: List[Dict],
    embed_fn=None,
    similarity_threshold: float = 0.75,
    min_chunk_size: int = 100
) -> List[Chunk]:
    """
    Split passages based on semantic similarity between consecutive sentences.
    
    Args:
        passages: List of passage dicts
        embed_fn: Embedding function (text -> np.ndarray). If None, falls back to character-based splits
        similarity_threshold: Cosine similarity threshold below which to split
        min_chunk_size: Minimum characters per chunk
    
    Returns:
        List of Chunk objects
    """
    chunks = []
    
    for passage in passages:
        text = passage.get('text', '')
        doc_id = passage.get('id', 'unknown')
        language = passage.get('language', 'unknown')
        source = passage.get('source', 'unknown')
        
        if not text:
            continue
        
        # Split into sentences (simple regex, works for most languages)
        sentences = re.split(r'(?<=[.!?।])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            continue
        
        # If no embedding function, fall back to fixed-size groups
        if embed_fn is None:
            current_chunk = []
            current_size = 0
            chunk_idx = 0
            
            for sent in sentences:
                current_chunk.append(sent)
                current_size += len(sent)
                
                if current_size >= 256:  # Default chunk size
                    chunk_text = ' '.join(current_chunk)
                    chunk_id = f"{doc_id}_semantic_{chunk_idx}"
                    chunks.append(Chunk(
                        text=chunk_text,
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        strategy_name="semantic",
                        metadata={
                            'language': language,
                            'source': source,
                            'num_sentences': len(current_chunk),
                            'fallback_mode': True
                        }
                    ))
                    current_chunk = []
                    current_size = 0
                    chunk_idx += 1
            
            # Add remaining
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_id = f"{doc_id}_semantic_{chunk_idx}"
                chunks.append(Chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    strategy_name="semantic",
                    metadata={
                        'language': language,
                        'source': source,
                        'num_sentences': len(current_chunk),
                        'fallback_mode': True
                    }
                ))
        else:
            # Embed all sentences
            embeddings = []
            for sent in sentences:
                emb = embed_fn(sent)
                embeddings.append(emb)
            
            embeddings = np.array(embeddings)
            
            # Calculate cosine similarities between consecutive sentences
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-8
                )
                similarities.append(sim)
            
            # Find split points where similarity drops below threshold
            split_indices = [0]
            for i, sim in enumerate(similarities):
                if sim < similarity_threshold:
                    split_indices.append(i + 1)
            split_indices.append(len(sentences))
            
            # Create chunks from split points
            chunk_idx = 0
            for i in range(len(split_indices) - 1):
                start = split_indices[i]
                end = split_indices[i + 1]
                chunk_sentences = sentences[start:end]
                chunk_text = ' '.join(chunk_sentences)
                
                # Skip too-small chunks
                if len(chunk_text) < min_chunk_size:
                    continue
                
                chunk_id = f"{doc_id}_semantic_{chunk_idx}"
                chunks.append(Chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    strategy_name="semantic",
                    metadata={
                        'language': language,
                        'source': source,
                        'num_sentences': len(chunk_sentences),
                        'avg_similarity': float(np.mean([similarities[j] for j in range(start, min(end - 1, len(similarities)))]) if start < len(similarities) else 0)
                    }
                ))
                chunk_idx += 1
    
    return chunks
