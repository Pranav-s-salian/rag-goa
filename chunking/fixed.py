"""
Fixed-size chunking with overlap (baseline control).
"""
from typing import List, Dict
from . import Chunk


def chunk(passages: List[Dict], chunk_size: int = 256, overlap: int = 51) -> List[Chunk]:
    """
    Split passages into fixed-size chunks with overlap.
    
    Args:
        passages: List of passage dicts with 'id', 'text', and optional metadata
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between chunks (default 20% of 256)
    
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
        
        # Calculate step size
        step = chunk_size - overlap
        
        # Create chunks
        for i in range(0, len(text), step):
            chunk_text = text[i:i + chunk_size]
            
            # Skip very small final chunks
            if len(chunk_text) < chunk_size // 2 and i > 0:
                continue
            
            chunk_id = f"{doc_id}_fixed_{i // step}"
            
            chunks.append(Chunk(
                text=chunk_text,
                doc_id=doc_id,
                chunk_id=chunk_id,
                strategy_name="fixed",
                metadata={
                    'language': language,
                    'source': source,
                    'chunk_size': chunk_size,
                    'overlap': overlap
                },
                start_char=i,
                end_char=i + len(chunk_text)
            ))
    
    return chunks
