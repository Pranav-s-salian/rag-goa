"""
Hierarchical parent-child chunking: small chunks for retrieval, larger parent chunks for context.
"""
from typing import List, Dict
from . import Chunk


def chunk(
    passages: List[Dict],
    child_size: int = 128,
    parent_size: int = 512,
    overlap: int = 20
) -> List[Chunk]:
    """
    Create hierarchical chunks: small children for retrieval, large parents for generation context.
    
    Args:
        passages: List of passage dicts
        child_size: Size of child chunks (used for retrieval)
        parent_size: Size of parent chunks (used for generation context)
        overlap: Overlap between chunks
    
    Returns:
        List of Chunk objects (both parent and child chunks)
    """
    chunks = []
    
    for passage in passages:
        text = passage.get('text', '')
        doc_id = passage.get('id', 'unknown')
        language = passage.get('language', 'unknown')
        source = passage.get('source', 'unknown')
        
        if not text:
            continue
        
        # First, create parent chunks
        parent_step = parent_size - overlap
        parents = []
        
        for i in range(0, len(text), parent_step):
            parent_text = text[i:i + parent_size]
            if len(parent_text) < parent_size // 2 and i > 0:
                continue
            
            parent_id = f"{doc_id}_hier_parent_{i // parent_step}"
            parent_chunk = Chunk(
                text=parent_text,
                doc_id=doc_id,
                chunk_id=parent_id,
                strategy_name="hierarchical_parent",
                metadata={
                    'language': language,
                    'source': source,
                    'level': 'parent',
                    'chunk_size': parent_size
                },
                start_char=i,
                end_char=i + len(parent_text),
                child_chunk_ids=[]
            )
            parents.append(parent_chunk)
            chunks.append(parent_chunk)
        
        # Now create child chunks and link them to parents
        child_step = child_size - overlap
        
        for i in range(0, len(text), child_step):
            child_text = text[i:i + child_size]
            if len(child_text) < child_size // 2 and i > 0:
                continue
            
            child_id = f"{doc_id}_hier_child_{i // child_step}"
            
            # Find which parent this child belongs to
            child_midpoint = i + len(child_text) // 2
            parent_chunk = None
            
            for parent in parents:
                if parent.start_char <= child_midpoint < parent.end_char:
                    parent_chunk = parent
                    break
            
            parent_id = parent_chunk.chunk_id if parent_chunk else None
            
            child_chunk = Chunk(
                text=child_text,
                doc_id=doc_id,
                chunk_id=child_id,
                strategy_name="hierarchical_child",
                metadata={
                    'language': language,
                    'source': source,
                    'level': 'child',
                    'chunk_size': child_size,
                    'parent_text': parent_chunk.text if parent_chunk else None
                },
                start_char=i,
                end_char=i + len(child_text),
                parent_chunk_id=parent_id
            )
            
            # Link child to parent
            if parent_chunk:
                if parent_chunk.child_chunk_ids is None:
                    parent_chunk.child_chunk_ids = []
                parent_chunk.child_chunk_ids.append(child_id)
            
            chunks.append(child_chunk)
    
    return chunks
