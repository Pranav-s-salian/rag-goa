"""
Sentence-window chunking: index single sentences, retrieve with surrounding context.
"""
from typing import List, Dict
import re
from . import Chunk


def chunk(passages: List[Dict], window_size: int = 2) -> List[Chunk]:
    """
    Create chunks where each chunk is a single sentence, but metadata includes surrounding context.
    
    Args:
        passages: List of passage dicts
        window_size: Number of sentences before/after to include as context
    
    Returns:
        List of Chunk objects (one per sentence, with context in metadata)
    """
    chunks = []
    
    for passage in passages:
        text = passage.get('text', '')
        doc_id = passage.get('id', 'unknown')
        language = passage.get('language', 'unknown')
        source = passage.get('source', 'unknown')
        
        if not text:
            continue
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?।])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            continue
        
        # Create a chunk for each sentence
        for i, sentence in enumerate(sentences):
            # Determine window boundaries
            start_idx = max(0, i - window_size)
            end_idx = min(len(sentences), i + window_size + 1)
            
            # Context window (for generation, not indexing)
            context_sentences = sentences[start_idx:end_idx]
            context_text = ' '.join(context_sentences)
            
            chunk_id = f"{doc_id}_sentwin_{i}"
            
            chunks.append(Chunk(
                text=sentence,  # Index only the single sentence
                doc_id=doc_id,
                chunk_id=chunk_id,
                strategy_name="sentence_window",
                metadata={
                    'language': language,
                    'source': source,
                    'sentence_index': i,
                    'total_sentences': len(sentences),
                    'window_size': window_size,
                    'context_text': context_text,  # Full context for generation
                    'context_start': start_idx,
                    'context_end': end_idx
                }
            ))
    
    return chunks
