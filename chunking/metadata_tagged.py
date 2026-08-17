"""
Metadata-aware chunking: tag chunks with source doc id, language, topic/category for filtering.
"""
from typing import List, Dict, Optional
import re
from . import Chunk


def extract_topic(text: str) -> str:
    """
    Simple topic extraction based on keywords.
    In production, this could use a classifier or keyword extraction model.
    """
    text_lower = text.lower()
    
    # Simple keyword-based topic detection
    topics = {
        'technology': ['machine', 'computer', 'software', 'algorithm', 'data', 'ai', 'neural', 'model'],
        'science': ['research', 'study', 'experiment', 'scientific', 'theory', 'hypothesis'],
        'health': ['health', 'medical', 'disease', 'treatment', 'patient', 'doctor'],
        'business': ['company', 'market', 'business', 'finance', 'investment', 'economy'],
        'education': ['learning', 'education', 'student', 'teacher', 'school', 'university'],
    }
    
    scores = {}
    for topic, keywords in topics.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[topic] = score
    
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return 'general'


def chunk(
    passages: List[Dict],
    chunk_size: int = 256,
    overlap: int = 51,
    extract_topics: bool = True
) -> List[Chunk]:
    """
    Create chunks with rich metadata for filtering at query time.
    
    Args:
        passages: List of passage dicts
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        extract_topics: Whether to extract topics from text
    
    Returns:
        List of Chunk objects with extensive metadata
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
        
        # Extract passage-level metadata
        passage_topic = extract_topic(text) if extract_topics else 'general'
        word_count = len(text.split())
        char_count = len(text)
        
        # Determine content type
        has_numbers = bool(re.search(r'\d', text))
        has_special_chars = bool(re.search(r'[^\w\s]', text))
        
        # Create chunks
        for i in range(0, len(text), step):
            chunk_text = text[i:i + chunk_size]
            
            # Skip very small final chunks
            if len(chunk_text) < chunk_size // 2 and i > 0:
                continue
            
            chunk_id = f"{doc_id}_metadata_{i // step}"
            
            # Chunk-specific metadata
            chunk_word_count = len(chunk_text.split())
            chunk_topic = extract_topic(chunk_text) if extract_topics else passage_topic
            
            chunks.append(Chunk(
                text=chunk_text,
                doc_id=doc_id,
                chunk_id=chunk_id,
                strategy_name="metadata_tagged",
                metadata={
                    # Document-level metadata
                    'language': language,
                    'source': source,
                    'doc_topic': passage_topic,
                    'doc_word_count': word_count,
                    'doc_char_count': char_count,
                    
                    # Chunk-level metadata
                    'chunk_topic': chunk_topic,
                    'chunk_word_count': chunk_word_count,
                    'chunk_position': i // step,
                    'chunk_size': chunk_size,
                    'overlap': overlap,
                    
                    # Content characteristics
                    'has_numbers': has_numbers,
                    'has_special_chars': has_special_chars,
                    
                    # Searchable tags (for metadata filtering)
                    'tags': [language, source, chunk_topic, passage_topic],
                },
                start_char=i,
                end_char=i + len(chunk_text)
            ))
    
    return chunks
