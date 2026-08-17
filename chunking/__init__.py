"""
Chunking strategies for RAG pipeline.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class Chunk(BaseModel):
    """Unified chunk model for all strategies."""
    text: str = Field(..., description="The chunk text content")
    doc_id: str = Field(..., description="Source document ID")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    strategy_name: str = Field(..., description="Chunking strategy used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Position information
    start_char: Optional[int] = Field(None, description="Start character position in original doc")
    end_char: Optional[int] = Field(None, description="End character position in original doc")
    
    # Hierarchical support
    parent_chunk_id: Optional[str] = Field(None, description="Parent chunk ID for hierarchical strategy")
    child_chunk_ids: Optional[list[str]] = Field(None, description="Child chunk IDs for hierarchical strategy")
    
    class Config:
        frozen = False
