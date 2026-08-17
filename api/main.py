"""
FastAPI application for voice-enabled RAG system.
"""
import os
import logging
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import sys

sys.path.append(str(Path(__file__).parent.parent))

from harness.pipeline import RAGPipeline, PipelineConfig, PipelineResult
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Voice-Enabled RAG System",
    description="High-speed retrieval-augmented generation with voice input and guardrails",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[RAGPipeline] = None


class QueryRequest(BaseModel):
    """Request model for text query."""
    query: str
    top_k: Optional[int] = 10
    enable_rerank: Optional[bool] = False
    enable_groundedness_check: Optional[bool] = True
    enable_hallucination_check: Optional[bool] = True


class QueryResponse(BaseModel):
    """Response model for query results."""
    answer: Optional[str]
    refused: bool
    refusal_reason: Optional[str] = None
    source_chunks: Optional[list[str]] = None
    latency: dict
    stage_details: Optional[dict] = None
    error: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline
    
    logger.info("Starting RAG API server...")
    
    # Load configuration from environment
    config = PipelineConfig(
        strategy_name=os.getenv("STRATEGY_NAME", "fixed"),
        models_dir=os.getenv("MODELS_DIR", str(Path(__file__).parent.parent / "models")),
        top_k=int(os.getenv("TOP_K", "10")),
        enable_rerank=os.getenv("RERANK_ENABLED", "false").lower() == "true",
        enable_input_filter=os.getenv("ENABLE_INPUT_FILTER", "true").lower() == "true",
        enable_groundedness_check=os.getenv("ENABLE_GROUNDEDNESS_CHECK", "true").lower() == "true",
        enable_hallucination_check=os.getenv("ENABLE_HALLUCINATION_CHECK", "true").lower() == "true",
        groundedness_threshold=float(os.getenv("GROUNDEDNESS_THRESHOLD", "0.3")),
        hallucination_threshold=float(os.getenv("HALLUCINATION_THRESHOLD", "0.5")),
        max_context_length=int(os.getenv("MAX_CONTEXT_LENGTH", "2048"))
    )
    
    logger.info(f"Configuration: {config}")
    
    try:
        pipeline = RAGPipeline(config)
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "status": "healthy",
        "pipeline": "initialized",
        "strategy": pipeline.config.strategy_name
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Process a text query through the RAG pipeline.
    
    Args:
        request: Query request with text and configuration
    
    Returns:
        QueryResponse with answer and latency metrics
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        logger.info(f"Processing query: {request.query[:50]}...")
        
        # Run pipeline
        result = await pipeline.run(query=request.query)
        
        # Build response
        response = QueryResponse(
            answer=result.answer,
            refused=result.refused,
            refusal_reason=result.refusal_reason,
            source_chunks=result.source_chunks,
            latency={
                "stt_ms": result.stt_ms,
                "retrieval_ms": result.retrieval_ms,
                "generation_ms": result.generation_ms,
                "guardrail_ms": result.guardrail_ms,
                "total_ms": result.total_ms
            },
            stage_details=result.stage_details,
            error=result.error
        )
        
        logger.info(f"Query processed successfully ({result.total_ms:.2f}ms)")
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/audio", response_model=QueryResponse)
async def query_audio_endpoint(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None)
):
    """
    Process an audio query through the RAG pipeline.
    
    Args:
        audio: Audio file (WAV, MP3, etc.)
        language: Optional language hint
    
    Returns:
        QueryResponse with answer and latency metrics
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    temp_file = None
    
    try:
        # Save uploaded audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"Processing audio query: {audio.filename}")
        
        # Run pipeline with audio
        result = await pipeline.run(audio_file=temp_file_path)
        
        # Build response
        response = QueryResponse(
            answer=result.answer,
            refused=result.refused,
            refusal_reason=result.refusal_reason,
            source_chunks=result.source_chunks,
            latency={
                "stt_ms": result.stt_ms,
                "retrieval_ms": result.retrieval_ms,
                "generation_ms": result.generation_ms,
                "guardrail_ms": result.guardrail_ms,
                "total_ms": result.total_ms
            },
            stage_details=result.stage_details,
            error=result.error
        )
        
        logger.info(f"Audio query processed successfully ({result.total_ms:.2f}ms)")
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing audio query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temp file
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Voice-Enabled RAG System",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "text_query": "/query (POST)",
            "audio_query": "/query/audio (POST)",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
