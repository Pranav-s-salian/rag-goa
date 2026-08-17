"""
Flask application for voice-enabled RAG system.
Fast and simple with built-in HTML templates.
"""
import os
import time
import logging
from pathlib import Path
import tempfile
import asyncio
from functools import wraps

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from harness.pipeline import RAGPipeline, PipelineConfig
from utils.audio_converter import convert_to_wav

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# CORS
CORS(app)

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg', 'flac', 'webm'}

# Global pipeline instance
pipeline = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def async_route(f):
    """Decorator to run async functions in Flask routes."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper


@app.before_request
def initialize_pipeline():
    """Initialize pipeline before first request."""
    global pipeline
    
    if pipeline is None:
        logger.info("Initializing RAG pipeline...")
        
        config = PipelineConfig(
            strategy_name=os.getenv("STRATEGY_NAME", "fixed"),
            models_dir=os.getenv("MODELS_DIR", str(Path(__file__).parent / "models")),
            top_k=int(os.getenv("TOP_K", "10")),
            enable_rerank=os.getenv("RERANK_ENABLED", "false").lower() == "true",
            enable_input_filter=os.getenv("ENABLE_INPUT_FILTER", "true").lower() == "true",
            enable_groundedness_check=os.getenv("ENABLE_GROUNDEDNESS_CHECK", "true").lower() == "true",
            enable_hallucination_check=os.getenv("ENABLE_HALLUCINATION_CHECK", "true").lower() == "true",
            groundedness_threshold=float(os.getenv("GROUNDEDNESS_THRESHOLD", "0.3")),
            hallucination_threshold=float(os.getenv("HALLUCINATION_THRESHOLD", "0.5")),
            max_context_length=int(os.getenv("MAX_CONTEXT_LENGTH", "2048")),
            # Production index settings
            use_production_index=os.getenv("USE_PRODUCTION_INDEX", "true").lower() == "true",
            production_bucket=os.getenv("PRODUCTION_BUCKET"),  # No fallback - must be in .env
            index_nprobe=int(os.getenv("INDEX_NPROBE", "16"))
        )
        
        try:
            pipeline = RAGPipeline(config)
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)


@app.route('/')
def index():
    """Home page with query interface."""
    return render_template('index.html')


@app.route('/query', methods=['POST'])
@async_route
async def query_text():
    """Handle text query."""
    if pipeline is None:
        return jsonify({'error': 'Pipeline not initialized'}), 503
    
    try:
        # Get query from form
        query = request.form.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        logger.info(f"Processing text query: {query[:50]}...")
        
        # Run pipeline
        start_time = time.time()
        result = await pipeline.run(query=query)
        elapsed = time.time() - start_time
        
        # Build response
        response = {
            'success': not result.refused,
            'answer': result.answer,
            'refused': result.refused,
            'refusal_reason': result.refusal_reason,
            'source_chunks': result.source_chunks[:3] if result.source_chunks else [],  # Top 3 for display
            'latency': {
                'stt_ms': round(result.stt_ms, 2),
                'retrieval_ms': round(result.retrieval_ms, 2),
                'generation_ms': round(result.generation_ms, 2),
                'guardrail_ms': round(result.guardrail_ms, 2),
                'total_ms': round(result.total_ms, 2)
            },
            'elapsed_seconds': round(elapsed, 2)
        }
        
        logger.info(f"Query processed in {elapsed:.2f}s")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/query/audio', methods=['POST'])
@async_route
async def query_audio():
    """Handle audio query."""
    if pipeline is None:
        return jsonify({'error': 'Pipeline not initialized'}), 503
    
    temp_file_path = None
    converted_wav_path = None
    
    try:
        # Check if file is present
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"audio_{int(time.time())}_{filename}")
        file.save(temp_file_path)
        
        logger.info(f"Processing audio query: {filename}")
        
        # Convert to WAV if not already WAV format
        # Sarvam API requires WAV format (16kHz mono recommended)
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        if file_ext != 'wav':
            logger.info(f"Converting {file_ext} to WAV format for Sarvam API...")
            try:
                converted_wav_path = convert_to_wav(
                    temp_file_path,
                    sample_rate=16000,  # 16kHz for speech
                    channels=1  # Mono
                )
                audio_file_for_stt = converted_wav_path
                
                # Verify the converted file
                wav_size = os.path.getsize(audio_file_for_stt)
                logger.info(f"✓ Conversion successful: {audio_file_for_stt} ({wav_size:,} bytes)")
                
                # Quick validation - check it's not empty and has reasonable size
                if wav_size < 100:
                    raise ValueError(f"Converted WAV file too small ({wav_size} bytes) - likely corrupted")
                
                # Verify it's a valid WAV file by checking header
                try:
                    import wave
                    with wave.open(audio_file_for_stt, 'rb') as wav_check:
                        n_channels = wav_check.getnchannels()
                        sample_rate = wav_check.getframerate()
                        n_frames = wav_check.getnframes()
                        duration = n_frames / sample_rate
                        logger.info(f"  WAV verified: {n_channels}ch, {sample_rate}Hz, {duration:.2f}s duration")
                except Exception as e:
                    raise ValueError(f"Converted file is not a valid WAV: {e}")
                
            except Exception as e:
                logger.error(f"Audio conversion failed: {e}")
                return jsonify({'error': f'Audio conversion failed: {str(e)}'}), 500
        else:
            # Already WAV, use as-is
            audio_file_for_stt = temp_file_path
            logger.info(f"Using original WAV file: {audio_file_for_stt}")
        
        # Log the file being sent to pipeline for debugging
        logger.info(f"🎤 Passing audio file to pipeline: {audio_file_for_stt}")
        logger.info(f"   File exists: {os.path.exists(audio_file_for_stt)}")
        logger.info(f"   File size: {os.path.getsize(audio_file_for_stt):,} bytes")
        
        # Run pipeline with the WAV file
        start_time = time.time()
        result = await pipeline.run(audio_file=audio_file_for_stt)
        elapsed = time.time() - start_time
        
        # Build response
        response = {
            'success': not result.refused,
            'answer': result.answer,
            'refused': result.refused,
            'refusal_reason': result.refusal_reason,
            'source_chunks': result.source_chunks[:3] if result.source_chunks else [],
            'latency': {
                'stt_ms': round(result.stt_ms, 2),
                'retrieval_ms': round(result.retrieval_ms, 2),
                'generation_ms': round(result.generation_ms, 2),
                'guardrail_ms': round(result.guardrail_ms, 2),
                'total_ms': round(result.total_ms, 2)
            },
            'elapsed_seconds': round(elapsed, 2)
        }
        
        logger.info(f"Audio query processed in {elapsed:.2f}s")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing audio query: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Cleanup temp files (both original and converted)
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temp file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")
        
        if converted_wav_path and os.path.exists(converted_wav_path):
            try:
                os.unlink(converted_wav_path)
                logger.debug(f"Cleaned up converted file: {converted_wav_path}")
            except Exception as e:
                logger.warning(f"Failed to delete converted file: {e}")


@app.route('/health')
def health():
    """Health check endpoint."""
    if pipeline is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Pipeline not initialized'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'pipeline': 'initialized',
        'strategy': pipeline.config.strategy_name
    })


@app.route('/api/status')
def api_status():
    """API status with configuration."""
    if pipeline is None:
        return jsonify({'error': 'Pipeline not initialized'}), 503
    
    return jsonify({
        'status': 'ready',
        'config': {
            'strategy': pipeline.config.strategy_name,
            'top_k': pipeline.config.top_k,
            'rerank_enabled': pipeline.config.enable_rerank,
            'input_filter': pipeline.config.enable_input_filter,
            'groundedness_check': pipeline.config.enable_groundedness_check,
            'hallucination_check': pipeline.config.enable_hallucination_check
        }
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({'error': 'File too large (max 16MB)'}), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server error."""
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting Flask server on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )
