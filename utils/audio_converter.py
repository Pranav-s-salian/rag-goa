"""
Audio format conversion utilities using ffmpeg.
Converts various audio formats to 16kHz mono WAV for Sarvam API compatibility.
"""
import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def check_ffmpeg_available() -> bool:
    """
    Check if ffmpeg is available on the system PATH.
    
    Returns:
        True if ffmpeg is available, False otherwise
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def convert_to_wav(
    input_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 16000,
    channels: int = 1
) -> str:
    """
    Convert audio file to WAV format using ffmpeg.
    
    Args:
        input_path: Path to input audio file (any format supported by ffmpeg)
        output_path: Optional path for output WAV file (auto-generated if None)
        sample_rate: Target sample rate in Hz (default: 16000 for speech)
        channels: Number of audio channels (1=mono, 2=stereo; default: 1)
    
    Returns:
        Path to converted WAV file
    
    Raises:
        RuntimeError: If ffmpeg is not available or conversion fails
    """
    # Check ffmpeg availability
    if not check_ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is not available on system PATH. "
            "Please install ffmpeg: https://ffmpeg.org/download.html"
        )
    
    # Generate output path if not provided
    if output_path is None:
        # Create temp file with .wav extension
        fd, output_path = tempfile.mkstemp(suffix='.wav', prefix='converted_')
        os.close(fd)  # Close the file descriptor, we just need the path
    
    try:
        # Build ffmpeg command
        # -i: input file
        # -ar: audio sample rate
        # -ac: audio channels
        # -y: overwrite output file without asking
        # -loglevel error: only show errors
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ar', str(sample_rate),
            '-ac', str(channels),
            '-y',  # Overwrite output
            '-loglevel', 'error',
            output_path
        ]
        
        logger.info(f"Converting {input_path} to WAV format...")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        # Run conversion
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout for conversion
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            raise RuntimeError(f"ffmpeg conversion failed: {error_msg}")
        
        # Verify output file was created
        if not os.path.exists(output_path):
            raise RuntimeError("Conversion completed but output file not found")
        
        output_size = os.path.getsize(output_path)
        if output_size == 0:
            raise RuntimeError("Conversion produced empty output file")
        
        logger.info(f"✓ Conversion successful: {output_path} ({output_size:,} bytes)")
        
        return output_path
    
    except subprocess.TimeoutExpired:
        # Clean up partial output file
        if os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except:
                pass
        raise RuntimeError("Audio conversion timed out (file too large or corrupted)")
    
    except Exception as e:
        # Clean up partial output file
        if output_path and os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except:
                pass
        raise


def get_audio_info(file_path: str) -> dict:
    """
    Get audio file information using ffprobe.
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Dictionary with audio metadata (format, duration, sample_rate, channels)
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration,format_name',
            '-show_entries', 'stream=sample_rate,channels,codec_name',
            '-of', 'default=noprint_wrappers=1',
            file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        if result.returncode != 0:
            return {}
        
        # Parse output
        info = {}
        for line in result.stdout.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                info[key] = value
        
        return info
    
    except Exception as e:
        logger.warning(f"Failed to get audio info: {e}")
        return {}


if __name__ == "__main__":
    # Test conversion utility
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python audio_converter.py <input_audio_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    # Check ffmpeg
    if not check_ffmpeg_available():
        print("Error: ffmpeg not found on system PATH")
        print("Install ffmpeg: https://ffmpeg.org/download.html")
        sys.exit(1)
    
    print(f"Converting {input_file} to WAV...")
    
    try:
        # Get input info
        info = get_audio_info(input_file)
        if info:
            print(f"Input format: {info.get('format_name', 'unknown')}")
            print(f"Codec: {info.get('codec_name', 'unknown')}")
            print(f"Duration: {info.get('duration', 'unknown')}s")
        
        # Convert
        output_file = convert_to_wav(input_file)
        print(f"\n✓ Conversion successful!")
        print(f"Output: {output_file}")
        print(f"Size: {os.path.getsize(output_file):,} bytes")
        
        # Clean up
        cleanup = input("Delete converted file? (y/n): ")
        if cleanup.lower() == 'y':
            os.unlink(output_file)
            print("Cleaned up.")
    
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        sys.exit(1)
