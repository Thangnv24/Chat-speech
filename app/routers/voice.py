"""
Voice/Speech-to-Text API endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import io
import os
from groq import Groq

router = APIRouter(prefix="/voice", tags=["voice"])

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None


class TranscriptionResponse(BaseModel):
    text: str
    duration: float
    language: str
    processing_time: Optional[float] = None


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = "vi"
):
    """
    Transcribe audio file to text
    
    Args:
        file: Audio file (wav, mp3, m4a, etc.)
        language: Language code (vi, en, etc.)
    
    Returns:
        Transcription result with text and metadata
    """
    if not groq_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Speech-to-Text service not configured. Please set GROQ_API_KEY."
        )
    
    # Validate file type
    allowed_types = ["audio/wav", "audio/mpeg", "audio/mp4", "audio/x-m4a", 
                     "audio/ogg", "audio/flac", "audio/webm"]
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed types: {', '.join(allowed_types)}"
        )
    
    # Check file size (max 25MB for Groq)
    contents = await file.read()
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large. Maximum size is 25MB."
        )
    
    try:
        # Create file-like object
        audio_file = io.BytesIO(contents)
        audio_file.name = file.filename or "audio.wav"
        
        # Transcribe using Groq
        import time
        start_time = time.time()
        
        transcription = groq_client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            language=language if language else None,
            response_format="verbose_json"
        )
        
        processing_time = time.time() - start_time
        
        return TranscriptionResponse(
            text=transcription.text,
            duration=transcription.duration,
            language=transcription.language,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )


@router.get("/health")
async def voice_health_check():
    """Check if voice service is available"""
    return {
        "service": "voice",
        "status": "available" if groq_client else "unavailable",
        "provider": "Groq (Whisper large-v3)" if groq_client else None,
        "message": "Ready" if groq_client else "GROQ_API_KEY not configured"
    }
