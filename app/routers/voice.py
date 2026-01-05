"""
Voice Chat Router - Speech-to-Speech with RAG
"""
import os
import io
import time
import tempfile
from pathlib import Path
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.core.database import get_session
from app.crud import message as message_crud
from app.crud import session as session_crud
from app.schemas import MessageCreate, MessageTypeEnum, MessageResponse
from app.service.speech.stt import GroqSTT, AudioPreprocessor
from app.service.speech.tts import text_to_speech, detect_language
from app.service.RAG.rag_pipeline import create_pipeline
from app.utils.logger import setup_logging

logger = setup_logging("voice_chat")

# Create temp directory if not exists
TEMP_DIR = Path(tempfile.gettempdir()) / "voice_chat"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/voice", tags=["Voice Chat"])

# RAG Pipeline singleton
rag_pipeline = None

def get_rag_pipeline():
    global rag_pipeline
    if rag_pipeline is None:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        rag_pipeline = create_pipeline(qdrant_url=qdrant_url)
        rag_pipeline.load_existing_store()
        rag_pipeline.initialize_retriever()
    return rag_pipeline


class VoiceChatRequest(BaseModel):
    session_id: UUID
    k: int = 5
    search_mode: str = "hybrid"
    language: str = "vi"  # vi or en


class VoiceChatResponse(BaseModel):
    transcribed_text: str
    answer: str
    audio_duration: float
    processing_time: float
    query_time: float
    num_retrieved: int
    user_message: MessageResponse
    ai_message: MessageResponse


@router.post("/chat", response_model=VoiceChatResponse)
async def voice_chat(
    session_id: str,
    audio_file: UploadFile = File(...),
    k: int = 5,
    search_mode: str = "hybrid",
    language: str = "vi",
    db: AsyncSession = Depends(get_session),
):
    """
    Voice chat endpoint:
    1. Transcribe audio to text (STT)
    2. Query RAG for answer
    3. Save messages to database
    4. Return text response (TTS handled by client)
    """
    
    start_time = time.time()
    
    try:
        # Verify session exists
        session = await session_crud.get_session(db, UUID(session_id))
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # 1. Speech-to-Text
        logger.info(f"Processing audio file: {audio_file.filename}, content_type: {audio_file.content_type}")
        
        # Read audio file
        audio_bytes = await audio_file.read()
        logger.info(f"Audio size: {len(audio_bytes)} bytes")
        
        if len(audio_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty audio file"
            )
        
        # Save temporarily with proper path
        # Use original extension if not wav
        file_ext = ".webm" if "webm" in str(audio_file.content_type) else ".wav"
        temp_input = TEMP_DIR / f"voice_input_{session_id}{file_ext}"
        
        with open(temp_input, "wb") as f:
            f.write(audio_bytes)
        
        logger.info(f"Saved to: {temp_input}")
        
        # Convert to WAV if needed (Groq accepts various formats, but WAV is safest)
        if file_ext != ".wav":
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(str(temp_input))
                temp_wav = TEMP_DIR / f"voice_input_{session_id}.wav"
                audio.export(str(temp_wav), format="wav")
                temp_input.unlink()  # Remove original
                temp_input = temp_wav
                logger.info(f"Converted to WAV: {temp_input}")
            except Exception as e:
                logger.warning(f"Could not convert audio: {e}, using original format")
        
        # Transcribe
        stt = GroqSTT()
        transcription_result = stt.transcribe_file(str(temp_input), language=language)
        
        transcribed_text = transcription_result.get("text", "")
        audio_duration = transcription_result.get("duration", 0)
        
        if not transcribed_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not transcribe audio"
            )
        
        logger.info(f"Transcribed: {transcribed_text}")
        
        # 2. Save user message (transcribed text)
        user_message_data = MessageCreate(
            session_id=UUID(session_id),
            message_type=MessageTypeEnum.Human,
            content=transcribed_text,
            retrieved_context=None
        )
        user_message = await message_crud.create_message(db, user_message_data)
        
        # 3. Query RAG
        logger.info("Querying RAG...")
        pipeline = get_rag_pipeline()
        
        rag_result = pipeline.query(
            query=transcribed_text,
            k=k,
            search_mode=search_mode,
            include_sources=True
        )
        
        answer = rag_result.get('answer', 'No answer generated')
        context = rag_result.get('context', '')
        query_time = rag_result.get('query_time', 0)
        num_retrieved = rag_result.get('num_retrieved', 0)
        
        logger.info(f"RAG answer: {answer[:100]}...")
        
        # 4. Save AI message
        ai_message_data = MessageCreate(
            session_id=UUID(session_id),
            message_type=MessageTypeEnum.AI,
            content=answer,
            retrieved_context=context
        )
        ai_message = await message_crud.create_message(db, ai_message_data)
        
        # Clean up temp file
        try:
            temp_input.unlink()
        except:
            pass
        
        processing_time = time.time() - start_time
        
        return VoiceChatResponse(
            transcribed_text=transcribed_text,
            answer=answer,
            audio_duration=audio_duration,
            processing_time=processing_time,
            query_time=query_time,
            num_retrieved=num_retrieved,
            user_message=user_message,
            ai_message=ai_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice chat error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/tts")
async def text_to_speech_endpoint(
    text: str,
    language: str = "auto"
):
    """
    Convert text to speech
    Returns audio file
    """
    
    try:
        # Detect language if auto
        if language == "auto":
            language = detect_language(text)
        
        # Generate speech with proper temp path
        output_file = TEMP_DIR / f"tts_output_{int(time.time())}.wav"
        text_to_speech(text, str(output_file))
        
        # Read audio file
        with open(output_file, "rb") as f:
            audio_bytes = f.read()
        
        # Clean up
        try:
            output_file.unlink()
        except:
            pass
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav"
            }
        )
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/health")
async def voice_health():
    """Check voice service health"""
    
    health = {
        "status": "healthy",
        "components": {}
    }
    
    # Check STT
    try:
        groq_key = os.getenv("GROQ_API_KEY")
        health["components"]["stt"] = "configured" if groq_key else "missing_key"
    except:
        health["components"]["stt"] = "error"
    
    # Check TTS
    try:
        eleven_key = os.getenv("ELEVEN_API_KEY")
        health["components"]["tts"] = "configured" if eleven_key else "missing_key"
    except:
        health["components"]["tts"] = "error"
    
    # Check RAG
    try:
        pipeline = get_rag_pipeline()
        rag_health = pipeline.health_check()
        health["components"]["rag"] = rag_health["status"]
    except Exception as e:
        health["components"]["rag"] = f"error: {str(e)}"
    
    # Overall status
    if any(v == "error" or "error:" in str(v) for v in health["components"].values()):
        health["status"] = "unhealthy"
    elif any(v == "missing_key" for v in health["components"].values()):
        health["status"] = "degraded"
    
    return health
