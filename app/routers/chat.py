from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.core.database import get_session
from app.crud import message as message_crud
from app.crud import session as session_crud
from app.schemas import MessageCreate, MessageResponse, MessageTypeEnum
from app.service.RAG.rag_pipeline import create_pipeline
import os

router = APIRouter(prefix="/chat", tags=["Chat"])

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

class ChatRequest(BaseModel):
    query: str
    session_id: UUID
    k: int = 5
    search_mode: str = "hybrid"

class ChatResponse(BaseModel):
    answer: str
    query_time: float
    num_retrieved: int
    user_message: MessageResponse
    ai_message: MessageResponse

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_session),
):
    try:
        # Verify session exists
        session = await session_crud.get_session(db, request.session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Save user message
        user_message_data = MessageCreate(
            session_id=request.session_id,
            message_type=MessageTypeEnum.Human,
            content=request.query,
            retrieved_context=None
        )
        user_message = await message_crud.create_message(db, user_message_data)
        
        # Get RAG answer
        pipeline = get_rag_pipeline()
        result = pipeline.query(
            query=request.query,
            k=request.k,
            search_mode=request.search_mode,
            include_sources=True
        )
        
        answer = result.get('answer', 'No answer generated')
        context = result.get('context', '')
        
        # Save AI message
        ai_message_data = MessageCreate(
            session_id=request.session_id,
            message_type=MessageTypeEnum.AI,
            content=answer,
            retrieved_context=context
        )
        ai_message = await message_crud.create_message(db, ai_message_data)
        
        return ChatResponse(
            answer=answer,
            query_time=result.get('query_time', 0),
            num_retrieved=result.get('num_retrieved', 0),
            user_message=user_message,
            ai_message=ai_message
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/health")
async def chat_health():
    """Check RAG system health"""
    try:
        pipeline = get_rag_pipeline()
        health = pipeline.health_check()
        return health
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
