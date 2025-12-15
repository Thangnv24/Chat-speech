from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session as get_db
from app.crud.session import (
    create_session,
    get_sessions,
    get_session as get_session_crud,
    update_session_name,
    delete_session,
)
from app.schemas import ChatSessionCreate, ChatSessionUpdate, ChatSessionResponse

router = APIRouter(prefix="/sessions", tags=["Sessions"])

# Get list sessions
@router.get("/", response_model=list[ChatSessionResponse])
async def get_chat_sessions(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    return await get_sessions(db, skip=skip, limit=limit)

# Get session by id
@router.get("/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(
    session_id: UUID, 
    db: AsyncSession = Depends(get_db)
):
    db_session = await get_session_crud(db, session_id)
    if db_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Session not found"
        )
    return db_session

# Create session
@router.post("/", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_chat_session(
    chat_session: ChatSessionCreate, 
    db: AsyncSession = Depends(get_db)
):
    return await create_session(db, chat_session)

# Update session name by id
@router.patch("/{session_id}/name", response_model=ChatSessionResponse)
async def update_chat_session_name(
    session_id: UUID,
    session_name: str,
    db: AsyncSession = Depends(get_db),
):
    db_session = await get_session_crud(db, session_id)
    if db_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Session not found"
        )
    return await update_session_name(db, session_id, session_name)

# Update session (full update)
@router.put("/{session_id}", response_model=ChatSessionResponse)
async def update_chat_session(
    session_id: UUID,
    session_update: ChatSessionUpdate,
    db: AsyncSession = Depends(get_db),
):
    db_session = await get_session_crud(db, session_id)
    if db_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Session not found"
        )
    
    if session_update.session_name:
        db_session = await update_session_name(db, session_id, session_update.session_name)

    return db_session

# Delete session by id
@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(
    session_id: UUID, 
    db: AsyncSession = Depends(get_db)
):
    db_session = await get_session_crud(db, session_id)
    if db_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Session not found"
        )
    await delete_session(db, session_id)
    return