from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.crud.message import (
    create_message,
    get_messages,
    get_message,
    get_messages_by_session,
    delete_message,
)

from app.crud.session import get_session
from app.schemas import MessageCreate, MessageResponse

router = APIRouter(prefix="/messages", tags=["Messages"])

# Send message
@router.post("/", response_model=MessageResponse)
async def send_message(
    message: MessageCreate,
    session_id: UUID,
    db: AsyncSession = Depends(get_async_session),
):
    session = await get_session(session_id, db)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    message = await create_message(message, session_id, db)
    return message

# Get message by id
@router.get("/{message_id}", response_model=MessageResponse)
async def get_message_endpoint(
    message_id: UUID,
    db: AsyncSession = Depends(get_async_session),
):
    message = await get_message(message_id, db)
    if message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found",
        )
    return message

# Get list of messages
@router.get("/", response_model=list[MessageResponse])
async def list_messages(db: AsyncSession = Depends(get_async_session)):
    return await get_messages(db)

# Delete message by id
@router.delete("/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message_endpoint(
    message_id: UUID, db: AsyncSession = Depends(get_async_session)
):
    existing_message = await get_message(message_id, db)
    if not existing_message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
        )
    await delete_message(message_id, db)
    return

# Get list messages by session id
@router.get("/session/{session_id}", response_model=list[MessageResponse])
async def get_messages_by_session_endpoint(
    session_id: UUID, db: AsyncSession = Depends(get_async_session)
):
    messages = await get_messages_by_session(db, session_id)
    if not messages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No messages found for this session",
        )