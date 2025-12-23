# from uuid import UUID
# from fastapi import APIRouter, Depends, HTTPException, status
# from sqlalchemy.ext.asyncio import AsyncSession

# from app.core.database import get_session
# from app.crud import message as message_crud
# from app.crud import session as session_crud
# from app.schemas import MessageCreate, MessageResponse

# router = APIRouter(prefix="/messages", tags=["Messages"])

# # Send message
# @router.post("/", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
# async def send_message(
#     message: MessageCreate,
#     db: AsyncSession = Depends(get_session),
# ):
#     # Verify session exists
#     session = await session_crud.get_session(db, message.session_id)
#     if session is None:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail="Session not found",
#         )
    
#     new_message = await message_crud.create_message(db, message)
#     return new_message

# # Get message by id
# @router.get("/{message_id}", response_model=MessageResponse)
# async def get_message_endpoint(
#     message_id: UUID,
#     db: AsyncSession = Depends(get_session),
# ):
#     message = await message_crud.get_message(db, message_id)
#     if message is None:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail="Message not found",
#         )
#     return message

# # Get list of messages
# @router.get("/", response_model=list[MessageResponse])
# async def list_messages(
#     skip: int = 0,
#     limit: int = 100,
#     db: AsyncSession = Depends(get_session)
# ):
#     return await message_crud.get_messages(db, skip=skip, limit=limit)

# # Get messages by session
# @router.get("/session/{session_id}", response_model=list[MessageResponse])
# async def get_messages_by_session_endpoint(
#     session_id: UUID,
#     skip: int = 0,
#     limit: int = 100,
#     db: AsyncSession = Depends(get_session)
# ):
#     messages = await message_crud.get_messages_by_session(db, session_id, skip=skip, limit=limit)
#     return messages

# # Delete message by id
# @router.delete("/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
# async def delete_message_endpoint(
#     message_id: UUID,
#     db: AsyncSession = Depends(get_session)
# ):
#     existing_message = await message_crud.get_message(db, message_id)
#     if not existing_message:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail="Message not found"
#         )
#     await message_crud.delete_message(db, message_id)
#     return

from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session
from app.crud import message as message_crud
from app.crud import session as session_crud
from app.schemas import MessageCreate, MessageResponse

router = APIRouter(prefix="/messages", tags=["Messages"])

# Send message
@router.post("/", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def send_message(
    message: MessageCreate,
    db: AsyncSession = Depends(get_session),
):
    # Verify session exists
    session = await session_crud.get_session(db, message.session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    
    new_message = await message_crud.create_message(db, message)
    return new_message

# Get message by id
@router.get("/{message_id}", response_model=MessageResponse)
async def get_message_endpoint(
    message_id: UUID,
    db: AsyncSession = Depends(get_session),
):
    message = await message_crud.get_message(db, message_id)
    if message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found",
        )
    return message

# Get list of messages
@router.get("/", response_model=list[MessageResponse])
async def list_messages(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_session)
):
    return await message_crud.get_messages(db, skip=skip, limit=limit)

# Get messages by session
@router.get("/session/{session_id}", response_model=list[MessageResponse])
async def get_messages_by_session_endpoint(
    session_id: UUID,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_session)
):
    messages = await message_crud.get_messages_by_session(db, session_id, skip=skip, limit=limit)
    return messages

# Delete message by id
@router.delete("/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message_endpoint(
    message_id: UUID,
    db: AsyncSession = Depends(get_session)
):
    existing_message = await message_crud.get_message(db, message_id)
    if not existing_message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )
    await message_crud.delete_message(db, message_id)
    return
