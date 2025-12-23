import uuid

from sqlalchemy import update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.base import ChatSession, Message
from app.schemas import ChatSessionCreate

# Get list session
async def get_sessions(db: AsyncSession, skip: int = 0, limit: int = 100) -> list[ChatSession]:
    result = await db.execute(select(ChatSession).offset(skip).limit(limit))
    return list(result.scalars().all())

# Get session by session id
async def get_session(db: AsyncSession, session_id: uuid.UUID) -> ChatSession | None:
    result = await db.execute(
        select(ChatSession).filter(ChatSession.session_id == session_id)
    )
    return result.scalar_one_or_none()

# Create session automatic by session data
async def create_session(db: AsyncSession, session_data: SessionCreate) -> ChatSession:
    new_session = ChatSession(**session_data.model_dump())
    db.add(new_session)
    await db.commit()
    await db.refresh(new_session)
    welcome_message = Message(
        session_id=new_session.session_id,
        message_type="AI",
        content="Hi, how can I help you?",
    )
    db.add(welcome_message)
    await db.commit()
    await db.refresh(welcome_message)
    return new_session

# Update session name by session id
async def update_session_name(
    db: AsyncSession, session_id: uuid.UUID, name: str
) -> ChatSession | None:
    result = await db.execute(
        update(ChatSession)
        .where(ChatSession.session_id == session_id)
        .values(session_name=name)
        .returning(ChatSession)
    )
    updated_session = result.scalar_one_or_none()
    if updated_session:
        await db.commit()
        await db.refresh(updated_session)
    return updated_session

# Delete session name by session id
async def delete_session(db: AsyncSession, session_id: uuid.UUID) -> ChatSession | None:
    session = await get_session(db, session_id)
    if session:
        await db.execute(
            delete(ChatSession).where(ChatSession.session_id == session_id)
        )
        await db.commit()
        return session
    return None
