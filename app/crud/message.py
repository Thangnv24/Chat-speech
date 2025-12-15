import uuidfrom sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.base import Message
from app.schemas import MessageCreate

# Create messager 
async def create_message(db: AsyncSession, message_data: MessageCreate, user_id: int) -> Message:
    db_message = Message(
        session_id=message_data.session_id,
        message_type=message_data.message_type.value,
        user_id=user_id
        content=message_data.content,
        retrieved_context=message_data.retrieved_context,
    )
    db.add(db_message)
    await db.commit()
    await db.refresh(db_message)
    return db_message

# Get message by message id
async def get_message(db: AsyncSession, message_id: uuid.UUID) -> Message | None:
    result = await db.execute(select(Message).filter(Message.message_id == message_id))
    return result.scalar_one_or_none()

# Get message by user id
async def get_message_by_user(db: AsyncSession, user_id: int) -> Message | None:
    result = await db.execute(select(Message).filter(Message.user_id == user_id))
    return result.scalar_one_or_none()

# Get list message
async def get_messages(db: AsyncSession) -> list[Message]:
    result = await db.execute(select(Message))
    return list(result.scalars().all())

# Get message by session
async def get_messages_by_session(
    db: AsyncSession, session_id: uuid.UUID
) -> list[Message]:
    result = await db.execute(select(Message).filter(Message.session_id == session_id))
    return list(result.scalars().all())

# Delete message by message id
async def delete_message(db: AsyncSession, message_id: uuid.UUID) -> Message | None:
    message_obj = await get_message(db, message_id)
    if message_obj:
        await db.delete(message_obj)
        await db.commit()
        return message_obj
    return None
