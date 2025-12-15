import uuid

from typing import Optional

from sqlalchemy import select. update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import User
from app.schemas import UserCreate, UserUpdate

# Get list users
async def get_users(db: AsyncSession, skip: int = 0, limit: int = 100) -> list[User]:
    result = await session.execute(select(User).offset(skip).limit(limit))
    return result.scalars().all()

# Get user by user id
async def get_user(db: AsyncSession, user_id: uuid.UUID) -> Optional[User]:
    result = await session.execute(select(User).filter(User.user_id == user_id))
    return result.scalars().first()

# Get user by email
async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    result = await session.execute(select(User).filter(User.email == email))
    return result.scalars().first()

# Create user
async def create_user(db: AsyncSession, user: UserCreate) -> User:
    db_user = User(**user.model_dump())
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

# Check if email exists
async def email_exists(
    db: AsyncSession, email: str, exclude_user_id: Optional[uuid.UUID] = None
) -> bool:
    query = select(User).filter(User.email == email)
    if exclude_user_id:
        query = query.filter(User.user_id != exclude_user_id)
    result = await db.execute(query)
    return result.scalar_one_or_none() is not None