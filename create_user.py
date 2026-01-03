
import asyncio
import uuid
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.models.base import User
from app.core.config import settings

async def create_test_user():
    # Create engine
    engine = create_async_engine(settings.DATABASE_URL, echo=True)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # Create user with specific UUID (same as in localStorage)
        user_id = "5fe83e6b-d456-4bd4-b2f9-8a825995f3ee"  # From your error log
        
        # Check if user exists
        from sqlalchemy import select
        result = await session.execute(select(User).where(User.user_id == uuid.UUID(user_id)))
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            print(f"✅ User already exists: {existing_user.email}")
            return
        
        # Create new user
        user = User(
            user_id=uuid.UUID(user_id),
            email=f"user_{user_id[:8]}@example.com",
            full_name="Test User"
        )
        
        session.add(user)
        await session.commit()
        
        print(f"✅ Created user: {user.email}")
        print(f"   User ID: {user.user_id}")

if __name__ == "__main__":
    asyncio.run(create_test_user())
