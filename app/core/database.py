import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.models.base import Base
from app.utils.logger import setup_logger

logger = setup_logger("Database", "DEBUG")

DATABASE_URL = os.environ.get("DATABASE_URL", settings.DATABASE_URL)

# Create async engine
aengine = create_async_engine(
    DATABASE_URL,
    echo=settings.DATABASE_ECHO,
    future=True,
    pool_recycle=1800,
    pool_pre_ping=True,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
)

# Create async session maker
async_session_maker = async_sessionmaker(
    aengine, 
    expire_on_commit=False,
    class_=AsyncSession,
)


async def init_db() -> None:
    logger.info("Initializing PostgreSQL database...")
    
    async with aengine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("PostgreSQL database initialized successfully.")
    
    # Optional: Create default data
    async with async_session_maker() as session:
        # You can add initialization logic here
        # For example: creating a default user, etc.
        pass

async def drop_db() -> None:
    logger.warning("Dropping all tables from PostgreSQL database...")
    
    async with aengine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    logger.info("All tables dropped successfully.")

@exception_handler
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session
