import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    create_async_engine, 
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
from sqlalchemy import event, text
from sqlalchemy.exc import SQLAlchemyError
from .exception_handler import exception_handler
from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base class for all models
Base = declarative_base()

# Global engine instance
engine: Optional[AsyncEngine] = None
AsyncSessionLocal: Optional[async_sessionmaker[AsyncSession]] = None


def create_engine() -> AsyncEngine:
    """
    Create async SQLAlchemy engine with optimized settings.
    
    Returns:
        AsyncEngine: Configured async database engine
    """
    return create_async_engine(
        settings.sqlalchemy_database_uri,
        echo=settings.DEBUG,  # Log SQL queries in debug mode
        echo_pool=settings.DEBUG,  # Log connection pool events
        pool_pre_ping=True,  # Validate connections before use
        pool_size=10,  # Number of connections to maintain
        max_overflow=20,  # Additional connections when pool is full
        pool_timeout=30,  # Timeout for getting connection from pool
        pool_recycle=3600,  # Recycle connections after 1 hour
        # Use NullPool for testing to avoid connection issues
        poolclass=NullPool if settings.DEBUG else None,
        # Connection arguments
        connect_args={
            "server_settings": {
                "application_name": settings.PROJECT_NAME,
            }
        }
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """
    Create async session factory.
    
    Args:
        engine: Database engine
        
    Returns:
        async_sessionmaker: Session factory
    """
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,  # Don't expire objects after commit
        autocommit=False,  # Manual transaction control
        autoflush=False,  # Manual flush control
    )

@exception_handler
async def init_db() -> None:
    global engine, AsyncSessionLocal

    logger.info("Initializing database connection...")
    
    # Create engine
    engine = create_engine()
    
    # Test connection
    async with engine.begin() as conn:
        await conn.execute(text("SELECT 1"))
    
    # Create session factory
    AsyncSessionLocal = create_session_factory(engine)
    
    logger.info("Database initialized successfully")
    logger.info(f"Connected to: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}")



async def close_db() -> None:
    global engine
    
    if engine:
        logger.info("Closing database connections...")
        await engine.dispose()
        logger.info("Database connections closed")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    if not AsyncSessionLocal:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except SQLAlchemyError as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()

# Event listeners for connection management
from sqlalchemy.pool import Pool
from sqlalchemy.engine import Engine
from sqlalchemy import event

@event.listens_for(Pool, "connect")

def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.close()

@event.listens_for(Pool, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    """Log when a connection is checked out from the pool."""
    if settings.DEBUG:
        logger.debug("Connection checked out from pool")


@event.listens_for(Pool, "checkin")
def receive_checkin(dbapi_connection, connection_record):
    """Log when a connection is returned to the pool."""
    if settings.DEBUG:
        logger.debug("Connection returned to pool")


# Database utilities for CRUD operations
class DatabaseManager:
    """
    Utility class for common database operations.
    """
    
    @staticmethod
    async def execute_query(query: str, params: Optional[dict] = None) -> any:
        """
        Execute a raw SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query result
        """
        async with get_session_context() as session:
            result = await session.execute(text(query), params or {})
            await session.commit()
            return result
    
    @staticmethod
    async def get_table_count(table_name: str) -> int:
        """
        Get row count for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            int: Number of rows
        """
        query = f"SELECT COUNT(*) FROM {table_name}"
        result = await DatabaseManager.execute_query(query)
        return result.scalar()
    
    @staticmethod
    async def truncate_table(table_name: str) -> None:
        """
        Truncate a table (delete all rows).
        WARNING: This will delete all data!
        
        Args:
            table_name: Name of the table to truncate
        """
        query = f"TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE"
        await DatabaseManager.execute_query(query)
        logger.warning(f"Table {table_name} truncated")


# Export commonly used items
__all__ = [
    "Base",
    "engine", 
    "AsyncSessionLocal",
    "init_db",
    "close_db", 
    "get_session",
    "get_session_context",
    "create_tables",
    "drop_tables",
    "check_connection",
    "DatabaseManager"
]