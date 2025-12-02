from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import close_db
from app.routers.todo import router as todo_router
from app.routers.voice import router as voice_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    yield
    # Shutdown
    await close_db()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="0.1.0",
    description="FastAPI Todo Application with PostgreSQL",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(todo_router, prefix=settings.API_V1_PREFIX)
app.include_router(voice_router, prefix=settings.API_V1_PREFIX)


@app.get("/", tags=["health"])
async def health():
    """
    Health check endpoint.
    
    Returns:
        dict: Status information
    """
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": "0.1.0"
    }


@app.get("/health", tags=["health"])
async def health_check():
    """
    Detailed health check endpoint.
    
    Returns:
        dict: Detailed health status
    """
    return {
        "status": "healthy",
        "database": "connected",
        "service": settings.PROJECT_NAME
    }
