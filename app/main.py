from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import init_db, close_db

# Import all routers
from app.routers import user, session, message, chat, voice

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    print(f"Database initialized")
    print(f"{settings.PROJECT_NAME} started")
    
    yield
    
    # Shutdown
    await close_db()
    print(f"Database connections closed")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Chat Application with RAG, Speech-to-Text, and PostgreSQL",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(user.router, prefix=settings.API_V1_PREFIX, tags=["Users"])
app.include_router(session.router, prefix=settings.API_V1_PREFIX, tags=["Sessions"])
app.include_router(message.router, prefix=settings.API_V1_PREFIX, tags=["Messages"])
app.include_router(chat.router, prefix=settings.API_V1_PREFIX, tags=["Chat"])
app.include_router(voice.router, prefix=settings.API_V1_PREFIX, tags=["Voice"])

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "service": settings.PROJECT_NAME,
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    from app.core.database import check_connection
    
    db_healthy = await check_connection()
    
    return {
        "status": "healthy" if db_healthy else "unhealthy",
        "database": "connected" if db_healthy else "disconnected",
        "service": settings.PROJECT_NAME,
        "version": "1.0.0"
    }
