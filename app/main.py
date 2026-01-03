from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import init_db, close_db
from fastapi.staticfiles import StaticFiles

from app.routers import user, session, message, chat, voice, auth

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    
    # Create tables
    from app.core.database import create_tables
    await create_tables()
    
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
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix=settings.API_V1_PREFIX)
app.include_router(user.router, prefix=settings.API_V1_PREFIX)
app.include_router(session.router, prefix=settings.API_V1_PREFIX)
app.include_router(message.router, prefix=settings.API_V1_PREFIX)
app.include_router(chat.router, prefix=settings.API_V1_PREFIX)
app.include_router(voice.router, prefix=settings.API_V1_PREFIX)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - Redirect to UI"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

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
