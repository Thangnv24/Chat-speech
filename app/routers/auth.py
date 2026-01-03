from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr

from app.core.database import get_session
from app.crud import user as user_crud
from app.crud import session as session_crud
from app.schemas import UserCreate, UserResponse, ChatSessionCreate

router = APIRouter(prefix="/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    email: EmailStr


class LoginResponse(BaseModel):
    user: UserResponse
    session_id: UUID
    message: str


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_session),
):
    """
    Login or register user and create new chat session
    """
    # Check if user exists
    user = await user_crud.get_user_by_email(db, request.email)
    
    if not user:
        # Auto-register new user
        user_data = UserCreate(
            email=request.email,
            full_name=request.email.split('@')[0]  # Use email prefix as name
        )
        user = await user_crud.create_user(db, user_data)
    
    # Create new chat session for this login
    session_data = ChatSessionCreate(
        user_id=user.user_id,
        session_name="New Chat"
    )
    session = await session_crud.create_session(db, session_data)
    
    return LoginResponse(
        user=user,
        session_id=session.session_id,
        message="Login successful"
    )


@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_session),
):
    """
    Register new user
    """
    # Check if email exists
    if await user_crud.email_exists(db, user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    user = await user_crud.create_user(db, user_data)
    return user


class QuickUserCreate(BaseModel):
    user_id: UUID
    email: EmailStr
    full_name: str = "Anonymous User"


@router.post("/users/quick", response_model=UserResponse)
async def create_user_quick(
    user_data: QuickUserCreate,
    db: AsyncSession = Depends(get_session),
):
    """
    Quick user creation with specific UUID (for UI auto-registration)
    """
    # Check if user already exists
    user = await user_crud.get_user(db, user_data.user_id)
    if user:
        return user  # Return existing user
    
    # Create new user with provided UUID
    from app.models.user import User
    user = User(
        user_id=user_data.user_id,
        email=user_data.email,
        full_name=user_data.full_name
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return user
