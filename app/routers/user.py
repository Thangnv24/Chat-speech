from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session as get_db
from app.crud.user import (
    create_user,
    get_users,
    get_user,
    get_user_by_email,
    update_user,
    delete_user,
    email_exists,
)
from app.schemas import UserCreate, UserUpdate, UserResponse

router = APIRouter(prefix="/users", tags=["Users"])

# Create user
@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user_endpoint(
    user: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    if await email_exists(db, user.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    new_user = await create_user(db, user)
    return new_user

# Get user by id
@router.get("/{user_id}", response_model=UserResponse)
async def get_user_endpoint(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    user = await get_user(db, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user

# Get user by email
@router.get("/email/{email}", response_model=UserResponse)
async def get_user_by_email_endpoint(
    email: str,
    db: AsyncSession = Depends(get_db),
):
    user = await get_user_by_email(db, email)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user

# Get list of users
@router.get("/", response_model=list[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    return await get_users(db, skip=skip, limit=limit)

# Update user
@router.patch("/{user_id}", response_model=UserResponse)
async def update_user_endpoint(
    user_id: UUID,
    user_update: UserUpdate,
    db: AsyncSession = Depends(get_db),
):
    existing_user = await get_user(db, user_id)
    if not existing_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    if user_update.email and user_update.email != existing_user.email:
        if await email_exists(db, user_update.email, exclude_user_id=user_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )
    
    updated_user = await update_user(db, user_id, user_update)
    return updated_user

# Delete user by id
@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_endpoint(
    user_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    existing_user = await get_user(db, user_id)
    if not existing_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    await delete_user(db, user_id)
    return

