"""
Todo API Router
Handles CRUD operations for todo items.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_session
from app.models.todo import Todo
from app.schemas.todo import TodoCreate, TodoOut, TodoUpdate

router = APIRouter(prefix="/todos", tags=["todos"])


@router.post("", response_model=TodoOut, status_code=status.HTTP_201_CREATED)
async def create_todo(
    payload: TodoCreate,
    session: AsyncSession = Depends(get_session)
) -> TodoOut:
    """
    Create a new todo item.
    
    Args:
        payload: Todo creation data
        session: Database session
        
    Returns:
        TodoOut: Created todo item
    """
    todo = Todo(title=payload.title, completed=payload.completed)
    session.add(todo)
    await session.commit()
    await session.refresh(todo)
    return todo


@router.get("", response_model=List[TodoOut])
async def list_todos(
    skip: int = 0,
    limit: int = 100,
    session: AsyncSession = Depends(get_session)
) -> List[TodoOut]:
    """
    Get list of all todo items with pagination.
    
    Args:
        skip: Number of items to skip
        limit: Maximum number of items to return
        session: Database session
        
    Returns:
        List[TodoOut]: List of todo items
    """
    result = await session.execute(
        select(Todo).order_by(Todo.id).offset(skip).limit(limit)
    )
    return list(result.scalars().all())


@router.get("/{todo_id}", response_model=TodoOut)
async def get_todo(
    todo_id: int,
    session: AsyncSession = Depends(get_session)
) -> TodoOut:
    """
    Get a specific todo item by ID.
    
    Args:
        todo_id: ID of the todo item
        session: Database session
        
    Returns:
        TodoOut: Todo item
        
    Raises:
        HTTPException: If todo not found
    """
    todo = await session.get(Todo, todo_id)
    if not todo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Todo with id {todo_id} not found"
        )
    return todo


@router.patch("/{todo_id}", response_model=TodoOut)
async def update_todo(
    todo_id: int,
    payload: TodoUpdate,
    session: AsyncSession = Depends(get_session)
) -> TodoOut:
    """
    Update a todo item.
    
    Args:
        todo_id: ID of the todo item
        payload: Update data
        session: Database session
        
    Returns:
        TodoOut: Updated todo item
        
    Raises:
        HTTPException: If todo not found
    """
    todo = await session.get(Todo, todo_id)
    if not todo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Todo with id {todo_id} not found"
        )
    
    # Update only provided fields
    if payload.title is not None:
        todo.title = payload.title
    if payload.completed is not None:
        todo.completed = payload.completed
    
    await session.commit()
    await session.refresh(todo)
    return todo


@router.delete("/{todo_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_todo(
    todo_id: int,
    session: AsyncSession = Depends(get_session)
) -> None:
    """
    Delete a todo item.
    
    Args:
        todo_id: ID of the todo item
        session: Database session
        
    Raises:
        HTTPException: If todo not found
    """
    todo = await session.get(Todo, todo_id)
    if not todo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Todo with id {todo_id} not found"
        )
    
    await session.delete(todo)
    await session.commit()
