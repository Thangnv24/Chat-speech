"""
Pydantic schemas for Todo API.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class TodoBase(BaseModel):
    """Base schema for Todo with common fields."""
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Todo title/description"
    )
    completed: bool = Field(
        default=False,
        description="Completion status"
    )


class TodoCreate(TodoBase):
    """Schema for creating a new todo."""
    pass


class TodoUpdate(BaseModel):
    """Schema for updating an existing todo."""
    title: Optional[str] = Field(
        None,
        min_length=1,
        max_length=200,
        description="Updated todo title"
    )
    completed: Optional[bool] = Field(
        None,
        description="Updated completion status"
    )


class TodoOut(TodoBase):
    """Schema for todo response with all fields."""
    id: int = Field(..., description="Todo ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True)
