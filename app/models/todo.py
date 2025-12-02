"""
Todo database model.
"""
from datetime import datetime
from sqlalchemy import String, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.database import Base


class Todo(Base):
    """
    Todo item model.
    
    Attributes:
        id: Primary key
        title: Todo title/description
        completed: Completion status
        created_at: Timestamp when todo was created
        updated_at: Timestamp when todo was last updated
    """
    __tablename__ = "todos"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
        autoincrement=True
    )
    title: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        index=True
    )
    completed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    def __repr__(self) -> str:
        return f"<Todo(id={self.id}, title='{self.title}', completed={self.completed})>"
