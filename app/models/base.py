from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, JSON, Text, Enum, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column, relationship
import uuid
import enum
from app.core.database import Base

class MessageType(enum.Enum):
    AI = "AI"
    Human = "Human"

class User(Base):
    __tablename__ = "users"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    full_name: Mapped[str | None] = mapped_column(String(100), nullable=True)

    chat_session: Mapped[list["ChatSession"]] = relationship(
        back_populates="user", cascade="all,delete-orphan"
    )
    
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    session_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    session_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped["User"] = relationship(back_populates="chat_sessions")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="chat_session", cascade="all, delete-orphan"
    )

class Message(Base):
    __tablename__ = "messages"

    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    message_type: Mapped[str] = mapped_column(
        Enum(MessageType, name="message_types"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    retrieved_context: Mapped[str | None] = mapped_column(Text, nullable=True)

    chat_session: Mapped["ChatSession"] = relationship(
        back_populates="messages"
    )