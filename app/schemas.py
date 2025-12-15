from pydantic import BaseModel, EmailStr, Field, ConfigDict
from datetime import datetime
from uuid import UUID
from typing import Optional
from enum import Enum

# Enums
class MessageTypeEnum(str, Enum):
    AI = "AI"
    Human = "Human"

#  USER SCHEMAS 

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    pass

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None

class UserResponse(UserBase):
    user_id: UUID
    
    model_config = ConfigDict(from_attributes=True)

class UserWithSessions(UserResponse):
    chat_sessions: list["ChatSessionResponse"] = []
    
    model_config = ConfigDict(from_attributes=True)

#  CHAT SESSION SCHEMAS 

class ChatSessionBase(BaseModel):
    session_name: Optional[str] = None
    session_summary: Optional[str] = None

class ChatSessionCreate(ChatSessionBase):
    user_id: UUID

class ChatSessionUpdate(BaseModel):
    session_name: Optional[str] = None
    session_summary: Optional[str] = None

class ChatSessionResponse(ChatSessionBase):
    session_id: UUID
    user_id: UUID
    started_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class ChatSessionWithMessages(ChatSessionResponse):
    messages: list["MessageResponse"] = []
    
    model_config = ConfigDict(from_attributes=True)

class ChatSessionWithUserAndMessages(ChatSessionResponse):
    user: UserResponse
    messages: list["MessageResponse"] = []
    
    model_config = ConfigDict(from_attributes=True)

#  MESSAGE SCHEMAS 

class MessageBase(BaseModel):
    content: str = Field(..., min_length=1)
    message_type: MessageTypeEnum
    retrieved_context: Optional[str] = None

class MessageCreate(MessageBase):
    session_id: UUID

class MessageResponse(MessageBase):
    message_id: UUID
    session_id: UUID
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class MessageWithSession(MessageResponse):
    chat_session: ChatSessionResponse
    
    model_config = ConfigDict(from_attributes=True)

#  PAGINATION SCHEMAS 

class PaginationParams(BaseModel):
    skip: int = Field(0, ge=0, description="Number of records to skip")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of records to return")

class PaginatedResponse(BaseModel):
    total: int
    skip: int
    limit: int
    items: list

class PaginatedUsers(PaginatedResponse):
    items: list[UserResponse]

class PaginatedChatSessions(PaginatedResponse):
    items: list[ChatSessionResponse]

class PaginatedMessages(PaginatedResponse):
    items: list[MessageResponse]

#  BULK OPERATIONS SCHEMAS 

class BulkDeleteResponse(BaseModel):
    deleted_count: int
    success: bool
    message: str

class BulkCreateMessages(BaseModel):
    messages: list[MessageCreate]

class BulkCreateResponse(BaseModel):
    created_count: int
    items: list[MessageResponse]