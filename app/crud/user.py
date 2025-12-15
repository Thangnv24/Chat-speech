from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, func, desc
from typing import Optional
from uuid import UUID
from datetime import datetime

from app.models.base import User, ChatSession, Message
from app.schemas import UserCreate, UserUpdate


class UserCRUD:
    """CRUD operations for User model"""

    @staticmethod
    def create(db: Session, user_in: UserCreate) -> User:
        """Create a new user"""
        user = User(
            email=user_in.email,
            full_name=user_in.full_name,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def get_by_id(db: Session, user_id: UUID) -> Optional[User]:
        """Get user by ID"""
        return db.query(User).filter(User.user_id == user_id).first()

    @staticmethod
    def get_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email"""
        return db.query(User).filter(User.email == email).first()

    @staticmethod
    def get_by_id_with_sessions(db: Session, user_id: UUID) -> Optional[User]:
        """Get user by ID with all chat sessions"""
        return (
            db.query(User)
            .options(joinedload(User.chat_sessions))
            .filter(User.user_id == user_id)
            .first()
        )

    @staticmethod
    def get_by_email_with_sessions(db: Session, email: str) -> Optional[User]:
        """Get user by email with all chat sessions"""
        return (
            db.query(User)
            .options(joinedload(User.chat_sessions))
            .filter(User.email == email)
            .first()
        )

    @staticmethod
    def get_all(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        order_by: str = "email",
    ) -> list[User]:
        """Get all users with pagination"""
        query = db.query(User)
        
        if order_by == "email":
            query = query.order_by(User.email)
        elif order_by == "full_name":
            query = query.order_by(User.full_name)
        else:
            query = query.order_by(User.user_id)
        
        return query.offset(skip).limit(limit).all()

    @staticmethod
    def get_total_count(db: Session) -> int:
        """Get total number of users"""
        return db.query(User).count()

    @staticmethod
    def search_users(
        db: Session,
        search_term: str,
        skip: int = 0,
        limit: int = 100,
    ) -> list[User]:
        """Search users by email or full name"""
        search_pattern = f"%{search_term}%"
        return (
            db.query(User)
            .filter(
                or_(
                    User.email.ilike(search_pattern),
                    User.full_name.ilike(search_pattern),
                )
            )
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def update(
        db: Session, user_id: UUID, user_update: UserUpdate
    ) -> Optional[User]:
        """Update a user"""
        user = db.query(User).filter(User.user_id == user_id).first()
        
        if not user:
            return None

        update_data = user_update.model_dump(exclude_unset=True)
        
        for field, value in update_data.items():
            setattr(user, field, value)

        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def delete(db: Session, user_id: UUID) -> bool:
        """Delete a user (cascade delete sessions and messages)"""
        user = db.query(User).filter(User.user_id == user_id).first()
        
        if not user:
            return False

        db.delete(user)
        db.commit()
        return True

    @staticmethod
    def exists(db: Session, user_id: UUID) -> bool:
        """Check if user exists by ID"""
        return db.query(User).filter(User.user_id == user_id).first() is not None

    @staticmethod
    def email_exists(db: Session, email: str, exclude_user_id: Optional[UUID] = None) -> bool:
        """Check if email already exists (useful for validation)"""
        query = db.query(User).filter(User.email == email)
        
        if exclude_user_id:
            query = query.filter(User.user_id != exclude_user_id)
        
        return query.first() is not None

    @staticmethod
    def get_user_statistics(db: Session, user_id: UUID) -> Optional[dict]:
        """Get user statistics (sessions count, messages count, etc.)"""
        user = db.query(User).filter(User.user_id == user_id).first()
        
        if not user:
            return None

        sessions_count = (
            db.query(func.count(ChatSession.session_id))
            .filter(ChatSession.user_id == user_id)
            .scalar()
        )

        messages_count = (
            db.query(func.count(Message.message_id))
            .join(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .scalar()
        )

        latest_session = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .order_by(desc(ChatSession.started_at))
            .first()
        )

        return {
            "user_id": user.user_id,
            "email": user.email,
            "full_name": user.full_name,
            "total_sessions": sessions_count or 0,
            "total_messages": messages_count or 0,
            "latest_session_date": latest_session.started_at if latest_session else None,
        }

    @staticmethod
    def get_active_users(
        db: Session,
        days: int = 30,
        skip: int = 0,
        limit: int = 100,
    ) -> list[User]:
        """Get users who have been active in the last N days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        return (
            db.query(User)
            .join(ChatSession)
            .filter(ChatSession.started_at >= cutoff_date)
            .distinct()
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_inactive_users(
        db: Session,
        days: int = 90,
        skip: int = 0,
        limit: int = 100,
    ) -> list[User]:
        """Get users who haven't been active in the last N days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get users with no sessions or only old sessions
        active_user_ids = (
            db.query(User.user_id)
            .join(ChatSession)
            .filter(ChatSession.started_at >= cutoff_date)
            .distinct()
        )
        
        return (
            db.query(User)
            .filter(~User.user_id.in_(active_user_ids))
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_users_with_session_count(
        db: Session,
        min_sessions: int = 1,
        skip: int = 0,
        limit: int = 100,
    ) -> list[tuple[User, int]]:
        """Get users with their session count (filtered by minimum sessions)"""
        return (
            db.query(User, func.count(ChatSession.session_id).label("session_count"))
            .outerjoin(ChatSession)
            .group_by(User.user_id)
            .having(func.count(ChatSession.session_id) >= min_sessions)
            .order_by(desc("session_count"))
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def bulk_create(db: Session, users_in: list[UserCreate]) -> list[User]:
        """Create multiple users at once"""
        users = [
            User(
                email=user.email,
                full_name=user.full_name,
            )
            for user in users_in
        ]
        
        db.add_all(users)
        db.commit()
        
        for user in users:
            db.refresh(user)
        
        return users

    @staticmethod
    def get_or_create(db: Session, email: str, full_name: Optional[str] = None) -> tuple[User, bool]:
        """Get existing user or create new one. Returns (user, created)"""
        user = UserCRUD.get_by_email(db, email)
        
        if user:
            return user, False
        
        user_create = UserCreate(email=email, full_name=full_name)
        user = UserCRUD.create(db, user_create)
        return user, True

    @staticmethod
    def update_full_name(db: Session, user_id: UUID, full_name: str) -> Optional[User]:
        """Quick update for full name only"""
        user = db.query(User).filter(User.user_id == user_id).first()
        
        if not user:
            return None
        
        user.full_name = full_name
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def get_users_by_email_list(db: Session, emails: list[str]) -> list[User]:
        """Get multiple users by email list"""
        return db.query(User).filter(User.email.in_(emails)).all()


# Import for datetime calculation
from datetime import timedelta

# Create singleton instance
user_crud = UserCRUD()