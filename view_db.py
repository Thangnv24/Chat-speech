import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, text, func
from app.models.base import User, ChatSession, Message

DATABASE_URL = os.getenv(
    "SQLALCHEMY_DATABASE_URL",
    "postgresql+asyncpg://chat_user:chat_password@localhost:5432/chat_db"
)


async def view_all():
    try:
        engine = create_async_engine(DATABASE_URL, echo=False)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        async with async_session() as session:
            # Statistics
            print("\nStatistics:")
            user_count = await session.execute(text("SELECT COUNT(*) FROM users"))
            session_count = await session.execute(text("SELECT COUNT(*) FROM chat_sessions"))
            message_count = await session.execute(text("SELECT COUNT(*) FROM messages"))
            
            print(f"Users:    {user_count.scalar()}")
            print(f"Sessions: {session_count.scalar()}")
            print(f"Messages: {message_count.scalar()}")
            
            # Users
            print("\nUsers:")
            result = await session.execute(select(User))
            users = result.scalars().all()
            
            if not users:
                print("No users")
            else:
                for user in users:
                    print(f"\n  ID: {user.user_id}")
                    print(f"  Email: {user.email}")
                    print(f"  Name: {user.full_name}")
            
            # Sessions
            print("\n\nSessions:")
            query = select(
                ChatSession,
                func.count(Message.message_id).label('msg_count')
            ).outerjoin(
                Message, ChatSession.session_id == Message.session_id
            ).group_by(
                ChatSession.session_id
            )
            
            result = await session.execute(query)
            sessions = result.all()
            
            if not sessions:
                print("No sessions")
            else:
                for sess, msg_count in sessions:
                    print(f"\n  ID: {sess.session_id}")
                    print(f"  User: {sess.user_id}")
                    print(f"  Name: {sess.session_name or 'Unnamed'}")
                    print(f"  Messages: {msg_count}")
            
            # Recent messages
            print("\n\nRecent messages (last 5):")
            result = await session.execute(
                select(Message).order_by(Message.message_id.desc()).limit(5)
            )
            messages = result.scalars().all()
            
            if not messages:
                print("No messages")
            else:
                for msg in messages:
                    content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
                    print(f"\n  ID: {msg.message_id}")
                    print(f"  Session: {msg.session_id}")
                    print(f"  Type: {msg.message_type}")
                    print(f"  Content: {content}")
            
            print()
        
        await engine.dispose()
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nCheck:")
        print("  docker ps | grep postgres")
        print("  Check DATABASE_URL in .env")


if __name__ == "__main__":
    asyncio.run(view_all())