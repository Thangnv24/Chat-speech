"""
Script to view all data in PostgreSQL database
Usage: python view_db.py
"""
import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, text, func
from app.models.base import User, ChatSession, Message

# Get database URL from environment
DATABASE_URL = os.getenv(
    "SQLALCHEMY_DATABASE_URL",
    "postgresql+asyncpg://chat_user:chat_password@localhost:5432/chat_db"
)


async def view_all_data():
    """View all data in database"""
    
    print("=" * 70)
    print("📊 DATABASE VIEWER")
    print("=" * 70)
    print(f"Database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'N/A'}")
    print()
    
    try:
        engine = create_async_engine(DATABASE_URL, echo=False)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        async with async_session() as session:
            # Statistics first
            print("=" * 70)
            print("📈 STATISTICS")
            print("=" * 70)
            
            user_count = await session.execute(text("SELECT COUNT(*) FROM users"))
            session_count = await session.execute(text("SELECT COUNT(*) FROM chat_sessions"))
            message_count = await session.execute(text("SELECT COUNT(*) FROM messages"))
            
            print(f"Total Users:    {user_count.scalar():>5}")
            print(f"Total Sessions: {session_count.scalar():>5}")
            print(f"Total Messages: {message_count.scalar():>5}")
            print()
            
            # Users
            print("=" * 70)
            print("👥 USERS")
            print("=" * 70)
            result = await session.execute(select(User).order_by(User.created_at.desc()))
            users = result.scalars().all()
            
            if not users:
                print("(No users found)")
            else:
                for i, user in enumerate(users, 1):
                    print(f"\n[{i}] User ID: {user.user_id}")
                    print(f"    Email:   {user.email}")
                    print(f"    Name:    {user.full_name}")
                    print(f"    Created: {user.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            
            print()
            
            # Sessions with message count
            print("=" * 70)
            print("💬 CHAT SESSIONS")
            print("=" * 70)
            
            query = select(
                ChatSession,
                func.count(Message.message_id).label('message_count')
            ).outerjoin(
                Message, ChatSession.session_id == Message.session_id
            ).group_by(
                ChatSession.session_id
            ).order_by(
                ChatSession.started_at.desc()
            )
            
            result = await session.execute(query)
            sessions_data = result.all()
            
            if not sessions_data:
                print("(No sessions found)")
            else:
                for i, (sess, msg_count) in enumerate(sessions_data, 1):
                    print(f"\n[{i}] Session ID: {sess.session_id}")
                    print(f"    User ID:    {sess.user_id}")
                    print(f"    Name:       {sess.session_name or '(Unnamed)'}")
                    print(f"    Started:    {sess.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"    Messages:   {msg_count}")
            
            print()
            
            # Recent Messages
            print("=" * 70)
            print("💭 RECENT MESSAGES (Last 10)")
            print("=" * 70)
            
            result = await session.execute(
                select(Message)
                .order_by(Message.created_at.desc())
                .limit(10)
            )
            messages = result.scalars().all()
            
            if not messages:
                print("(No messages found)")
            else:
                for i, msg in enumerate(messages, 1):
                    content_preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
                    print(f"\n[{i}] Message ID: {msg.message_id}")
                    print(f"    Session:    {msg.session_id}")
                    print(f"    Type:       {msg.message_type}")
                    print(f"    Content:    {content_preview}")
                    print(f"    Created:    {msg.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            
            print()
            
            # User activity summary
            print("=" * 70)
            print("📊 USER ACTIVITY SUMMARY")
            print("=" * 70)
            
            query = select(
                User.email,
                User.full_name,
                func.count(ChatSession.session_id).label('session_count'),
                func.count(Message.message_id).label('message_count')
            ).outerjoin(
                ChatSession, User.user_id == ChatSession.user_id
            ).outerjoin(
                Message, ChatSession.session_id == Message.session_id
            ).group_by(
                User.user_id
            ).order_by(
                func.count(Message.message_id).desc()
            )
            
            result = await session.execute(query)
            activity = result.all()
            
            if not activity:
                print("(No activity)")
            else:
                print(f"\n{'Email':<30} {'Name':<20} {'Sessions':>10} {'Messages':>10}")
                print("-" * 70)
                for email, name, sess_count, msg_count in activity:
                    print(f"{email:<30} {name:<20} {sess_count:>10} {msg_count:>10}")
            
            print()
        
        await engine.dispose()
        
        print("=" * 70)
        print("✅ Done!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error connecting to database:")
        print(f"   {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check PostgreSQL is running:")
        print("      docker ps | grep postgres")
        print("   2. Check .env file has correct DATABASE_URL")
        print("   3. Test connection:")
        print("      psql -h localhost -U chat_user -d chat_db")
        print()


async def export_to_json():
    """Export all data to JSON file"""
    import json
    from datetime import datetime
    
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # Get all data
        users_result = await session.execute(select(User))
        sessions_result = await session.execute(select(ChatSession))
        messages_result = await session.execute(select(Message))
        
        users = users_result.scalars().all()
        sessions = sessions_result.scalars().all()
        messages = messages_result.scalars().all()
        
        # Convert to dict
        data = {
            "export_time": datetime.now().isoformat(),
            "users": [
                {
                    "user_id": str(u.user_id),
                    "email": u.email,
                    "full_name": u.full_name,
                    "created_at": u.created_at.isoformat()
                }
                for u in users
            ],
            "sessions": [
                {
                    "session_id": str(s.session_id),
                    "user_id": str(s.user_id),
                    "session_name": s.session_name,
                    "started_at": s.started_at.isoformat()
                }
                for s in sessions
            ],
            "messages": [
                {
                    "message_id": str(m.message_id),
                    "session_id": str(m.session_id),
                    "message_type": m.message_type,
                    "content": m.content,
                    "created_at": m.created_at.isoformat()
                }
                for m in messages
            ]
        }
        
        # Save to file
        filename = f"db_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exported to {filename}")
        print(f"   Users: {len(users)}")
        print(f"   Sessions: {len(sessions)}")
        print(f"   Messages: {len(messages)}")
    
    await engine.dispose()


async def clear_test_data():
    """Clear test data (users with email starting with 'user_')"""
    
    print("⚠️  WARNING: This will delete all test users and their data!")
    print("   Test users: email like 'user_%@example.com'")
    confirm = input("   Continue? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        print("❌ Cancelled")
        return
    
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # Delete test users (cascade will delete sessions and messages)
        result = await session.execute(
            text("DELETE FROM users WHERE email LIKE 'user_%@example.com' RETURNING user_id")
        )
        deleted_ids = result.fetchall()
        await session.commit()
        
        print(f"✅ Deleted {len(deleted_ids)} test users and their data")
    
    await engine.dispose()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "export":
            asyncio.run(export_to_json())
        elif command == "clear":
            asyncio.run(clear_test_data())
        else:
            print("Usage:")
            print("  python view_db.py         # View all data")
            print("  python view_db.py export  # Export to JSON")
            print("  python view_db.py clear   # Clear test data")
    else:
        asyncio.run(view_all_data())
