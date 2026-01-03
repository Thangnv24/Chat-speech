# 🚀 Setup Guide - Chat RAG Application

## ✅ Fixed: Foreign Key Error

**Problem:** Session creation failed because user didn't exist in database.

**Solution:** Added auto-user creation endpoint.

---

## 📋 Quick Start

### 1. Clear Browser Data (Important!)
```javascript
// Open browser console (F12) and run:
localStorage.clear();
location.reload();
```

### 2. Start Application
```bash
# Make sure database is running
# Then start the app
uvicorn app.main:app --reload
```

### 3. Access UI
- Home: http://localhost:8000
- Chat: http://localhost:8000/static/chat.html

---

## 🔧 How It Works Now

1. **First Visit:**
   - UI generates random UUID for user
   - Saves to localStorage
   - Calls `/api/v1/auth/users/quick` to create user in DB

2. **Create Session:**
   - Click "+ New Chat"
   - Backend creates session with existing user_id
   - ✅ No more foreign key errors!

3. **Chat:**
   - Type message and press Enter
   - AI responds using RAG

---

## 🐛 Troubleshooting

### Still getting foreign key error?

**Option 1: Clear localStorage**
```javascript
// In browser console (F12):
localStorage.clear();
location.reload();
```

**Option 2: Create user manually**
```bash
python create_user.py
```

### Check if user exists
```sql
-- In PostgreSQL:
SELECT * FROM users;
```

---

## 📝 API Endpoints

### User Management
- `POST /api/v1/auth/users/quick` - Auto-create user with UUID
- `POST /api/v1/auth/register` - Register with email
- `POST /api/v1/auth/login` - Login and create session

### Sessions
- `GET /api/v1/sessions/` - List all sessions
- `POST /api/v1/sessions/` - Create new session
- `GET /api/v1/sessions/{id}` - Get session details

### Chat
- `POST /api/v1/chat/` - Send message and get AI response
- `GET /api/v1/messages/session/{id}` - Get session messages

---

## 🎯 Next Steps

1. **Test the fix:**
   - Clear localStorage
   - Reload page
   - Create new session
   - Send message

2. **Optional improvements:**
   - Add user profile page
   - Session naming
   - Message editing
   - File upload

---

**All fixed! Ready to chat! 🎉**
