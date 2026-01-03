# ⚡ Quick Start Guide

## 🚀 Chạy Nhanh (5 phút)

### 1. Cài đặt
```bash
# Clone và setup
git clone <repo-url>
cd mini_pj
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Setup Database
```bash
# Tạo PostgreSQL database
psql -U postgres
CREATE DATABASE chatbot_db;
\q

# Cấu hình .env
echo DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/chatbot_db > .env
```

### 3. Chạy Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Migration
```bash
alembic upgrade head
```

### 5. Tạo User
```bash
# Option 1: Script Python
python create_user.py

# Option 2: API
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "full_name": "Test User"}'
```

### 6. Start Server
```bash
uvicorn app.main:app --reload
```

### 7. Truy cập
- 🌐 Web UI: http://localhost:8000
- 📚 API Docs: http://localhost:8000/docs
- 💬 Chat: http://localhost:8000/static/chat.html

## 🎯 Test Workflow

```bash
# 1. Login (auto-create session)
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com"}'

# Response: {"user": {...}, "session_id": "xxx", "message": "Login successful"}

# 2. Chat
curl -X POST "http://localhost:8000/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Hello",
    "session_id": "xxx",
    "k": 5
  }'

# 3. Get history
curl "http://localhost:8000/chat/sessions/xxx"
```

## 📝 Các Lệnh Hữu Ích

```bash
# List users
python create_user.py list

# Check database
psql -U postgres -d chatbot_db -c "SELECT * FROM users;"

# Check Qdrant
curl http://localhost:6333/health

# Reset database
alembic downgrade base && alembic upgrade head
```

## 🔥 Features

✅ **Auto-create session khi login** - Không cần tạo session thủ công  
✅ **Session management** - CRUD operations cho sessions  
✅ **RAG Chat** - AI chat với context retrieval  
✅ **Message history** - Lưu và load lịch sử chat  
✅ **Web UI** - Modern chat interface  

## 📖 Chi Tiết

Xem file `SETUP_GUIDE.md` để biết thêm chi tiết về:
- Cấu hình nâng cao
- Troubleshooting
- Production deployment
- API documentation

## 🆘 Lỗi Thường Gặp

**Database connection error:**
```bash
# Check PostgreSQL running
pg_ctl status
# Restart
pg_ctl restart
```

**Qdrant connection error:**
```bash
# Check Qdrant
docker ps | grep qdrant
# Restart
docker restart <container-id>
```

**Migration error:**
```bash
# Reset
alembic downgrade base
alembic upgrade head
```
