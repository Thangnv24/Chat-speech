# 🚀 Quick Reference - Chat RAG Application

## 📦 Khởi động nhanh

```bash
# 1. Start Docker services
docker-compose up -d db qdrant

# 2. Activate Python environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Run migrations
alembic upgrade head

# 4. Ingest data
python ingest_data.py

# 5. Start app
uvicorn app.main:app --reload

# 6. Open browser
http://localhost:8000/static/chat.html
```

---

## 🐳 Docker Commands

```bash
# Start all services
docker-compose up -d

# Start specific services
docker-compose up -d db qdrant

# Stop all services
docker-compose down

# Stop and remove volumes (⚠️ deletes data)
docker-compose down -v

# View logs
docker-compose logs -f
docker-compose logs -f api
docker-compose logs -f qdrant

# Restart service
docker-compose restart api

# Check running containers
docker ps

# Rebuild image
docker-compose build --no-cache api
```

---

## 🗄️ Database Commands

### PostgreSQL

```bash
# Connect to database
docker exec -it chat-postgres psql -U chat_user -d chat_db

# Or if running locally
psql -h localhost -U chat_user -d chat_db

# View data with Python script
python view_db.py

# Export data to JSON
python view_db.py export

# Clear test data
python view_db.py clear
```

### Common SQL queries

```sql
-- View all tables
\dt

-- Count records
SELECT COUNT(*) FROM users;
SELECT COUNT(*) FROM chat_sessions;
SELECT COUNT(*) FROM messages;

-- View recent data
SELECT * FROM users ORDER BY created_at DESC LIMIT 5;
SELECT * FROM chat_sessions ORDER BY started_at DESC LIMIT 5;
SELECT * FROM messages ORDER BY created_at DESC LIMIT 10;

-- Delete test data
DELETE FROM users WHERE email LIKE 'user_%@example.com';

-- Exit
\q
```

---

## 🔷 Qdrant Commands

```bash
# View all collections
python view_qdrant.py

# View collection details
python view_qdrant.py math_philosophy

# Search in collection
python view_qdrant.py search math_philosophy "What is mathematics?"

# Check health
python view_qdrant.py health

# Delete collection
python view_qdrant.py delete old_collection

# Check via curl
curl http://localhost:6333/health
curl http://localhost:6333/collections
curl http://localhost:6333/collections/math_philosophy
```

### Qdrant Dashboard
- URL: http://localhost:6333/dashboard

---

## 📥 Data Ingestion

```bash
# Ingest all files in data/ folder
python ingest_data.py

# Add new documents
# 1. Copy files to data/ folder
cp new_doc.pdf data/

# 2. Run ingest again
python ingest_data.py
```

---

## 🧪 Testing

```bash
# Test RAG pipeline only
python app/service/RAG/main.py

# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/sessions/

# Test chat
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is mathematics?",
    "session_id": "your-session-id",
    "k": 5,
    "search_mode": "hybrid"
  }'
```

---

## 🔧 Troubleshooting

### Clear browser cache
```javascript
// Open browser console (F12)
localStorage.clear();
location.reload();
```

### Reset database
```bash
# Stop and remove volumes
docker-compose down -v

# Start fresh
docker-compose up -d db
alembic upgrade head
```

### Reset Qdrant
```bash
# Delete collection
python view_qdrant.py delete math_philosophy

# Or remove volume
docker-compose down -v
docker-compose up -d qdrant

# Re-ingest data
python ingest_data.py
```

### Check logs
```bash
# Application logs
tail -f logs/app.log
tail -f logs/error.log

# Docker logs
docker-compose logs -f api
docker-compose logs -f qdrant
docker-compose logs -f db
```

---

## 🌐 URLs

- **Home**: http://localhost:8000
- **Chat UI**: http://localhost:8000/static/chat.html
- **API Docs**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **PgAdmin**: http://localhost:8080 (admin@example.com / admin)

---

## 📝 Environment Variables

Key variables in `.env`:

```env
# Database
SQLALCHEMY_DATABASE_URL=postgresql+asyncpg://chat_user:chat_password@localhost:5432/chat_db

# Qdrant
QDRANT_URL=http://localhost:6333

# LLM (choose one)
GEMINI_API_KEY=your_key
QWEN_API_KEY=your_key
OLLAMA_BASE_URL=http://localhost:11434
```

---

## 🎯 Common Workflows

### Add new documents
```bash
cp new_doc.pdf data/
python ingest_data.py
```

### View all data
```bash
python view_db.py        # PostgreSQL
python view_qdrant.py    # Qdrant
```

### Fresh start
```bash
docker-compose down -v
docker-compose up -d
alembic upgrade head
python ingest_data.py
uvicorn app.main:app --reload
```

### Backup data
```bash
# Export PostgreSQL
python view_db.py export

# Backup Qdrant (copy volume)
docker cp chat-qdrant:/qdrant/storage ./qdrant_backup
```

---

## 📚 More Info

- Full setup guide: `COMPLETE_SETUP_GUIDE.md`
- Architecture: `architecture.md`
- API documentation: http://localhost:8000/docs
