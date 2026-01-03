# 🚀 HƯỚNG DẪN SETUP HOÀN CHỈNH - Chat RAG Application

## 📋 MỤC LỤC
1. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
2. [Cài đặt Docker](#cài-đặt-docker)
3. [Setup Qdrant Vector Database](#setup-qdrant)
4. [Setup PostgreSQL Database](#setup-postgresql)
5. [Ingest dữ liệu vào Qdrant](#ingest-dữ-liệu)
6. [Chạy ứng dụng](#chạy-ứng-dụng)
7. [Truy cập và quản lý dữ liệu](#truy-cập-dữ-liệu)
8. [Troubleshooting](#troubleshooting)

---

## 1️⃣ YÊU CẦU HỆ THỐNG

### Phần mềm cần thiết:
- **Python 3.11+**
- **Docker Desktop** (Windows/Mac) hoặc Docker Engine (Linux)
- **Git**
- **PostgreSQL Client** (tùy chọn, để truy cập trực tiếp DB)

### Kiểm tra cài đặt:
```bash
# Kiểm tra Python
python --version

# Kiểm tra Docker
docker --version
docker-compose --version

# Kiểm tra Git
git --version
```

---

## 2️⃣ CÀI ĐẶT DOCKER

### Windows:
1. Tải **Docker Desktop**: https://www.docker.com/products/docker-desktop/
2. Cài đặt và khởi động Docker Desktop
3. Đảm bảo WSL 2 được bật (Docker sẽ tự động cài)

### Mac:
```bash
# Dùng Homebrew
brew install --cask docker
```

### Linux (Ubuntu/Debian):
```bash
# Cài Docker Engine
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Cài Docker Compose
sudo apt-get install docker-compose-plugin

# Thêm user vào docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Kiểm tra Docker hoạt động:
```bash
docker run hello-world
```

---

## 3️⃣ SETUP QDRANT VECTOR DATABASE

### Cách 1: Chạy Qdrant với Docker (Recommended)

#### A. Chạy standalone Qdrant:
```bash
# Chạy Qdrant container
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

# Kiểm tra Qdrant đang chạy
docker ps | grep qdrant

# Xem logs
docker logs qdrant

# Kiểm tra health
curl http://localhost:6333/health
```

#### B. Chạy Qdrant với docker-compose (cùng toàn bộ stack):
```bash
# Chỉ chạy Qdrant
docker-compose up -d qdrant

# Kiểm tra
docker-compose ps
docker-compose logs qdrant
```

### Cách 2: Chạy Qdrant local (không dùng Docker)

```bash
# Cài Qdrant binary
# Windows: Tải từ https://github.com/qdrant/qdrant/releases
# Linux/Mac:
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz

# Chạy Qdrant
./qdrant

# Hoặc với config tùy chỉnh
./qdrant --config-path ./config/config.yaml
```

### Truy cập Qdrant Dashboard:
- URL: http://localhost:6333/dashboard
- API: http://localhost:6333

### Kiểm tra collections:
```bash
# Xem tất cả collections
curl http://localhost:6333/collections

# Xem chi tiết collection
curl http://localhost:6333/collections/math_philosophy
```

---

## 4️⃣ SETUP POSTGRESQL DATABASE

### Cách 1: Chạy PostgreSQL với Docker

#### A. Standalone PostgreSQL:
```bash
# Chạy PostgreSQL container
docker run -d \
  --name postgres \
  -e POSTGRES_DB=chat_db \
  -e POSTGRES_USER=chat_user \
  -e POSTGRES_PASSWORD=chat_password \
  -p 5432:5432 \
  -v pgdata:/var/lib/postgresql/data \
  postgres:16-alpine

# Kiểm tra
docker ps | grep postgres
docker logs postgres
```

#### B. Với docker-compose:
```bash
# Chạy cả PostgreSQL và PgAdmin
docker-compose up -d db pgadmin

# Kiểm tra
docker-compose ps
```

### Cách 2: Cài PostgreSQL local

#### Windows:
1. Tải installer: https://www.postgresql.org/download/windows/
2. Cài đặt với password cho user `postgres`
3. Tạo database:
```sql
CREATE DATABASE chat_db;
CREATE USER chat_user WITH PASSWORD 'chat_password';
GRANT ALL PRIVILEGES ON DATABASE chat_db TO chat_user;
```

#### Linux (Ubuntu/Debian):
```bash
# Cài PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Khởi động service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Tạo database
sudo -u postgres psql
CREATE DATABASE chat_db;
CREATE USER chat_user WITH PASSWORD 'chat_password';
GRANT ALL PRIVILEGES ON DATABASE chat_db TO chat_user;
\q
```

### Kiểm tra kết nối:
```bash
# Dùng psql
psql -h localhost -U chat_user -d chat_db

# Hoặc dùng Python
python -c "import psycopg2; conn = psycopg2.connect('postgresql://chat_user:chat_password@localhost:5432/chat_db'); print('Connected!')"
```

---

## 5️⃣ INGEST DỮ LIỆU VÀO QDRANT

### Chuẩn bị dữ liệu:

1. **Đặt file vào thư mục `data/`:**
```bash
# Cấu trúc thư mục
data/
├── doc_1.pdf
├── document_2.pdf
├── article.txt
└── research_paper.pdf
```

2. **Định dạng file hỗ trợ:**
- PDF (.pdf)
- Text (.txt)
- Markdown (.md)
- Word (.docx) - nếu có thư viện python-docx

### Cách 1: Dùng script Python (Recommended)

#### A. Tạo script ingest:
```bash
# File: ingest_data.py
```

```python
import os
from pathlib import Path
from app.service.RAG.main import SimpleRAG

def ingest_documents():
    # Khởi tạo RAG
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    rag = SimpleRAG(qdrant_url=qdrant_url, collection_name="math_philosophy")
    
    # Lấy tất cả file PDF và TXT trong thư mục data
    data_dir = Path("data")
    documents = []
    
    for ext in ["*.pdf", "*.txt", "*.md"]:
        documents.extend(data_dir.glob(ext))
    
    if not documents:
        print("❌ Không tìm thấy file nào trong thư mục data/")
        return
    
    print(f"📚 Tìm thấy {len(documents)} file:")
    for doc in documents:
        print(f"  - {doc.name}")
    
    # Ingest
    print("\n🔄 Bắt đầu ingest...")
    document_paths = [str(doc) for doc in documents]
    success = rag.ingest(document_paths)
    
    if success:
        print("\n✅ Ingest thành công!")
        print("\n📊 Thông tin collection:")
        rag.get_info()
    else:
        print("\n❌ Ingest thất bại!")

if __name__ == "__main__":
    ingest_documents()
```

#### B. Chạy script:
```bash
# Đảm bảo Qdrant đang chạy
curl http://localhost:6333/health

# Chạy ingest
python ingest_data.py
```

### Cách 2: Dùng interactive terminal

```bash
# Chạy RAG terminal
python app/service/RAG/main.py

# Khi được hỏi, nhập đường dẫn file (cách nhau bởi dấu phẩy)
# Ví dụ:
data/doc_1.pdf, data/document_2.pdf, data/article.txt
```

### Cách 3: Dùng API endpoint (nếu có)

```bash
# Gửi request ingest qua API
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "document_paths": ["data/doc_1.pdf", "data/document_2.pdf"]
  }'
```

### Kiểm tra dữ liệu đã ingest:

```bash
# Kiểm tra collection trong Qdrant
curl http://localhost:6333/collections/math_philosophy

# Hoặc dùng Python
python -c "
from app.service.RAG.main import SimpleRAG
rag = SimpleRAG()
rag.setup()
rag.get_info()
"
```

### Thêm dữ liệu mới:

```bash
# 1. Thêm file mới vào thư mục data/
cp new_document.pdf data/

# 2. Chạy lại ingest (sẽ tự động merge với dữ liệu cũ)
python ingest_data.py
```

---

## 6️⃣ CHẠY ỨNG DỤNG

### Setup môi trường:

#### 1. Clone repository (nếu chưa có):
```bash
git clone <your-repo-url>
cd mini_pj
```

#### 2. Tạo virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

#### 3. Cài dependencies:
```bash
pip install -r requirements.txt
```

#### 4. Cấu hình .env:
```bash
# Copy từ example
cp .env.example .env

# Chỉnh sửa .env
# Windows: notepad .env
# Linux/Mac: nano .env
```

**Nội dung .env quan trọng:**
```env
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=chat_db
POSTGRES_USER=chat_user
POSTGRES_PASSWORD=chat_password
SQLALCHEMY_DATABASE_URL=postgresql+asyncpg://chat_user:chat_password@localhost:5432/chat_db

# Qdrant
QDRANT_URL=http://localhost:6333

# LLM (chọn 1 trong các option)
GEMINI_API_KEY=your_gemini_key
# hoặc
QWEN_API_KEY=your_qwen_key
# hoặc
OLLAMA_BASE_URL=http://localhost:11434
```

#### 5. Chạy migrations:
```bash
# Tạo database schema
alembic upgrade head
```

### Chạy ứng dụng:

#### Option 1: Chạy development server (Local)

```bash
# Đảm bảo Qdrant và PostgreSQL đang chạy
docker ps

# Chạy FastAPI
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Option 2: Chạy toàn bộ với Docker Compose (Recommended)

```bash
# Chạy tất cả services
docker-compose up -d

# Xem logs
docker-compose logs -f api

# Chỉ chạy một số services
docker-compose up -d db qdrant  # Chỉ DB
docker-compose up -d            # Tất cả
```

#### Option 3: Chạy từng phần (Hybrid)

```bash
# 1. Chạy DB với Docker
docker-compose up -d db qdrant

# 2. Chạy API local
uvicorn app.main:app --reload
```

### Kiểm tra ứng dụng:

```bash
# Health check
curl http://localhost:8000/health

# API docs
# Mở browser: http://localhost:8000/docs

# Chat UI
# Mở browser: http://localhost:8000/static/chat.html
```

---

## 7️⃣ TRUY CẬP VÀ QUẢN LÝ DỮ LIỆU

### A. Truy cập PostgreSQL

#### Cách 1: Dùng psql (Command line)

```bash
# Nếu PostgreSQL chạy trong Docker
docker exec -it chat-postgres psql -U chat_user -d chat_db

# Nếu PostgreSQL chạy local
psql -h localhost -U chat_user -d chat_db
```

**Các lệnh SQL hữu ích:**
```sql
-- Xem tất cả tables
\dt

-- Xem cấu trúc table
\d users
\d chat_sessions
\d messages

-- Xem tất cả users
SELECT * FROM users;

-- Xem tất cả sessions
SELECT * FROM chat_sessions;

-- Xem messages của một session
SELECT * FROM messages WHERE session_id = 'your-session-id';

-- Đếm số lượng
SELECT COUNT(*) FROM users;
SELECT COUNT(*) FROM chat_sessions;
SELECT COUNT(*) FROM messages;

-- Xem sessions với số lượng messages
SELECT 
    cs.session_id,
    cs.session_name,
    cs.started_at,
    COUNT(m.message_id) as message_count
FROM chat_sessions cs
LEFT JOIN messages m ON cs.session_id = m.session_id
GROUP BY cs.session_id
ORDER BY cs.started_at DESC;

-- Xem user với số sessions
SELECT 
    u.user_id,
    u.email,
    u.full_name,
    COUNT(cs.session_id) as session_count
FROM users u
LEFT JOIN chat_sessions cs ON u.user_id = cs.user_id
GROUP BY u.user_id;

-- Xóa dữ liệu test
DELETE FROM messages WHERE session_id = 'session-id';
DELETE FROM chat_sessions WHERE session_id = 'session-id';
DELETE FROM users WHERE email LIKE 'user_%@example.com';

-- Thoát
\q
```

#### Cách 2: Dùng PgAdmin (GUI)

```bash
# Khởi động PgAdmin
docker-compose up -d pgadmin

# Truy cập: http://localhost:8080
# Login:
#   Email: admin@example.com
#   Password: admin
```

**Kết nối database trong PgAdmin:**
1. Click "Add New Server"
2. General tab:
   - Name: Chat DB
3. Connection tab:
   - Host: db (nếu dùng Docker) hoặc localhost
   - Port: 5432
   - Database: chat_db
   - Username: chat_user
   - Password: chat_password
4. Save

#### Cách 3: Dùng Python script

```python
# File: view_db.py
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, text
from app.models.base import User, ChatSession, Message

DATABASE_URL = "postgresql+asyncpg://chat_user:chat_password@localhost:5432/chat_db"

async def view_all_data():
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # Users
        print("=" * 50)
        print("USERS:")
        print("=" * 50)
        result = await session.execute(select(User))
        users = result.scalars().all()
        for user in users:
            print(f"ID: {user.user_id}")
            print(f"Email: {user.email}")
            print(f"Name: {user.full_name}")
            print(f"Created: {user.created_at}")
            print("-" * 50)
        
        # Sessions
        print("\n" + "=" * 50)
        print("CHAT SESSIONS:")
        print("=" * 50)
        result = await session.execute(select(ChatSession))
        sessions = result.scalars().all()
        for sess in sessions:
            print(f"ID: {sess.session_id}")
            print(f"User: {sess.user_id}")
            print(f"Name: {sess.session_name}")
            print(f"Started: {sess.started_at}")
            print("-" * 50)
        
        # Messages
        print("\n" + "=" * 50)
        print("MESSAGES:")
        print("=" * 50)
        result = await session.execute(select(Message).limit(10))
        messages = result.scalars().all()
        for msg in messages:
            print(f"ID: {msg.message_id}")
            print(f"Session: {msg.session_id}")
            print(f"Type: {msg.message_type}")
            print(f"Content: {msg.content[:100]}...")
            print(f"Created: {msg.created_at}")
            print("-" * 50)
        
        # Statistics
        print("\n" + "=" * 50)
        print("STATISTICS:")
        print("=" * 50)
        user_count = await session.execute(text("SELECT COUNT(*) FROM users"))
        session_count = await session.execute(text("SELECT COUNT(*) FROM chat_sessions"))
        message_count = await session.execute(text("SELECT COUNT(*) FROM messages"))
        
        print(f"Total Users: {user_count.scalar()}")
        print(f"Total Sessions: {session_count.scalar()}")
        print(f"Total Messages: {message_count.scalar()}")

if __name__ == "__main__":
    asyncio.run(view_all_data())
```

Chạy script:
```bash
python view_db.py
```

### B. Truy cập Qdrant

#### Cách 1: Qdrant Dashboard (GUI)
```bash
# Mở browser
http://localhost:6333/dashboard

# Xem collections, vectors, search
```

#### Cách 2: REST API
```bash
# Xem tất cả collections
curl http://localhost:6333/collections

# Xem chi tiết collection
curl http://localhost:6333/collections/math_philosophy

# Search vectors
curl -X POST http://localhost:6333/collections/math_philosophy/points/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, ...],
    "limit": 5
  }'

# Xem một vector cụ thể
curl http://localhost:6333/collections/math_philosophy/points/1
```

#### Cách 3: Python script
```python
# File: view_qdrant.py
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

# Xem tất cả collections
collections = client.get_collections()
print("Collections:")
for col in collections.collections:
    print(f"  - {col.name}")

# Xem chi tiết collection
collection_name = "math_philosophy"
info = client.get_collection(collection_name)
print(f"\nCollection: {collection_name}")
print(f"Vectors count: {info.vectors_count}")
print(f"Points count: {info.points_count}")

# Xem một số vectors
points = client.scroll(
    collection_name=collection_name,
    limit=5,
    with_payload=True,
    with_vectors=False
)
print(f"\nSample points:")
for point in points[0]:
    print(f"ID: {point.id}")
    print(f"Payload: {point.payload}")
    print("-" * 50)
```

Chạy:
```bash
python view_qdrant.py
```

---

## 8️⃣ TROUBLESHOOTING

### Lỗi: "Collection not found: math_philosophy"

**Nguyên nhân:** Chưa ingest dữ liệu vào Qdrant

**Giải pháp:**
```bash
# 1. Kiểm tra Qdrant đang chạy
curl http://localhost:6333/health

# 2. Ingest dữ liệu
python ingest_data.py

# 3. Kiểm tra collection đã tạo
curl http://localhost:6333/collections
```

### Lỗi: "Foreign key violation - user not found"

**Nguyên nhân:** User chưa tồn tại trong database

**Giải pháp:**
```bash
# Clear localStorage trong browser
# F12 > Console:
localStorage.clear();
location.reload();

# Hoặc tạo user thủ công
python create_user.py
```

### Lỗi: "Connection refused" khi kết nối PostgreSQL

**Giải pháp:**
```bash
# Kiểm tra PostgreSQL đang chạy
docker ps | grep postgres

# Nếu không chạy, start lại
docker-compose up -d db

# Kiểm tra logs
docker-compose logs db

# Test kết nối
psql -h localhost -U chat_user -d chat_db
```

### Lỗi: "Connection refused" khi kết nối Qdrant

**Giải pháp:**
```bash
# Kiểm tra Qdrant đang chạy
docker ps | grep qdrant

# Start Qdrant
docker-compose up -d qdrant

# Kiểm tra logs
docker-compose logs qdrant

# Test kết nối
curl http://localhost:6333/health
```

### Lỗi: "Module not found"

**Giải pháp:**
```bash
# Cài lại dependencies
pip install -r requirements.txt

# Hoặc cài từng package thiếu
pip install <package-name>
```

### Lỗi: "Alembic migration failed"

**Giải pháp:**
```bash
# Reset database
docker-compose down -v
docker-compose up -d db

# Chạy lại migrations
alembic upgrade head

# Nếu vẫn lỗi, xóa alembic version
psql -h localhost -U chat_user -d chat_db -c "DROP TABLE IF EXISTS alembic_version;"
alembic upgrade head
```

### Docker container không start

**Giải pháp:**
```bash
# Xem logs chi tiết
docker-compose logs <service-name>

# Restart service
docker-compose restart <service-name>

# Rebuild image
docker-compose build --no-cache <service-name>
docker-compose up -d <service-name>

# Xóa tất cả và start lại
docker-compose down -v
docker-compose up -d
```

---

## 📝 CHECKLIST SETUP HOÀN CHỈNH

### Trước khi chạy ứng dụng:

- [ ] Docker Desktop đang chạy
- [ ] File .env đã được cấu hình đúng
- [ ] PostgreSQL container đang chạy (`docker ps | grep postgres`)
- [ ] Qdrant container đang chạy (`docker ps | grep qdrant`)
- [ ] Database migrations đã chạy (`alembic upgrade head`)
- [ ] Dữ liệu đã được ingest vào Qdrant
- [ ] Virtual environment đã được activate
- [ ] Dependencies đã được cài đặt

### Test từng component:

```bash
# 1. Test PostgreSQL
psql -h localhost -U chat_user -d chat_db -c "SELECT 1;"

# 2. Test Qdrant
curl http://localhost:6333/health

# 3. Test API
curl http://localhost:8000/health

# 4. Test RAG
python app/service/RAG/main.py
```

---

## 🎯 QUICK START (TL;DR)

```bash
# 1. Start services
docker-compose up -d db qdrant

# 2. Setup Python
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env với database và API keys

# 4. Setup database
alembic upgrade head

# 5. Ingest data
python ingest_data.py

# 6. Run app
uvicorn app.main:app --reload

# 7. Open browser
http://localhost:8000/static/chat.html
```

---

**Chúc bạn setup thành công! 🎉**

Nếu gặp vấn đề, check lại từng bước trong phần Troubleshooting.
