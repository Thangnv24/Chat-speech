

# 📚 Backend Design Document — RAG + Speech Chat System

## 🧭 Mục Lục

* [🏗️ Thiết kế tổng quan](#️-thiết-kế-tổng-quan)
* [📊 Database Schema](#-database-schema)
* [📁 Folder Structure](#-folder-structure)
* [🛠️ Frameworks & Dependencies](#️-frameworks--dependencies)
* [🔄 API Endpoints](#-api-endpoints)
* [🔌 WebSocket Design](#-websocket-design)
* [🏢 Service Layer Design](#-service-layer-design)
* [💾 Caching Strategy](#-caching-strategy)
* [🔄 Integration Flow](#-integration-flow)
* [📱 Frontend Integration](#-frontend-integration)
* [🔐 Security Considerations](#-security-considerations)
* [📊 Monitoring & Analytics](#-monitoring--analytics)
* [🚀 Implementation Phases](#-implementation-phases)
* [🎯 Key Design Decisions](#-key-design-decisions)

---

# 🏗️ Thiết Kế Tổng Quan

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                    CHAT BACKEND SYSTEM                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Frontend (Web/Mobile)                                      │
│         ↓                                                   │
│  FastAPI REST API + WebSocket                               │
│         ↓                                                   │
│  Business Logic Layer                                       │
│    - Chat Service                                           │
│    - RAG Service (existing)                                 │
│    - Speech Service (existing)                              │
│         ↓                                                   │
│  Database Layer (PostgreSQL)                                │
│    - Users                                                  │
│    - Chat Sessions                                          │
│    - Messages                                               │
│    - RAG Context                                            │
│         ↓                                                   │
│  External Services                                          │
│    - Qdrant (Vector DB)                                     │
│    - LLM APIs (Gemini/Groq)                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

# 📊 Database Schema

## 1. **Users Table**

| Field          | Type      | Notes                    |
| -------------- | --------- | ------------------------ |
| id             | UUID (PK) |                          |
| username       | VARCHAR   | UNIQUE                   |
| email          | VARCHAR   | UNIQUE, nullable         |
| full_name      | VARCHAR   | nullable                 |
| avatar_url     | VARCHAR   | nullable                 |
| preferences    | JSONB     | chat settings, language… |
| created_at     | TIMESTAMP |                          |
| updated_at     | TIMESTAMP |                          |
| last_active_at | TIMESTAMP |                          |
| is_active      | BOOLEAN   | default TRUE             |

---

## 2. **Chat Sessions Table**

| Field           | Type                         |
| --------------- | ---------------------------- |
| id              | UUID (PK)                    |
| user_id         | UUID (FK → users.id)         |
| title           | VARCHAR                      |
| session_type    | ENUM('rag','speech','mixed') |
| rag_config      | JSONB                        |
| speech_config   | JSONB                        |
| metadata        | JSONB                        |
| created_at      | TIMESTAMP                    |
| updated_at      | TIMESTAMP                    |
| last_message_at | TIMESTAMP                    |
| is_archived     | BOOLEAN                      |

---

## 3. **Messages Table**

| Field           | Type                                |
| --------------- | ----------------------------------- |
| id              | UUID (PK)                           |
| session_id      | UUID (FK → chat_sessions.id)        |
| message_type    | ENUM('user','assistant','system')   |
| content_type    | ENUM('text','audio','image','file') |
| content         | TEXT                                |
| audio_url       | VARCHAR (nullable)                  |
| rag_context     | JSONB                               |
| processing_time | FLOAT                               |
| tokens_used     | INTEGER                             |
| created_at      | TIMESTAMP                           |
| edited_at       | TIMESTAMP                           |
| is_deleted      | BOOLEAN                             |

---

## 4. **RAG Context Table (Optional)**

| Field               | Type                    |
| ------------------- | ----------------------- |
| id                  | UUID (PK)               |
| message_id          | UUID (FK → messages.id) |
| query               | TEXT                    |
| search_mode         | VARCHAR                 |
| retrieved_documents | JSONB                   |
| generation_prompt   | TEXT                    |
| llm_provider        | VARCHAR                 |
| created_at          | TIMESTAMP               |

---

# 📁 Folder Structure

```
app/
├── models/
│   ├── user.py
│   ├── chat_session.py
│   ├── message.py
│   └── rag_context.py
│
├── schemas/
│   ├── user.py
│   ├── chat.py
│   └── rag.py
│
├── services/
│   ├── chat_service.py
│   ├── user_service.py
│   ├── message_service.py
│   └── session_service.py
│
├── routers/
│   ├── chat.py
│   ├── websocket.py
│   ├── users.py
│   └── speech.py
│
├── core/
│   ├── database.py
│   ├── websocket_manager.py
│   └── dependencies.py
│
├── utils/
│   ├── session_utils.py
│   ├── message_utils.py
│   └── audio_utils.py
│
└── migrations/
    └── versions/
        ├── 001_create_users.py
        ├── 002_create_chat_sessions.py
        ├── 003_create_messages.py
        └── 004_create_rag_contexts.py
```

---

# 🛠️ Frameworks & Dependencies

### Core Backend

```txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
sqlalchemy>=2.0.0
asyncpg>=0.29.0
alembic>=1.13.0
pydantic>=2.5.0
```

### New Additions

```txt
websockets>=11.0.0
python-socketio>=5.9.0
redis>=4.6.0
celery>=5.3.0
```

### Authentication (Optional)

```txt
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-multipart>=0.0.6
```

### File Storage

```txt
aiofiles>=23.2.0
boto3>=1.28.0
```

---

# 🔄 API Endpoints

## 1. User Management

```
POST   /api/v1/users/register
POST   /api/v1/users/login
GET    /api/v1/users/me
PUT    /api/v1/users/me
GET    /api/v1/users/{user_id}
```

## 2. Chat Sessions

```
GET    /api/v1/chat/sessions
POST   /api/v1/chat/sessions
GET    /api/v1/chat/sessions/{id}
PUT    /api/v1/chat/sessions/{id}
DELETE /api/v1/chat/sessions/{id}
```

## 3. Messages

```
GET    /api/v1/chat/sessions/{id}/messages
POST   /api/v1/chat/sessions/{id}/messages
PUT    /api/v1/chat/messages/{id}
DELETE /api/v1/chat/messages/{id}
```

## 4. RAG Integration

```
POST   /api/v1/chat/sessions/{id}/rag-query
GET    /api/v1/chat/messages/{id}/rag-context
```

## 5. Speech Integration

```
POST   /api/v1/chat/sessions/{id}/speech
POST   /api/v1/chat/messages/{id}/tts
```

## 6. WebSocket

```
WS     /ws/chat/{session_id}
```

---

# 🔌 WebSocket Design

## Connection

```
ws://localhost:8000/ws/chat/{session_id}?user_id={user_id}
```

## Message Types

```json
{
  "type": "user_message",
  "content": "Hello",
  "content_type": "text"
}
```

```json
{
  "type": "assistant_response",
  "content": "Hi there!",
  "rag_context": {},
  "processing_time": 1.23
}
```

```json
{
  "type": "typing_indicator",
  "is_typing": true
}
```

```json
{
  "type": "error",
  "message": "Something went wrong"
}
```

---

# 🏢 Service Layer Design

## ChatService

```python
class ChatService:
    async def create_session(user_id, session_type, config)
    async def get_user_sessions(user_id, limit, offset)
    async def process_message(session_id, content, content_type)
    async def get_session_messages(session_id, limit, offset)
    async def archive_session(session_id)
```

## MessageService

```python
class MessageService:
    async def create_message(session_id, content, message_type)
    async def process_rag_message(session_id, query, config)
    async def process_speech_message(session_id, audio_data)
    async def get_message_history(session_id, limit)
    async def update_message(message_id, content)
```

## RAGIntegrationService

```python
class RAGIntegrationService:
    async def query_with_session_context(session_id, query)
    async def get_session_rag_history(session_id)
    async def update_rag_config(session_id, config)
```

---

# 💾 Caching Strategy

| Cache Key               | Data             | TTL |
| ----------------------- | ---------------- | --- |
| `session:{id}`          | session data     | 1h  |
| `user:{id}`             | user profile     | 30m |
| `messages:{session_id}` | last 20 messages | 1h  |
| `rag_context:{hash}`    | RAG results      | 24h |

---

# 🔄 Integration Flow

### Typical Chat Flow

1. User connects via WebSocket
2. Frontend sends message
3. Backend saves to DB
4. RAG query → call pipeline
5. Speech → audio processing
6. LLM generates response
7. Save assistant response
8. Send via WebSocket
9. Update session metadata

### RAG Example (MessageService)

```python
async def process_rag_message(self, session_id: str, query: str):
    user_msg = await self.create_message(session_id, query, "user")

    from app.service.RAG.rag_pipeline import create_pipeline
    pipeline = create_pipeline()
    rag_result = await pipeline.query(query, k=5)

    assistant_msg = await self.create_message(
        session_id,
        rag_result["answer"],
        "assistant",
        rag_context=rag_result
    )

    return assistant_msg
```

---

# 📱 Frontend Integration

```javascript
const ws = new WebSocket(
  `ws://localhost:8000/ws/chat/${sessionId}?user_id=${userId}`
);

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    switch(data.type) {
        case 'assistant_response':
            displayMessage(data.content);
            break;
        case 'typing_indicator':
            showTypingIndicator(data.is_typing);
            break;
    }
};

function sendMessage(content) {
    ws.send(JSON.stringify({
        type: 'user_message',
        content: content,
        content_type: 'text'
    }));
}
```

---

# 🔐 Security Considerations

### Authentication

* JWT tokens cho REST API
* WebSocket auth via query params / headers
* Rate limiting

### Data Protection

* Encrypt sensitive content
* Audit logs
* GDPR data deletion

---

# 📊 Monitoring & Analytics

### Session Metrics

* Active sessions
* Avg session duration
* Messages per session

### Performance

* RAG response time
* Speech processing time
* WebSocket connection count

### Usage

* Popular queries
* User engagement
* Error rates

---

# 🚀 Implementation Phases

### **Phase 1 — Core Chat (Week 1)**

* Models & migrations
* CRUD services
* REST API
* Basic WebSocket

### **Phase 2 — RAG Integration (Week 2)**

* Connect RAG pipeline
* Session-aware queries
* Context persistence

### **Phase 3 — Speech Integration (Week 3)**

* Audio upload
* Speech-to-text
* Text-to-speech

### **Phase 4 — Advanced (Week 4)**

* Redis caching
* Celery tasks
* Analytics
* Performance tuning

---

# 🎯 Key Design Decisions

### 1. Session Management

* Stored in DB + cached in Redis
* Auto-title from first message
* Archive instead of delete

### 2. Message Storage

* Immutable (soft delete only)
* Support text/audio/image/file
* Embedded RAG context

### 3. Real-Time Infrastructure

* WebSocket real-time
* Redis pub/sub optional
* Graceful reconnection

### 4. Service Integration

* Clean service layer separation
* Async processing
* Fallback for failures

---

# ✅ Lộ trình triển khai tiếp theo (ngắn gọn – đúng trọng tâm)

### **1. Database Models + Migrations**

Tạo bảng **Users**, **ChatSessions**, **Messages**, **RAGContexts** bằng SQLAlchemy + Alembic.

### **2. Service Layer (Backend Logic)**

Viết logic cho:

* `UserService`
* `SessionService`
* `MessageService` (gọi RAG + Speech)
  *→ Đây là lõi xử lý backend, cần xong trước khi viết API.*

### **3. REST API Routers (CRUD)**

Tạo các endpoint:

* `/users/*`
* `/chat/sessions/*`
* `/chat/messages/*`
* `/chat/sessions/{id}/rag-query`
* `/chat/sessions/{id}/speech`

### **4. WebSocket Chat**

* Tạo `/ws/chat/{session_id}`
* Kết nối WebSocketManager
* Broadcast message theo session
* Tích hợp MessageService vào WebSocket flow

### **5. Caching (Redis)** – *không bắt buộc lúc đầu*

* Cache session
* Cache last messages
* Cache rag results

### **6. Analytics + Monitoring + Optimize**

Làm sau cùng để tránh mất thời gian sớm.

---

# 🎯 Tóm tắt 1 câu

**Bạn nên làm theo thứ tự: Models → Services → REST API → WebSocket → Redis Cache → Monitoring.**

---

