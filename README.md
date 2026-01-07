# 🤖 Chat-speech System - Comprehensive Documentation

## 📋 Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [RAG Pipeline - Detailed Flow](#rag-pipeline---detailed-flow)
4. [Speech Processing - Detailed Flow](#speech-processing---detailed-flow)
5. [Features](#features)
6. [Installation](#installation)
7. [Usage](#usage)
8. [API Endpoints](#api-endpoints)

---

## 🎯 Overview

Chat RAG System là một ứng dụng AI tích hợp:
- **RAG (Retrieval-Augmented Generation)**: Tìm kiếm thông tin từ vector store
- **Voice Chat**: Giao tiếp bằng giọng nói (STT + TTS)
- **Hybrid Search**: Kết hợp dense + sparse search
- **Multi-language**: Hỗ trợ tiếng Việt và tiếng Anh
- **Specialized Processing**: Xử lý chuyên biệt cho Toán học và Triết học

---

## 🏗️ Architecture

```
mini_pj/
├── alembic/                # Quản lý migration database (SQLAlchemy/Alembic)
├── app/                    # Mã nguồn chính của ứng dụng FastAPI
│   ├── config/             # Cấu hình LLM, prompts và các tham số hệ thống
│   ├── core/               # Các thành phần cốt lõi (Kết nối DB, LLM Client, Exception Handlers)
│   ├── crud/               # Các hàm thực hiện thao tác CRUD (Create, Read, Update, Delete) với DB
│   ├── models/             # Định nghĩa các bảng cơ sở dữ liệu (SQLAlchemy models)
│   ├── routers/            # Định nghĩa các API endpoints (chia theo module: auth, chat, voice...)
│   ├── service/            # Logic nghiệp vụ (Business Logic) chính
│   │   ├── RAG/            # Xử lý Ingest tài liệu, chunking và pipeline RAG
│   │   └── speech/         # Dịch vụ Speech-to-Text (STT) và Text-to-Speech (TTS)
│   ├── utils/              # Các hàm tiện ích dùng chung trong project
│   ├── main.py             # Điểm khởi tạo và chạy ứng dụng FastAPI
│   └── schemas.py          # Định nghĩa kiểu dữ liệu Input/Output (Pydantic schemas)
├── data/ / dataset.json    # Dữ liệu đầu vào, tài liệu PDF/Text để Ingest vào Vector DB
├── static/                 # Chứa các file giao diện web tĩnh (HTML, CSS, JS cho Chat UI)
├── logs/                   # Thư mục lưu trữ nhật ký hoạt động (logs) của hệ thống
├── qdrant_storage/         # Dữ liệu lưu trữ cục bộ của Vector Database Qdrant
├── vector_data/            # (Tùy chọn) Lưu trữ các file vector store khác (như FAISS index)
├── .env                    # Lưu trữ biến môi trường (API Key, Database URL...)
├── Dockerfile              # File cấu hình build image cho ứng dụng
├── docker-compose.yml      # Cấu hình khởi chạy các dịch vụ (Postgres, Qdrant...) bằng Docker
├── Makefile                # Các lệnh tắt (shortcuts) để chạy/build project nhanh
├── pyproject.toml          # Cấu hình tool và dependencies (nếu dùng poetry/uv)
├── requirements.txt        # Danh sách các thư viện Python cần cài đặt
├── create_user.py          # Script tiện ích để tạo tài liệu user thủ công
├── view_db.py              # Script kiểm tra dữ liệu trực tiếp trong Database
└── view_qdrant.py          # Script kiểm tra các points trong Vector DB Qdrant


┌─────────────────────────────────────────────────────────────┐
│                    Chat RAG System                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │   Voice Input    │         │   Text Input     │         │
│  │   (Microphone)   │         │   (Chat UI)      │         │
│  └────────┬─────────┘         └────────┬─────────┘         │
│           │                            │                   │
│           ▼                            ▼                   │
│  ┌──────────────────────────────────────────────┐          │
│  │         Speech-to-Text (STT)                 │          │
│  │  Groq Whisper large-v3-turbo                 │          │
│  │  - Audio Preprocessing                       │          │
│  │  - Noise Reduction                           │          │
│  │  - Voice Activity Detection (VAD)            │          │
│  └────────────────┬─────────────────────────────┘          │
│                   │                                        │
│                   ▼                                        │
│  ┌──────────────────────────────────────────────┐          │
│  │         Query Processing                     │          │
│  │  - Query Type Detection (Math/Philosophy)    │          │
│  │  - Query Expansion (optional)                │          │
│  └────────────────┬─────────────────────────────┘          │
│                   │                                        │
│                   ▼                                        │
│  ┌──────────────────────────────────────────────┐          │
│  │      Hybrid Retrieval (RAG)                  │          │
│  │  - Dense Search (Embeddings)                 │          │
│  │  - Sparse Search (BM25/TF-IDF)               │          │
│  │  - Reranking (Cross-Encoder)                 │          │
│  └────────────────┬─────────────────────────────┘          │
│                   │                                        │
│                   ▼                                        │
│  ┌──────────────────────────────────────────────┐          │
│  │      LLM Answer Generation                   │          │
│  │  - Gemini / Qwen / Ollama                    │          │
│  │  - Context-aware Prompting                   │          │
│  └────────────────┬─────────────────────────────┘          │
│                   │                                        │
│                   ▼                                        │
│  ┌──────────────────────────────────────────────┐          │
│  │      Text-to-Speech (TTS)                    │          │
│  │  ElevenLabs - Multi-language Support         │          │
│  │  - Symbol Normalization                      │          │
│  │  - Language Detection                        │          │
│  └────────────────┬─────────────────────────────┘          │
│                   │                                        │
│                   ▼                                        │
│  ┌──────────────────────────────────────────────┐          │
│  │      Voice Output (Speaker)                  │          │
│  └──────────────────────────────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 RAG Pipeline - Detailed Flow

### 1️⃣ **Document Ingestion & Chunking**

#### A. Document Type Detection
```python
# app/service/RAG/chunking.py - AdvancedTextChunker

def detect_document_type(text: str) -> str:
    """
    Phát hiện loại tài liệu dựa trên pattern matching
    
    Returns:
    - "mathematics": Nếu có >70% toán học keywords
    - "philosophy": Nếu có >70% triết học keywords
    - "mixed": Nếu cân bằng giữa hai loại
    - "general": Nếu không có keywords đặc biệt
    """
```

**Math Keywords**: định lý, chứng minh, công thức, phương trình, tính toán
**Philosophy Keywords**: triết, quan điểm, học thuyết, tư tưởng, khái niệm

#### B. Mathematical Structure Extraction
```python
# Trích xuất các cấu trúc toán học quan trọng:

Math Patterns:
├── Theorem (Định lý)
├── Proof (Chứng minh)
├── Definition (Định nghĩa)
├── Equation (Phương trình)
├── Formula (Công thức)
└── Example (Ví dụ)

Philosophy Patterns:
├── Concept (Khái niệm)
├── Argument (Luận điểm)
├── Critique (Phê phán)
├── Doctrine (Học thuyết)
└── Dialectic (Biện chứng)
```

#### C. Context-Aware Chunking

**Cho Toán học:**
```
1. Trích xuất cấu trúc toán học
2. Chia theo sections (định lý, chứng minh, ví dụ)
3. Bảo toàn công thức và phương trình
4. Recursive splitting cho sections lớn
5. Metadata: math_structures, contains_equations
```

**Cho Triết học:**
```
1. Chia theo paragraphs
2. Chia theo philosophical concepts
3. Giữ nguyên luận điểm
4. Semantic clustering (TF-IDF + K-means)
5. Metadata: philosophy_structures
```

**Cho Mixed:**
```
1. Kết hợp cả hai phương pháp
2. Recursive splitting với separators
3. Metadata: cả math_structures và philosophy_structures
```

#### D. Semantic Chunking (TF-IDF + K-means)
```python
# Nhóm chunks theo semantic similarity

Process:
1. TF-IDF Vectorization (1000 features)
   - Tần suất thuật ngữ - tần suất tài liệu nghịch đảo
   - Loại bỏ stop words
   - Min_df=2, Max_df=0.8

2. K-means Clustering
   - Nhóm chunks thành clusters
   - Mỗi cluster = một chủ đề

3. Output: List[List[str]]
   - Mỗi sublist = một cluster
```

**Metadata được lưu:**
```python
{
    "chunk_id": int,
    "document_type": "mathematics|philosophy|mixed|general",
    "math_structures": ["theorem", "proof", "equation"],
    "philosophy_structures": ["concept", "argument"],
    "contains_equations": bool,
    "chunk_size": int,
    "source": "doc_1.pdf",
    "source_path": "./data/doc_1.pdf"
}
```

---

### 2️⃣ **Embedding & Vector Storage**

#### A. Dense Embeddings
```python
# Model: sentence-transformers/all-MiniLM-L6-v2
# Output: 384-dimensional vectors

Process:
1. Chunk text → Embedding model
2. Generate 384-dim dense vector
3. Store in Qdrant with COSINE distance
```

#### B. Sparse Embeddings (BM25)
```python
# Tokenization + TF-IDF

Process:
1. Tokenize text (Vietnamese: underthesea)
2. Count token frequencies
3. Top 100 tokens → sparse vector
4. Hash tokens to indices (0-10000)
5. Store in Qdrant with BM25 index
```

#### C. Hybrid Vector Store
```python
# HybridVectorStore - app/service/RAG/ingestor.py

class HybridVectorStore:
    - Dense vectors: COSINE distance
    - Sparse vectors: BM25 index
    - Thread-safe tokenization
    - RRF (Reciprocal Rank Fusion) for combining results
```

**Qdrant Collection Structure:**
```
Collection: math_philosophy
├── Dense vectors (384-dim)
├── Sparse vectors (BM25)
├── Payloads:
│   ├── text: chunk content
│   ├── metadata: document info
│   └── source: file path
└── Points: ~1000-10000 (tùy dữ liệu)
```

---

### 3️⃣ **Retrieval & Search**

#### A. Query Type Detection
```python
# HybridRetriever._detect_query_type()

Math keywords: toán, tính, phương trình, định lý, chứng minh, công thức
Philosophy keywords: triết, quan điểm, học thuyết, tư tưởng, khái niệm

Score = count(keywords in query)
Return: "mathematics" | "philosophy" | "general"
```

#### B. Hybrid Search Process
```
Query Input
    ↓
1. Dense Search
   - Embed query → 384-dim vector
   - COSINE similarity search
   - Top-k results (k*3 for reranking)
   
2. Sparse Search
   - Tokenize query
   - BM25 search
   - Top-k results (k*3 for reranking)

3. Combine Results (RRF)
   - Reciprocal Rank Fusion
   - Formula: score = 1/(k + rank)
   - Merge dense + sparse scores
   - Top-k final results

4. Reranking (Cross-Encoder)
   - Model: BAAI/bge-reranker-v2-m3
   - Re-score top-k results
   - Final ranking
```

**Search Modes:**
```python
- "dense": Dense search only
- "sparse": Sparse search only (BM25)
- "hybrid": Dense + Sparse + RRF (default)
```

#### C. Context Preparation
```python
# Prepare context for LLM

For each retrieved document:
1. Get document type (math/philosophy/mixed)
2. Extract structures (theorems, proofs, concepts)
3. Format with confidence scores
4. Add source information

Output:
[Tài liệu 1 - Độ tin cậy: 0.85]
Loại: mathematics
Cấu trúc Toán: theorem, proof
Nội dung: ...

[Tài liệu 2 - Độ tin cậy: 0.72]
...
```

---

### 4️⃣ **Answer Generation**

#### A. Prompt Selection
```python
# app/config/prompts.py - PROMPT_MAP

Prompts by query type:
├── mathematics: Math-specific prompt
├── philosophy: Philosophy-specific prompt
└── general: General prompt

Each prompt includes:
- System instruction
- Context placeholder
- Question placeholder
- Output format
```

#### B. LLM Integration
```python
# app/config/llm_config.py

Supported LLMs:
├── Gemini (Google)
├── Qwen (Alibaba)
└── Ollama (Local)

Process:
1. Select LLM based on config
2. Format prompt with context + question
3. Generate answer
4. Return response
```

#### C. Answer Generation Flow
```
Context + Question
    ↓
Format with Prompt Template
    ↓
Send to LLM
    ↓
Stream/Get Response
    ↓
Post-process (cleanup, formatting)
    ↓
Return Answer
```

---

## 🎤 Speech Processing - Detailed Flow

### 1️⃣ **Speech-to-Text (STT)**

#### A. Audio Recording
```python
# app/service/speech/stt.py - GroqSTT._record_audio()

Process:
1. Initialize PyAudio stream
   - Sample rate: 16kHz
   - Channels: 1 (mono)
   - Format: 16-bit PCM
   - Chunk size: 1024 samples

2. Voice Activity Detection (VAD)
   - Threshold: 1000 (volume level)
   - Detect speaking vs silence
   - Track silent chunks

3. Recording Logic
   - Wait for speech (timeout: 5s)
   - Record while speaking
   - Stop after silence (2s)
   - Max length: 30s (Groq limit)

4. Output: numpy array (int16)
```

#### B. Audio Preprocessing
```python
# AudioPreprocessor class

Step 1: Noise Reduction
├── Library: noisereduce
├── Method: Stationary noise reduction
├── Reduction: 80% (prop_decrease=0.8)
└── Output: Cleaner audio

Step 2: Normalization
├── Divide by max amplitude
├── Range: [-1, 1]
└── Prevent clipping

Step 3: Silence Trimming
├── Find non-silent samples (threshold: 500)
├── Keep 0.1s before/after
└── Remove leading/trailing silence

Step 4: Format Conversion
├── Convert to int16
├── Create WAV bytes
└── Ready for Groq API
```

#### C. Transcription
```python
# Groq Whisper large-v3-turbo

API Call:
POST https://api.groq.com/audio/transcriptions
├── Model: whisper-large-v3-turbo
├── Language: vi | en
├── Format: verbose_json
└── Response: text, duration, language

Output:
{
    "text": "Định lý Pytago là gì?",
    "duration": 3.5,
    "processing_time": 1.2,
    "language": "vi"
}
```

---

### 2️⃣ **Text-to-Speech (TTS)**

#### A. Language Detection
```python
# detect_language(text)

Library: langdetect
├── Detect language from text
├── Support: Vietnamese (vi), English (en)
└── Fallback: English

Output: "vi" | "en"
```

#### B. Text Normalization
```python
# normalize_text_for_reading(text, lang)

Process:
1. Symbol Replacement
   Math symbols:
   ├── + → " cộng "
   ├── = → " bằng "
   ├── ∀ → " với mọi "
   ├── ∃ → " tồn tại "
   ├── → → " suy ra "
   └── ... (20+ symbols)

2. Exponent Conversion
   ├── x^2 → "x bình phương"
   ├── x^3 → "x lập phương"

3. Citation Removal
   ├── [1], [2], ... → removed

4. Punctuation Normalization
   ├── ; → ,
   ├── — → ,

5. Whitespace Cleanup
   ├── Multiple spaces → single space
   └── Trim edges
```

#### C. Voice Synthesis
```python
# ElevenLabs API

Vietnamese:
├── Voice ID: 1d5Bb0SMBPB10Gx6iQeu
├── Model: eleven_turbo_v2_5
├── SSML: <lang xml:lang="vi-VN">text</lang>
└── Settings: stability=0.5, similarity_boost=0.75

English:
├── Voice ID: FxZjRiAEBESrb7srpme7
├── Model: eleven_multilingual_v2
└── Settings: stability=0.5, similarity_boost=0.75

Output:
├── Format: PCM 16000Hz
├── Channels: 1 (mono)
└── Saved as: WAV file
```

---

# Kết quả / Metrics (RAG EVALUATION)
Kết quả đánh giá RAG (k=10) thu được trong quá trình kiểm thử:

---------------

RAG EVALUATION (k=10)
| Chỉ số (Metric) | Giá trị (Value) |
| :--- | :--- |
| **Precision@10** | 0.85 |
| **Recall@10** | 0.72 |
| **F1 Score** | 0.78 |
| **Context Precision** | 0.81 |
| **Faithfulness** | 0.92 |
| **Answer Relevance** | 0.88 |

-----
## ✨ Features

### 🎯 Core Features

#### 1. **Hybrid Search**
- ✅ Dense search (semantic similarity)
- ✅ Sparse search (keyword matching)
- ✅ RRF combination
- ✅ Cross-encoder reranking

#### 2. **Document Processing**
- ✅ PDF extraction
- ✅ Text file processing
- ✅ Automatic type detection
- ✅ Structure preservation

#### 3. **Voice Chat**
- ✅ Real-time recording
- ✅ Audio preprocessing
- ✅ Noise reduction
- ✅ VAD (Voice Activity Detection)
- ✅ Multi-language support

#### 4. **RAG System**
- ✅ Context-aware chunking
- ✅ Semantic clustering
- ✅ Hybrid embeddings
- ✅ Query type detection

#### 5. **Session Management**
- ✅ Create/list/delete sessions
- ✅ Rename sessions
- ✅ Message history
- ✅ User persistence

#### 6. **Metrics & Evaluation**
- ✅ Precision@k, Recall@k, F1
- ✅ RAGAS metrics (Faithfulness, Relevance)
- ✅ Context precision
- ✅ Performance tracking

---

## 🚀 Installation

### Prerequisites
```bash
# System requirements
- Python 3.11+
- PostgreSQL 14+
- Qdrant (Docker or local)
- FFmpeg (for audio processing)
```

### Setup Steps

#### 1. Clone & Setup
```bash
git clone <repo>
cd mini_pj

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Configure Environment
```bash
# Copy example
cp .env.example .env

# Edit .env with your keys
GROQ_API_KEY=your_groq_key
ELEVEN_API_KEY=your_elevenlabs_key
GEMINI_API_KEY=your_gemini_key
QDRANT_URL=http://localhost:6333
SQLALCHEMY_DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/chat_db
```

#### 3. Start Services

- Cài dependencies: 
  ```bash
  uv pip install -r requirements.txt
  ```
- Load biến môi trường 
  - Trên Windows PowerShell:
    ```powershell
    .\load_env.ps1
    ```
  - Nếu bạn dùng `.env` file, có thể chạy:
    ```bash
    export $(cat .env | xargs)          # macOS / Linux (tùy shell)
    # hoặc dùng python-dotenv trong code để load tự động
    ```

3) Migration DB (nếu dùng Alembic)
```bash
alembic upgrade head
```

4) Ingest tài liệu (RAG)
- Chạy script chunk & ingest (đảm bảo biến môi trường cho embeddings/vector store đã được cấu hình):
```bash
python app/service/RAG/chunk_and_ingest.py
```

5) Chạy server dev
```bash
uvicorn app.main:app --reload
```
Sau đó truy cập API tại http://localhost:8000/docs (Để test swagger)
Truy cập vào http://localhost:8000/static/chat.html để test chức năng chat và speech

---

## 💬 Usage

### Text Chat
```
http://localhost:8000/static/chat.html
```

### Voice Chat
```
http://localhost:8000/static/voice_chat.html
```

### API Documentation
```
http://localhost:8000/docs
```

---

## 📡 API Endpoints

### Chat Endpoints

#### Text Chat
```bash
POST /api/v1/chat/
Content-Type: application/json

{
  "query": "Định lý Pytago là gì?",
  "session_id": "uuid",
  "k": 5,
  "search_mode": "hybrid"
}

Response:
{
  "answer": "...",
  "query_time": 1.23,
  "num_retrieved": 5,
  "user_message": {...},
  "ai_message": {...}
}
```

#### Voice Chat
```bash
POST /api/v1/voice/chat?session_id=uuid&language=vi
Content-Type: multipart/form-data

audio_file: <binary>

Response:
{
  "transcribed_text": "Định lý Pytago là gì?",
  "answer": "...",
  "audio_duration": 3.5,
  "processing_time": 2.1,
  "query_time": 0.8,
  "num_retrieved": 5
}
```

#### Text-to-Speech
```bash
POST /api/v1/voice/tts?text=Hello&language=auto

Response: audio/wav
```

### Session Endpoints

#### Create Session
```bash
POST /api/v1/sessions/
{
  "user_id": "uuid"
}
```

#### List Sessions
```bash
GET /api/v1/sessions/
```

#### Rename Session
```bash
PATCH /api/v1/sessions/{session_id}/name?session_name=New%20Name
```

### Message Endpoints

#### Get Messages
```bash
GET /api/v1/messages/session/{session_id}
```



