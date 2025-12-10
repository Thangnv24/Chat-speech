# 📚 RAG System - Complete Guide

## Tổng Quan

Hệ thống RAG (Retrieval-Augmented Generation) kết hợp tìm kiếm thông tin và sinh văn bản để trả lời câu hỏi dựa trên tài liệu của bạn.

### Kiến Trúc

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG SYSTEM                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. INGESTION PHASE                                         │
│     Documents (PDF/TXT)                                     │
│            ↓                                                 │
│     Advanced Chunking (chunking.py)                         │
│       - Context-aware splitting                             │
│       - Math/Philosophy structure preservation              │
│       - Semantic clustering                                 │
│            ↓                                                 │
│     Hybrid Embeddings (ingestor.py)                         │
│       - Dense: sentence-transformers                        │
│       - Sparse: BM25 (Vietnamese tokenization)              │
│            ↓                                                 │
│     Vector Database (Qdrant)                                │
│       - Persistent storage                                  │
│       - Fast similarity search                              │
│                                                              │
│  2. RETRIEVAL PHASE                                         │
│     User Query                                              │
│            ↓                                                 │
│     Query Type Detection (retriever.py)                     │
│       - Mathematics                                         │
│       - Philosophy                                          │
│       - General                                             │
│            ↓                                                 │
│     Hybrid Search                                           │
│       - Dense vector search (semantic)                      │
│       - Sparse BM25 search (keyword)                        │
│       - Reciprocal Rank Fusion (RRF)                        │
│            ↓                                                 │
│     Top-K Relevant Documents                                │
│                                                              │
│  3. GENERATION PHASE                                        │
│     Context Preparation                                     │
│            ↓                                                 │
│     LLM Generation (Gemini/Qwen/Ollama)                     │
│       - Context-aware prompts                               │
│       - Domain-specific templates                           │
│            ↓                                                 │
│     Final Answer                                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Cài Đặt Dependencies

```bash
# Core dependencies
pip install qdrant-client langchain langchain-core
pip install sentence-transformers rank-bm25
pip install PyMuPDF scikit-learn underthesea

# Optional: LLM providers
pip install google-generativeai  # For Gemini
pip install groq                 # For Groq
```

### 2. Khởi Động Qdrant

```bash
# Option 1: Docker (Recommended)
docker run -p 6333:6333 qdrant/qdrant

# Option 2: Qdrant Cloud (Free tier)
# Sign up at: https://cloud.qdrant.io
```

### 3. Cấu Hình LLM

Thêm vào `.env`:
```env
# Choose one or more
GEMINI_API_KEY=your_gemini_key
QWEN_API_KEY=your_qwen_key
GROQ_API_KEY=your_groq_key
```

### 4. Chạy Demo

```bash
cd app/service/RAG
python main.py
```

---

## 📖 Luồng Hoạt Động Chi Tiết

### Phase 1: Document Ingestion

#### Step 1.1: Document Loading
```python
from app.service.RAG.chunking import get_chunker

chunker = get_chunker()
chunks = chunker.process_pdf("document.pdf")
```

**Xử lý:**
- Đọc PDF với PyMuPDF (fitz)
- Extract text từ mỗi trang
- Detect document type (Math/Philosophy/General)

#### Step 1.2: Advanced Chunking

**Context-Aware Chunking:**
- **Mathematics**: Bảo toàn định lý, chứng minh, công thức
- **Philosophy**: Giữ nguyên luận điểm, khái niệm
- **Mixed**: Kết hợp cả hai

**Techniques:**
- Regex pattern matching cho structures
- Recursive splitting với overlap
- Semantic clustering (TF-IDF + K-means)

**Output:**
```python
Document(
    page_content="Định lý Pythagore: a² + b² = c²...",
    metadata={
        "chunk_id": 0,
        "document_type": "mathematics",
        "math_structures": ["theorem", "equation"],
        "contains_equations": True,
        "chunk_size": 512
    }
)
```

#### Step 1.3: Hybrid Embeddings

**Dense Embeddings:**
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Captures semantic meaning

**Sparse Embeddings (BM25):**
- Vietnamese tokenization (underthesea)
- Term frequency analysis
- Keyword-based matching

#### Step 1.4: Vector Storage (Qdrant)

```python
from app.service.RAG.ingestor import DocumentIngestor

ingestor = DocumentIngestor(
    qdrant_url="http://localhost:6333",
    collection_name="math_philosophy"
)

vector_store = ingestor.ingest_documents([
    "data/doc_1.pdf",
    "data/doc_2.txt"
])
```

**Qdrant Features:**
- Persistent storage
- Fast HNSW indexing
- Metadata filtering
- Scalable to millions of vectors

---

### Phase 2: Retrieval

#### Step 2.1: Query Processing

```python
from app.service.RAG.retriever import create_retriever

retriever = create_retriever(vector_store, search_type="hybrid")
result = retriever.retrieve(
    query="Định lý Pythagore là gì?",
    k=5
)
```

**Query Type Detection:**
- Keyword matching
- Math keywords: toán, tính, phương trình, định lý
- Philosophy keywords: triết, quan điểm, học thuyết

#### Step 2.2: Hybrid Search

**Dense Search:**
```python
# Semantic similarity using embeddings
dense_results = vector_store._dense_search(query, k=10)
```

**Sparse Search (BM25):**
```python
# Keyword-based matching
sparse_results = vector_store._sparse_search(query, k=10)
```

**Reciprocal Rank Fusion (RRF):**
```python
# Combine results
for doc in all_docs:
    rrf_score = 1/(k + dense_rank) + 1/(k + sparse_rank)
```

**Benefits:**
- Dense: Handles synonyms, paraphrasing
- Sparse: Exact keyword matching
- RRF: Best of both worlds

#### Step 2.3: Re-ranking

- Sort by RRF score
- Filter by score threshold
- Return top-K documents

---

### Phase 3: Generation

#### Step 3.1: Context Preparation

```python
context = """
[Tài liệu 1 - Độ tin cậy: 0.892]
Loại: mathematics
Cấu trúc Toán: theorem, equation
Nội dung: Định lý Pythagore phát biểu rằng...

[Tài liệu 2 - Độ tin cậy: 0.845]
...
"""
```

#### Step 3.2: Prompt Selection

**Domain-Specific Prompts:**
- Mathematics: Emphasize formulas, proofs
- Philosophy: Focus on concepts, arguments
- General: Balanced approach

```python
from app.config.prompts import PROMPT_MAP

prompt = PROMPT_MAP["mathematics"]["answer"]
formatted = prompt.format(context=context, question=query)
```

#### Step 3.3: LLM Generation

```python
from app.config.llm_config import llm_config

llm = llm_config.get_llm_client()
answer = llm(formatted_prompt)
```

**Supported LLMs:**
- **Gemini Pro**: Fast, accurate, free tier
- **Qwen**: Chinese/Vietnamese optimized
- **Ollama**: Local, private
- **Groq**: Ultra-fast inference

---

## 🎯 Features

### 1. Advanced Chunking

**Context-Aware:**
- Detects document type automatically
- Preserves mathematical structures
- Maintains philosophical arguments

**Techniques:**
- Rule-based splitting (theorems, proofs)
- Semantic clustering (TF-IDF)
- Recursive splitting with overlap

### 2. Hybrid Search

**Dense + Sparse:**
- Semantic understanding (embeddings)
- Keyword matching (BM25)
- Optimal combination (RRF)

**Search Modes:**
```python
# Hybrid (recommended)
result = retriever.retrieve(query, search_mode="hybrid")

# Dense only (semantic)
result = retriever.retrieve(query, search_mode="dense")

# Sparse only (keyword)
result = retriever.retrieve(query, search_mode="sparse")
```

### 3. Multi-LLM Support

**Automatic Fallback:**
1. Try Gemini (if API key available)
2. Try Qwen (if API key available)
3. Fall back to Ollama (local)

**Configuration:**
```python
from app.config.llm_config import LLMConfig

config = LLMConfig()
config.provider  # Auto-detected: GEMINI, QWEN, or OLLAMA
```

### 4. Evaluation Metrics

**Retrieval Metrics:**
- Precision@K, Recall@K
- F1 Score
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Mean Average Precision (MAP)

**Generation Metrics:**
- Answer Relevance
- Faithfulness (context adherence)
- Answer Length

**Performance Metrics:**
- Latency (mean, median, P95, P99)
- Throughput (queries/second)

```python
from app.service.RAG.rag_metrics import create_metrics

metrics = create_metrics()

# Evaluate retrieval
eval_results = metrics.evaluate_retrieval(
    retrieved_docs_list,
    relevant_docs_list,
    k_values=[1, 3, 5, 10]
)

# Evaluate generation
gen_results = metrics.evaluate_generation(
    answers, queries, contexts
)
```

### 5. Pipeline Orchestration

**Complete Workflow:**
```python
from app.service.RAG.rag_pipeline import create_pipeline

# Initialize
pipeline = create_pipeline()

# Ingest documents
stats = pipeline.ingest_documents([
    "data/doc_1.pdf",
    "data/doc_2.txt"
])

# Query
result = pipeline.query("Your question here", k=5)

# Batch processing
results = pipeline.batch_query([
    "Question 1",
    "Question 2",
    "Question 3"
])

# Health check
health = pipeline.health_check()
```

---

## 🔧 Configuration

### Qdrant Settings

```python
pipeline = create_pipeline(
    qdrant_url="http://localhost:6333",  # Local
    # qdrant_url="https://xyz.cloud.qdrant.io",  # Cloud
    collection_name="my_collection"
)
```

### Chunking Parameters

```python
chunks = chunker.process_pdf(
    pdf_path="document.pdf",
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200     # Overlap between chunks
)
```

### Retrieval Parameters

```python
result = retriever.retrieve(
    query="Your question",
    k=5,                      # Number of documents
    search_mode="hybrid",     # hybrid/dense/sparse
    include_sources=True      # Include source docs
)
```

### Search Weights

```python
results = vector_store.hybrid_search(
    query="Your question",
    top_k=5,
    dense_weight=0.7,    # Weight for semantic search
    sparse_weight=0.3    # Weight for keyword search
)
```

---

## 📊 Performance

### Benchmarks (Example)

**Ingestion:**
- 100 pages PDF: ~30-60s
- Chunking: ~10-20s
- Embedding: ~20-40s
- Storage: ~5-10s

**Retrieval:**
- Query processing: ~50-200ms
- Dense search: ~20-50ms
- Sparse search: ~10-30ms
- Hybrid fusion: ~10-20ms

**Generation:**
- Gemini: ~1-3s
- Qwen: ~2-4s
- Ollama (local): ~5-15s

**Total Query Time:**
- Hybrid + Gemini: ~1.5-3.5s
- Dense + Groq: ~0.5-1.5s (fastest)

---

## 🐛 Troubleshooting

### Qdrant Connection Failed

```bash
# Check if Qdrant is running
curl http://localhost:6333/health

# Restart Qdrant
docker restart <qdrant_container>
```

### No Documents Retrieved

**Possible causes:**
1. Collection empty → Run ingestion first
2. Query too specific → Try broader terms
3. Threshold too high → Lower score_threshold

### LLM Not Available

**Check:**
```python
from app.config.llm_config import llm_config

print(llm_config.provider)  # Should show GEMINI/QWEN/OLLAMA
print(llm_config.config)    # Check API key
```

### Slow Performance

**Optimizations:**
1. Reduce chunk_size (faster embedding)
2. Use dense-only search (skip BM25)
3. Lower k value (fewer documents)
4. Use Groq for generation (fastest)

---

## 📚 API Reference

### RAGPipeline

```python
class RAGPipeline:
    def ingest_documents(paths: List[str]) -> Dict
    def load_existing_store() -> bool
    def initialize_retriever(search_type: str) -> HybridRetriever
    def query(query: str, k: int, search_mode: str) -> Dict
    def batch_query(queries: List[str], k: int) -> List[Dict]
    def get_stats() -> Dict
    def health_check() -> Dict
```

### HybridRetriever

```python
class HybridRetriever:
    def retrieve(query: str, k: int, search_mode: str) -> Dict
    def hybrid_search(query: str, k: int) -> List[Tuple[Document, float]]
    def dense_search(query: str, k: int) -> List[Tuple[Document, float]]
    def sparse_search(query: str, k: int) -> List[Tuple[Document, float]]
```

### RAGMetrics

```python
class RAGMetrics:
    def precision_at_k(retrieved, relevant, k) -> float
    def recall_at_k(retrieved, relevant, k) -> float
    def mean_reciprocal_rank(retrieved_list, relevant_list) -> float
    def ndcg_at_k(retrieved, relevant, k) -> float
    def evaluate_retrieval(retrieved_list, relevant_list) -> Dict
    def evaluate_generation(answers, queries, contexts) -> Dict
```

---

## 🎓 Best Practices

### 1. Document Preparation

- **Clean PDFs**: Remove headers/footers
- **OCR if needed**: For scanned documents
- **Consistent formatting**: Better chunking

### 2. Chunking Strategy

- **Math documents**: chunk_size=800-1200
- **Philosophy**: chunk_size=1000-1500
- **General**: chunk_size=500-1000
- **Overlap**: 15-20% of chunk_size

### 3. Search Strategy

- **Precise queries**: Use dense search
- **Keyword queries**: Use sparse search
- **General queries**: Use hybrid search

### 4. LLM Selection

- **Speed priority**: Groq
- **Quality priority**: Gemini
- **Privacy priority**: Ollama (local)
- **Vietnamese**: Qwen

### 5. Evaluation

- Always evaluate on test set
- Track metrics over time
- A/B test different configurations

---

## 🔮 Future Enhancements

### Planned Features

1. **Multi-modal RAG**: Images, tables, charts
2. **Streaming responses**: Real-time generation
3. **Query expansion**: Automatic query refinement
4. **Re-ranking models**: Cross-encoder re-ranking
5. **Caching**: Redis cache for frequent queries
6. **Multi-language**: Better Vietnamese support
7. **Graph RAG**: Knowledge graph integration

---

## 📞 Support

### Resources

- **Documentation**: This guide
- **Examples**: `app/service/RAG/main.py`
- **Logs**: `logs/app.log`, `logs/error.log`

### Common Issues

See **Troubleshooting** section above.

---

## 📄 License

This RAG system is part of the FastAPI project.

---

**Happy RAG-ing! 🚀**
