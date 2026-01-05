# 📊 RAG Metrics Evaluation Guide

## 📋 Tổng quan

Script `rag_metrics.py` đánh giá chất lượng RAG system với các metrics:

### Retrieval Metrics:
- **Precision@k**: Độ chính xác của documents retrieved
- **Recall@k**: Độ phủ của documents retrieved
- **F1 Score**: Harmonic mean của Precision và Recall

### Generation Metrics (RAGAS):
- **Context Precision**: Độ chính xác của context
- **Faithfulness**: Độ trung thực của answer với context
- **Answer Relevance**: Độ liên quan của answer với question

---

## 🔧 Setup

### 1. Cài dependencies

```bash
pip install ragas datasets
```

### 2. Chuẩn bị dataset

Tạo file `dataset.json` với format:

```json
{
  "mathematics": [
    {
      "question": "Định lý Pytago là gì?",
      "answer": "Định lý Pytago phát biểu rằng..."
    },
    {
      "question": "Tích phân là gì?",
      "answer": "Tích phân là..."
    }
  ],
  "philosophy": [
    {
      "question": "Triết học Mác Lê Nin là gì?",
      "answer": "Triết học Mác Lê Nin là..."
    }
  ]
}
```

### 3. Đảm bảo vector store đã có data

```bash
# Ingest data trước
python ingest_data.py

# Check collection
python view_qdrant.py
```

---

## 🚀 Cách chạy

### Chạy evaluation đầy đủ:

```bash
python app/service/RAG/rag_metrics.py
```

### Chạy với subject cụ thể:

Sửa trong code:

```python
# Chỉ test Toán
test_queries, test_answers = load_rag_test_data("dataset.json", subject="mathematics")

# Chỉ test Triết
test_queries, test_answers = load_rag_test_data("dataset.json", subject="philosophy")

# Test cả hai
test_queries, test_answers = load_rag_test_data("dataset.json")
```

---

## 📊 Output mẫu

```
RAG EVALUATION (k=5)
--------------------------------------
METRIC                    | SCORE     
--------------------------------------
Precision@5               | 0.8500
Recall@5                  | 0.7200
F1 Score                  | 0.7800
Context Precision         | 0.8100
--------------------------------------
Faithfulness              | 0.9200
Answer Relevance          | 0.8800
--------------------------------------
```

---

## 🔍 Hiểu các metrics

### Precision@k
- **Cao (>0.8)**: Hầu hết documents retrieved đều relevant
- **Thấp (<0.5)**: Nhiều documents không liên quan

### Recall@k
- **Cao (>0.8)**: Tìm được hầu hết relevant documents
- **Thấp (<0.5)**: Bỏ sót nhiều relevant documents

### F1 Score
- **Cao (>0.8)**: Cân bằng tốt giữa Precision và Recall
- **Thấp (<0.5)**: Cần cải thiện retrieval

### Context Precision
- **Cao (>0.8)**: Context được rank tốt
- **Thấp (<0.5)**: Relevant context bị rank thấp

### Faithfulness
- **Cao (>0.9)**: Answer trung thực với context
- **Thấp (<0.7)**: Answer có thể hallucinate

### Answer Relevance
- **Cao (>0.8)**: Answer trả lời đúng câu hỏi
- **Thấp (<0.6)**: Answer không liên quan

---

## 🐛 Troubleshooting

### Lỗi: "AttributeError: 'NoneType' object has no attribute 'retrieve'"

**Nguyên nhân**: Retriever chưa được initialize

**Đã fix**: Code đã được update để tự động initialize

### Lỗi: "Vector store not initialized"

**Giải pháp**:
```bash
python ingest_data.py
```

### Lỗi: "FileNotFoundError: dataset.json"

**Giải pháp**: Tạo file `dataset.json` với format như trên

### Lỗi: "OPENAI_API_KEY not found"

**Nguyên nhân**: RAGAS cần OpenAI API key (hoặc Gemini)

**Giải pháp**: Thêm vào `.env`:
```env
GEMINI_API_KEY=your_gemini_key
```

Code đã set: `os.environ["OPENAI_API_KEY"] = os.getenv("GEMINI_API_KEY")`

### Evaluation chậm

**Nguyên nhân**: RAGAS gọi LLM cho mỗi query

**Giải pháp**: 
- Giảm số queries test
- Dùng model nhanh hơn
- Cache results

---

## 🎯 Cải thiện metrics

### Nếu Precision thấp:
1. Cải thiện chunking strategy
2. Tăng chunk overlap
3. Thử embedding model khác
4. Tune search parameters

### Nếu Recall thấp:
1. Tăng `k` (số documents retrieve)
2. Dùng hybrid search
3. Cải thiện query preprocessing
4. Thêm nhiều data vào vector store

### Nếu Faithfulness thấp:
1. Cải thiện prompt template
2. Instruct LLM rõ ràng hơn
3. Thêm context vào prompt
4. Dùng model tốt hơn

### Nếu Answer Relevance thấp:
1. Cải thiện retrieval (Precision/Recall)
2. Tune prompt template
3. Thêm query expansion
4. Dùng reranking

---

## 📝 Custom Evaluation

### Tạo custom metrics:

```python
from app.service.RAG.rag_metrics import RAGMetrics

metrics = RAGMetrics()

# Your test data
test_queries = ["question 1", "question 2"]
retrieved_contexts = [[doc1, doc2], [doc3, doc4]]
generated_answers = ["answer 1", "answer 2"]
ground_truth_docs = [[doc1, doc2], [doc3, doc4]]

# Evaluate
results = metrics.evaluate(
    retrieved_list=retrieved_contexts,
    relevant_list=ground_truth_docs,
    queries=test_queries,
    answers=generated_answers,
    contexts=retrieved_contexts,
    k=5
)

print(results)
```

### Chỉ tính retrieval metrics:

```python
metrics = RAGMetrics()

precision = metrics.precision_at_k(
    retrieved=["doc1", "doc2", "doc3"],
    relevant=["doc1", "doc3"],
    k=3
)

recall = metrics.recall_at_k(
    retrieved=["doc1", "doc2", "doc3"],
    relevant=["doc1", "doc3", "doc4"],
    k=3
)

f1 = metrics.f1_score(precision, recall)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")
```

---

## 🔬 Advanced Usage

### Batch evaluation:

```python
import json

# Load large dataset
with open("large_dataset.json") as f:
    data = json.load(f)

# Process in batches
batch_size = 10
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    # Evaluate batch
    # Save results
```

### Compare different configurations:

```python
configs = [
    {"k": 3, "search_mode": "dense"},
    {"k": 5, "search_mode": "hybrid"},
    {"k": 10, "search_mode": "sparse"}
]

for config in configs:
    # Run evaluation with config
    # Compare results
```

### Export results:

```python
import json

results = metrics.evaluate(...)

with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## 📚 References

- RAGAS: https://docs.ragas.io/
- Retrieval Metrics: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)
- RAG Evaluation: https://arxiv.org/abs/2312.10997

---

**Happy Evaluating! 📊**
