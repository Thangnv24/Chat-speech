# 🔧 RAG Metrics Fix - AttributeError

## ❌ Lỗi gặp phải:

### Lỗi 1:
```python
AttributeError: 'dict' object has no attribute 'page_content'
```

### Lỗi 2:
```python
AttributeError: 'str' object has no attribute 'get'
```

## 🔍 Nguyên nhân:

### Vấn đề 1: 
`rag_pipeline.query()` trong `SimpleRAG` chỉ trả về **string** (answer), không phải dict đầy đủ.

```python
# SimpleRAG.query() - CHỈ TRẢ VỀ STRING
def query(self, question, k=5, search_mode="hybrid"):
    result = self.pipeline.query(...)
    answer = result.get('answer', 'There is no answer')
    return answer  # ❌ Chỉ trả về string
```

### Vấn đề 2:
Cần full dict để lấy `retrieved_documents` cho metrics evaluation.

## ✅ Fix:

### 1. Thêm method `query_full()` vào `SimpleRAG`:

```python
# app/service/RAG/main.py

def query_full(self, question, k=5, search_mode="hybrid"):
    """Query and return full result dict (for metrics evaluation)"""
    if not self.is_ready:
        return {"answer": "Run setup() first", "error": True}
    
    result = self.pipeline.query(
        query=question,
        k=k,
        search_mode=search_mode,
        include_sources=True
    )
    
    return result  # ✅ Trả về full dict
```

### 2. Dùng `query_full()` trong metrics:

```python
# app/service/RAG/rag_metrics.py

# Query RAG - use query_full() to get full result dict
result = rag_pipeline.query_full(q, k=5, search_mode='hybrid')

# Extract retrieved documents from result
retrieved_docs = result.get('retrieved_documents', [])

# Get page_content from each document tuple (doc, score)
doc_contents = []
for doc_tuple in retrieved_docs:
    if isinstance(doc_tuple, tuple) and len(doc_tuple) >= 1:
        doc = doc_tuple[0]  # First element is the Document
        if hasattr(doc, 'page_content'):
            doc_contents.append(doc.page_content)

retrieved_contexts.append(doc_contents)

# Get generated answer
answer = result.get('answer', '')
generated_answers.append(answer)
```

## 🧪 Test trước khi chạy metrics:

```bash
# Test để hiểu output format
python test_rag_metrics.py
```

Output sẽ show:
- Cấu trúc của result dict
- Cách extract documents
- Format đúng cho metrics

## 🚀 Chạy metrics:

```bash
# Sau khi test OK
python app/service/RAG/rag_metrics.py
```

## 📊 Cấu trúc output:

### `query()` - Trả về string:
```python
answer = rag.query("question")
# Returns: "Định lý Pytago phát biểu rằng..."
```

### `query_full()` - Trả về dict:
```python
result = rag.query_full("question")
# Returns:
{
    'query': 'Định lý Pytago là gì?',
    'answer': 'Định lý Pytago phát biểu rằng...',
    'retrieved_documents': [
        (Document(page_content='...', metadata={...}), 0.85),
        (Document(page_content='...', metadata={...}), 0.72),
        ...
    ],
    'context': 'formatted context string',
    'query_time': 1.23,
    'num_retrieved': 5
}
```

## 🔑 Key Points:

1. **`SimpleRAG.query()`** trả về string (answer only)
2. **`SimpleRAG.query_full()`** trả về dict (full result)
3. **Metrics cần full dict** để lấy retrieved_documents
4. **`retrieved_documents`** là list of tuples: `(Document, score)`
5. **Cần extract** `page_content` từ Document object

## 📝 Checklist:

- [x] Thêm `query_full()` method vào `SimpleRAG`
- [x] Update `rag_metrics.py` để dùng `query_full()`
- [x] Update `test_rag_metrics.py`
- [ ] Chạy `test_rag_metrics.py` thành công
- [ ] Tạo file `dataset.json`
- [ ] Data đã được ingest vào Qdrant
- [ ] Chạy `rag_metrics.py` thành công

---

**Fix completed! 🎉**
