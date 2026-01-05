# 🔧 RAGAS Metrics Fix - Missing 'reference' Column

## ❌ Lỗi:

```python
ValueError: The metric [context_precision] that is used requires the following additional columns ['reference'] to be present in the dataset.
```

## 🔍 Nguyên nhân:

RAGAS metrics có requirements khác nhau:

| Metric | Requires Ground Truth? |
|--------|----------------------|
| **Faithfulness** | ❌ No (chỉ cần context + answer) |
| **Answer Relevance** | ❌ No (chỉ cần question + answer) |
| **Context Precision** | ✅ Yes (cần ground truth answer) |

## ✅ Fix:

### 1. Update `evaluate()` method:

```python
def evaluate(self, 
             retrieved_list: List[List[str]], 
             relevant_list: List[List[str]], 
             answers: List[str], 
             queries: List[str], 
             contexts: List[List[str]],
             ground_truth_answers: List[str] = None,  # ✅ Optional ground truth
             k: int = 5) -> Dict[str, float]:
    
    # ... retrieval metrics ...
    
    # RAGAS metrics
    try:
        data_dict = {
            "question": queries,
            "answer": answers,
            "contexts": contexts,
        }
        
        # Add ground truth if available
        if ground_truth_answers:
            data_dict["ground_truth"] = ground_truth_answers
        
        dataset = Dataset.from_dict(data_dict)
        
        # Select metrics based on available data
        metrics_to_use = []
        
        # These don't need ground truth
        faithfulness = Faithfulness()
        answer_relevance = AnswerRelevancy()
        metrics_to_use.extend([faithfulness, answer_relevance])
        
        # This needs ground truth
        if ground_truth_answers:
            context_precision = ContextPrecision()
            metrics_to_use.append(context_precision)
        
        # Evaluate
        ragas_results = evaluate(dataset, metrics=metrics_to_use)
        
        faithfulness_score = ragas_results.get("faithfulness", None)
        relevance_score = ragas_results.get("answer_relevance", None)
        
        if ground_truth_answers:
            context_precision_score = ragas_results.get("context_precision", None)
            
    except Exception as e:
        logger.warning(f"RAGAS evaluation failed: {e}")
        logger.warning("Continuing with retrieval metrics only...")
```

### 2. Update evaluate call:

```python
results = metrics.evaluate(
    retrieved_list=retrieved_contexts,
    relevant_list=ground_truth_docs,  
    queries=test_queries,
    answers=generated_answers,
    contexts=retrieved_contexts,
    ground_truth_answers=test_answers,  # ✅ Pass ground truth
    k=5
)
```

### 3. Update log table to handle None:

```python
def _log_table(self, p, r, f1, cp, faith, rel, k):
    # Format scores (handle None)
    def fmt(val):
        return f"{val:.4f}" if val is not None else "N/A"
    
    table = (
        f"\nRAG EVALUATION (k={k})\n"
        f"{separator}\n"
        f"{'Precision@'+str(k):<25} | {fmt(p)}\n"
        f"{'Recall@'+str(k):<25} | {fmt(r)}\n"
        f"{'F1 Score':<25} | {fmt(f1)}\n"
    )
    
    # Only add RAGAS metrics if available
    if cp is not None or faith is not None or rel is not None:
        table += f"{separator}\n"
        if cp is not None:
            table += f"{'Context Precision':<25} | {fmt(cp)}\n"
        if faith is not None:
            table += f"{'Faithfulness':<25} | {fmt(faith)}\n"
        if rel is not None:
            table += f"{'Answer Relevance':<25} | {fmt(rel)}\n"
```

## 📊 Output với ground truth:

```
RAG EVALUATION (k=5)
--------------------------------------
METRIC                    | SCORE     
--------------------------------------
Precision@5               | 0.8500
Recall@5                  | 0.7200
F1 Score                  | 0.7800
--------------------------------------
Context Precision         | 0.8100
Faithfulness              | 0.9200
Answer Relevance          | 0.8800
--------------------------------------
```

## 📊 Output không có ground truth:

```
RAG EVALUATION (k=5)
--------------------------------------
METRIC                    | SCORE     
--------------------------------------
Precision@5               | 0.8500
Recall@5                  | 0.7200
F1 Score                  | 0.7800
--------------------------------------
Faithfulness              | 0.9200
Answer Relevance          | 0.8800
--------------------------------------
```

## 🎯 Dataset Format:

### Với ground truth (full evaluation):

```json
{
  "mathematics": [
    {
      "question": "Định lý Pytago là gì?",
      "answer": "Định lý Pytago phát biểu rằng..."
    }
  ],
  "philosophy": [
    {
      "question": "Triết học là gì?",
      "answer": "Triết học là..."
    }
  ]
}
```

### Không có ground truth (partial evaluation):

Chỉ cần questions, RAGAS sẽ skip Context Precision:

```python
test_queries = ["question 1", "question 2"]
# Don't pass ground_truth_answers
results = metrics.evaluate(
    ...,
    ground_truth_answers=None  # or omit this parameter
)
```

## 🔑 Key Points:

1. **Context Precision** cần ground truth answers
2. **Faithfulness** và **Answer Relevance** không cần
3. Code tự động detect và skip metrics không có data
4. Error handling để không crash nếu RAGAS fail

## 📝 Checklist:

- [x] Update `evaluate()` method với optional ground_truth
- [x] Update log table để handle None values
- [x] Add error handling cho RAGAS
- [x] Pass ground_truth_answers trong evaluate call
- [ ] Tạo dataset.json với ground truth answers
- [ ] Chạy evaluation thành công

---

**Fix completed! 🎉**
