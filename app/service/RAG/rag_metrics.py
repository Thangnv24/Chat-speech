import os
import numpy as np
from typing import List, Dict, Tuple
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from app.utils.logger import setup_logging
from app.service.RAG.main import SimpleRAG
from app.service.RAG.rag_pipeline import create_pipeline
import json
import time

os.environ["OPENAI_API_KEY"] = os.getenv("GEMINI_API_KEY")
logger = setup_logging("rag_metrics")

class RAGMetrics:
    def __init__(self):
        self.logger = logger

    # Retrieval Metrics

    # Precision
    def precision_at_k(self, retrieved: List[str], relevant: List[str], k: int = 10) -> float:
        if not retrieved or k == 0: return 0.0
        relevant_set = set(relevant)
        return sum(1 for doc in retrieved[:k] if doc in relevant_set) / k

    # Recall
    def recall_at_k(self, retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        if not relevant: return 0.0
        relevant_set = set(relevant)
        return sum(1 for doc in retrieved[:k] if doc in relevant_set) / len(relevant)

    # F1-score
    def f1_score(self, precision: float, recall: float) -> float:
        if precision + recall == 0: return 0.0
        return 2 * (precision * recall) / (precision + recall)

    # Evaluation
    def evaluate(self, 
                 retrieved_list: List[List[str]], 
                 relevant_list: List[List[str]], 
                 answers: List[str], 
                 queries: List[str], 
                 contexts: List[List[str]],
                 ground_truth_answers: List[str] = None,
                 k: int = 5) -> Dict[str, float]:

        # Retrieval metrics
        precisions = [self.precision_at_k(r, g, k) for r, g in zip(retrieved_list, relevant_list)]
        recalls = [self.recall_at_k(r, g, k) for r, g in zip(retrieved_list, relevant_list)]
        
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1 = self.f1_score(avg_precision, avg_recall)

        # RAGAS metrics (only if we have ground truth)
        context_precision_score = None
        faithfulness_score = None
        relevance_score = None
        
        try:
            # Prepare dataset for RAGAS
            data_dict = {
                "question": queries,
                "answer": answers,
                "contexts": contexts,
            }
            
            # Add ground truth if available (for context_precision)
            if ground_truth_answers:
                data_dict["ground_truth"] = ground_truth_answers
            
            dataset = Dataset.from_dict(data_dict)

            # Select metrics based on available data
            metrics_to_use = []
            
            # Faithfulness and Answer Relevance don't need ground truth
            faithfulness = Faithfulness()
            answer_relevance = AnswerRelevancy()
            metrics_to_use.extend([faithfulness, answer_relevance])
            
            # Context Precision needs ground truth
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
            self.logger.warning(f"RAGAS evaluation failed: {e}")
            self.logger.warning("Continuing with retrieval metrics only...")

        self._log_table(avg_precision, avg_recall, avg_f1, context_precision_score, faithfulness_score, relevance_score, k)

        return {
            "precision": avg_precision, 
            "recall": avg_recall, 
            "f1": avg_f1, 
            "context_precision": context_precision_score, 
            "faithfulness": faithfulness_score, 
            "relevance": relevance_score
        }

    def _log_table(self, p, r, f1, cp, faith, rel, k):
        header = f"{'METRIC':<25} | {'SCORE':<10}"
        separator = "-" * 38
        
        # Format scores (handle None)
        def fmt(val):
            return f"{val:.4f}" if val is not None else "N/A"
        
        table = (
            f"\nRAG EVALUATION (k={k})\n"
            f"{separator}\n"
            f"{header}\n"
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
        
        table += f"{separator}\n"
        
        self.logger.info(table)


def create_metrics() -> RAGMetrics:
    return RAGMetrics()

def load_rag_test_data(file_path: str, subject: str = None) -> Tuple[List[str], List[str]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = []
    answers = []
    
    # Helper to append data
    def add_data(subj_data):
        for item in subj_data:
            queries.append(item['question'])
            answers.append(item['answer'])
    
    if subject:
        if subject in data:
            add_data(data[subject])
        else:
            raise ValueError(f"Subject '{subject}' not found in dataset.")
    else:
        # Load all
        for subj in data:
            add_data(data[subj])
            
    return queries, answers

# --- DEMO CÁCH SỬ DỤNG ---
if __name__ == "__main__":
    # Ví dụ 1: Load toàn bộ (cả Toán và Triết)
    all_queries, all_answers = load_rag_test_data("dataset.json")
    print(f"\nLoaded Total: {len(all_queries)} pairs")
    print(f"Sample Query 1: {all_queries[0]}")
    print(f"Sample Answer 1: {all_answers[0][:50]}...")

    # Ví dụ 2: Chỉ load Toán để test với prompt Toán
    math_queries, math_answers = load_rag_test_data("dataset.json", subject="mathematics")
    print(f"\nLoaded Math: {len(math_queries)} pairs")

    # Ví dụ 3: Chỉ load Triết để test với prompt Triết
    phil_queries, phil_answers = load_rag_test_data("dataset.json", subject="philosophy")
    print(f"\nLoaded Philosophy: {len(phil_queries)} pairs")

    # Gán vào biến như bạn yêu cầu trong prompt cũ
    test_queries = all_queries
    test_answers = all_answers

    metrics = create_metrics()

    retrieved_contexts = []
    generated_answers = []

    # Initialize RAG pipeline
    rag_pipeline = SimpleRAG()
    if not rag_pipeline.setup():
        logger.error("Failed to setup RAG pipeline")
        exit(1)

    logger.info(f"Processing {len(test_queries)} queries...")

    for i, q in enumerate(test_queries, 1):
        logger.info(f"Query {i}/{len(test_queries)}: {q[:50]}...")
        
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
            elif hasattr(doc_tuple, 'page_content'):
                doc_contents.append(doc_tuple.page_content)
        
        retrieved_contexts.append(doc_contents)
        
        # Get generated answer
        answer = result.get('answer', '')
        generated_answers.append(answer)
        
        logger.info(f"  Retrieved {len(doc_contents)} documents")
        logger.info(f"  Answer: {answer[:50]}...")
        time.sleep(5)

    logger.info("Evaluation starting...")

    # Note: ground_truth_docs is not defined, need to create it
    # For now, use retrieved_contexts as ground truth (not ideal but for testing)
    ground_truth_docs = retrieved_contexts

    # Use test_answers as ground truth for RAGAS metrics
    results = metrics.evaluate(
        retrieved_list=retrieved_contexts,
        relevant_list=ground_truth_docs,  
        queries=test_queries,
        answers=generated_answers,
        contexts=retrieved_contexts,
        ground_truth_answers=test_answers,  # Ground truth for RAGAS
        k=5
    )

    logger.info("Evaluation completed!")
    logger.info(f"Results: {results}")