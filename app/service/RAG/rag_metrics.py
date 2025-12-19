import os
import numpy as np
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from app.utils.logger import setup_logging
from app.service.RAG.main import SimpleRAG

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
                 k: int = 5) -> Dict[str, float]:

        data_dict = {
            "question": queries,
            "answer": answers,
            "contexts": contexts,
        }
        dataset = Dataset.from_dict(data_dict)

        precisions = [self.precision_at_k(r, g, k) for r, g in zip(retrieved_list, relevant_list)]
        recalls = [self.recall_at_k(r, g, k) for r, g in zip(retrieved_list, relevant_list)]
        
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1 = self.f1_score(avg_precision, avg_recall)

        #######################
        ragas_results = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevance, context_precision]
        )
        
        context_precision = ragas_results["context_precision"]
        faithfulness = ragas_results["faithfulness"]
        relevance = ragas_results["answer_relevance"]

        self._log_table(avg_precision, avg_recall, avg_f1, context_precision, faithfulness, relevance, k)

        return {
            "precision": avg_precision, "recall": avg_recall, "f1": avg_f1, 
            "context_precision": context_precision, "faithfulness": faithfulness, "relevance": relevance
        }

    def _log_table(self, p, r, f1, cp, faith, rel, k):
        header = f"{'METRIC':<25} | {'SCORE':<10}"
        separator = "-" * 38
        
        table = (
            f"\nRAG EVALUATION (k={k})\n"
            f"{separator}\n"
            f"{header}\n"
            f"{separator}\n"
            f"{'Precision@'+str(k):<25} | {p:.4f}\n"
            f"{'Recall@'+str(k):<25} | {r:.4f}\n"
            f"{'F1 Score':<25} | {f1:.4f}\n"
            f"{'Context Precision':<25} | {cp:.4f}\n"
            f"{separator}\n"
            f"{'Faithfulness':<25} | {faith:.4f}\n"
            f"{'Answer Relevance':<25} | {rel:.4f}\n"
            f"{separator}\n"
        )
        self.logger.info(table)


def create_metrics() -> RAGMetrics:
    return RAGMetrics()

test_queries = ["Định lý Pytago là gì?", "Triết học Mác Lê nin là gì?"]
test_answers = ["Pytago là định lý đại số tuyến tính của Ba Can", "Triết học Mác Lê Nin là một triết học tư tưởng của Mác Lê Nin"]

metrics = create_metrics()

retrieved_contexts = []
generated_answers = []
rag_pipeline = SimpleRAG()
for q in test_queries:
    docs = rag_pipeline.retriever.retrieve(q) 
    retrieved_contexts.append([d.page_content for d in docs])
    
    ans = rag_pipeline.query(q)
    generated_answers.append(ans)

results = metrics.evaluate(
    retrieved_list=retrieved_contexts,
    relevant_list=ground_truth_docs,  
    queries=test_queries,
    answers=generated_answers,
    contexts=retrieved_contexts
)