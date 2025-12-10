"""
RAG Evaluation Metrics
Provides comprehensive metrics for evaluating RAG system performance
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from app.utils.logger import setup_logging

logger = setup_logging("rag_metrics")


class RAGMetrics:
    """Metrics for evaluating RAG system performance"""
    
    def __init__(self):
        self.logger = logger
    
    # ============= RETRIEVAL METRICS =============
    
    def precision_at_k(self, 
                      retrieved_docs: List[str],
                      relevant_docs: List[str],
                      k: int = 5) -> float:
        """
        Precision@K: Proportion of retrieved documents that are relevant
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs (ground truth)
            k: Number of top documents to consider
        
        Returns:
            Precision score (0-1)
        """
        if not retrieved_docs or k == 0:
            return 0.0
        
        retrieved_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        
        relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant_set)
        
        return relevant_retrieved / k
    
    def recall_at_k(self,
                   retrieved_docs: List[str],
                   relevant_docs: List[str],
                   k: int = 5) -> float:
        """
        Recall@K: Proportion of relevant documents that are retrieved
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs (ground truth)
            k: Number of top documents to consider
        
        Returns:
            Recall score (0-1)
        """
        if not relevant_docs:
            return 0.0
        
        retrieved_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        
        relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant_set)
        
        return relevant_retrieved / len(relevant_docs)
    
    def f1_score(self,
                precision: float,
                recall: float) -> float:
        """
        F1 Score: Harmonic mean of precision and recall
        
        Args:
            precision: Precision score
            recall: Recall score
        
        Returns:
            F1 score (0-1)
        """
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def mean_reciprocal_rank(self,
                            retrieved_docs_list: List[List[str]],
                            relevant_docs_list: List[List[str]]) -> float:
        """
        MRR: Average of reciprocal ranks of first relevant document
        
        Args:
            retrieved_docs_list: List of retrieved document lists for each query
            relevant_docs_list: List of relevant document lists for each query
        
        Returns:
            MRR score (0-1)
        """
        reciprocal_ranks = []
        
        for retrieved, relevant in zip(retrieved_docs_list, relevant_docs_list):
            relevant_set = set(relevant)
            
            for rank, doc in enumerate(retrieved, 1):
                if doc in relevant_set:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def ndcg_at_k(self,
                 retrieved_docs: List[str],
                 relevant_docs: Dict[str, float],
                 k: int = 5) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: Dict mapping doc IDs to relevance scores (0-1)
            k: Number of top documents to consider
        
        Returns:
            NDCG score (0-1)
        """
        if not retrieved_docs or k == 0:
            return 0.0
        
        retrieved_k = retrieved_docs[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_k, 1):
            relevance = relevant_docs.get(doc, 0.0)
            dcg += relevance / np.log2(i + 1)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevances = sorted(relevant_docs.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def map_score(self,
                 retrieved_docs_list: List[List[str]],
                 relevant_docs_list: List[List[str]]) -> float:
        """
        MAP: Mean Average Precision across all queries
        
        Args:
            retrieved_docs_list: List of retrieved document lists
            relevant_docs_list: List of relevant document lists
        
        Returns:
            MAP score (0-1)
        """
        average_precisions = []
        
        for retrieved, relevant in zip(retrieved_docs_list, relevant_docs_list):
            if not relevant:
                continue
            
            relevant_set = set(relevant)
            precisions = []
            num_relevant = 0
            
            for i, doc in enumerate(retrieved, 1):
                if doc in relevant_set:
                    num_relevant += 1
                    precision = num_relevant / i
                    precisions.append(precision)
            
            if precisions:
                average_precisions.append(np.mean(precisions))
            else:
                average_precisions.append(0.0)
        
        return np.mean(average_precisions) if average_precisions else 0.0
    
    # ============= GENERATION METRICS =============
    
    def answer_relevance(self,
                        answer: str,
                        query: str,
                        context: str) -> Dict[str, float]:
        """
        Evaluate answer relevance (simple heuristics)
        
        Args:
            answer: Generated answer
            query: Original query
            context: Retrieved context
        
        Returns:
            Dict with relevance scores
        """
        scores = {}
        
        # Length check
        scores["length_score"] = min(len(answer.split()) / 50, 1.0)
        
        # Query term overlap
        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())
        overlap = len(query_terms & answer_terms)
        scores["query_overlap"] = overlap / len(query_terms) if query_terms else 0.0
        
        # Context usage
        context_terms = set(context.lower().split())
        context_usage = len(answer_terms & context_terms)
        scores["context_usage"] = min(context_usage / len(answer_terms), 1.0) if answer_terms else 0.0
        
        # Overall score
        scores["overall"] = np.mean([
            scores["length_score"],
            scores["query_overlap"],
            scores["context_usage"]
        ])
        
        return scores
    
    def faithfulness_score(self,
                          answer: str,
                          context: str) -> float:
        """
        Simple faithfulness check: How much of answer comes from context
        
        Args:
            answer: Generated answer
            context: Retrieved context
        
        Returns:
            Faithfulness score (0-1)
        """
        answer_sentences = answer.split('.')
        context_lower = context.lower()
        
        faithful_sentences = 0
        for sentence in answer_sentences:
            sentence = sentence.strip().lower()
            if not sentence:
                continue
            
            # Check if key terms from sentence appear in context
            terms = sentence.split()
            if len(terms) < 3:
                continue
            
            # Check if at least 50% of terms appear in context
            term_matches = sum(1 for term in terms if term in context_lower)
            if term_matches / len(terms) >= 0.5:
                faithful_sentences += 1
        
        total_sentences = len([s for s in answer_sentences if s.strip()])
        return faithful_sentences / total_sentences if total_sentences > 0 else 0.0
    
    # ============= PERFORMANCE METRICS =============
    
    def latency_metrics(self,
                       query_times: List[float]) -> Dict[str, float]:
        """
        Calculate latency statistics
        
        Args:
            query_times: List of query execution times in seconds
        
        Returns:
            Dict with latency metrics
        """
        if not query_times:
            return {}
        
        return {
            "mean": np.mean(query_times),
            "median": np.median(query_times),
            "p95": np.percentile(query_times, 95),
            "p99": np.percentile(query_times, 99),
            "min": np.min(query_times),
            "max": np.max(query_times),
            "std": np.std(query_times)
        }
    
    def throughput(self,
                  num_queries: int,
                  total_time: float) -> float:
        """
        Calculate queries per second
        
        Args:
            num_queries: Number of queries processed
            total_time: Total time in seconds
        
        Returns:
            Queries per second
        """
        return num_queries / total_time if total_time > 0 else 0.0
    
    # ============= COMPREHENSIVE EVALUATION =============
    
    def evaluate_retrieval(self,
                          retrieved_docs_list: List[List[str]],
                          relevant_docs_list: List[List[str]],
                          k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Comprehensive retrieval evaluation
        
        Args:
            retrieved_docs_list: List of retrieved document lists
            relevant_docs_list: List of relevant document lists
            k_values: List of k values to evaluate
        
        Returns:
            Dict with all retrieval metrics
        """
        self.logger.info("Evaluating retrieval performance")
        
        results = {
            "num_queries": len(retrieved_docs_list),
            "metrics": {}
        }
        
        # Precision and Recall at different K
        for k in k_values:
            precisions = []
            recalls = []
            
            for retrieved, relevant in zip(retrieved_docs_list, relevant_docs_list):
                p = self.precision_at_k(retrieved, relevant, k)
                r = self.recall_at_k(retrieved, relevant, k)
                precisions.append(p)
                recalls.append(r)
            
            results["metrics"][f"precision@{k}"] = np.mean(precisions)
            results["metrics"][f"recall@{k}"] = np.mean(recalls)
            
            # F1 score
            avg_p = np.mean(precisions)
            avg_r = np.mean(recalls)
            results["metrics"][f"f1@{k}"] = self.f1_score(avg_p, avg_r)
        
        # MRR
        results["metrics"]["mrr"] = self.mean_reciprocal_rank(
            retrieved_docs_list, relevant_docs_list
        )
        
        # MAP
        results["metrics"]["map"] = self.map_score(
            retrieved_docs_list, relevant_docs_list
        )
        
        self.logger.info(f"Retrieval evaluation completed: {results['metrics']}")
        return results
    
    def evaluate_generation(self,
                           answers: List[str],
                           queries: List[str],
                           contexts: List[str]) -> Dict[str, Any]:
        """
        Comprehensive generation evaluation
        
        Args:
            answers: List of generated answers
            queries: List of queries
            contexts: List of contexts
        
        Returns:
            Dict with generation metrics
        """
        self.logger.info("Evaluating generation quality")
        
        relevance_scores = []
        faithfulness_scores = []
        
        for answer, query, context in zip(answers, queries, contexts):
            rel = self.answer_relevance(answer, query, context)
            relevance_scores.append(rel["overall"])
            
            faith = self.faithfulness_score(answer, context)
            faithfulness_scores.append(faith)
        
        results = {
            "num_answers": len(answers),
            "metrics": {
                "avg_relevance": np.mean(relevance_scores),
                "avg_faithfulness": np.mean(faithfulness_scores),
                "avg_answer_length": np.mean([len(a.split()) for a in answers])
            }
        }
        
        self.logger.info(f"Generation evaluation completed: {results['metrics']}")
        return results


# Convenience function
def create_metrics() -> RAGMetrics:
    """Create a RAG metrics instance"""
    return RAGMetrics()
