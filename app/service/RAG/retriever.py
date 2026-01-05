import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from app.utils.logger import setup_logging
from app.config.prompts import PROMPT_MAP
from app.config.llm_config import llm_config
from sentence_transformers import CrossEncoder

logger = setup_logging("retriever")

class HybridRetriever:
    
    def __init__(self, 
                 vector_store,
                 search_type: str = "hybrid",
                 rerank_model: str = "BAAI/bge-reranker-v2-m3"):
        
        self.vector_store = vector_store
        self.search_type = search_type
        self.logger = logger

        self.llm_config = llm_config
        self.llm = self.llm_config.get_llm_client()
        
        self.prompts = PROMPT_MAP
        
        self.reranker = CrossEncoder(rerank_model)
    
    def _setup_prompts(self):
        pass
        
    
    def _detect_query_type(self, query: str) -> str:
        math_keywords = ["toán", "tính", "phương trình", "định lý", "chứng minh", "công thức"]
        philosophy_keywords = ["triết", "quan điểm", "học thuyết", "tư tưởng", "khái niệm", "luận"]
        
        query_lower = query.lower()
        
        math_score = sum(1 for keyword in math_keywords if keyword in query_lower)
        philosophy_score = sum(1 for keyword in philosophy_keywords if keyword in query_lower)
        
        if math_score > philosophy_score and math_score > 0:
            return "mathematics"
        elif philosophy_score > math_score and philosophy_score > 0:
            return "philosophy"
        else:
            return "general"
    
    def hybrid_search(self, 
                     query: str, 
                     k: int = 10,
                     dense_weight: float = 0.7,
                     sparse_weight: float = 0.3) -> List[Tuple[Document, float]]:
       
        self.logger.info(f"Performing hybrid search for: {query}")
        
        try:
            results = self.vector_store.hybrid_search(
                query=query,
                top_k=k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight
            )
            
            self.logger.info(f"Hybrid search returned {len(results)} documents")
            return results
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {str(e)}")
            try:
                results = self.vector_store._dense_search(query, k)
                self.logger.warning("Fallback to dense-only search")
                return results
            except Exception as e2:
                self.logger.error(f"Fallback search also failed: {str(e2)}")
                return []
    
    def dense_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        self.logger.info(f"Performing dense search for: {query}")
        
        try:
            results = self.vector_store._dense_search(query, k)
            self.logger.info(f"Dense search returned {len(results)} documents")
            return results
        except Exception as e:
            self.logger.error(f"Dense search failed: {str(e)}")
            return []
    
    def sparse_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        self.logger.info(f"Performing sparse search for: {query}")
        
        try:
            results = self.vector_store._sparse_search(query, k)
            self.logger.info(f"Sparse search returned {len(results)} documents")
            return results
        except Exception as e:
            self.logger.error(f"Sparse search failed: {str(e)}")
            return []
    
    def _rerank_documents(self, query: str, documents: List[Tuple[Document, float]], top_k: int = 5) -> List[Tuple[Document, float]]:
        if not documents:
            return []
        
        try:
            pairs = [[query, doc.page_content] for doc, _ in documents]
            scores = self.reranker.predict(pairs)
            
            reranked = [(documents[i][0], float(scores[i])) for i in range(len(documents))]
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            return reranked[:top_k]
        except Exception as e:
            self.logger.error(f"Reranking failed: {str(e)}")
            return documents[:top_k]
    
    def retrieve(self, 
                query: str, 
                k: int = 10,
                rerank: bool = True,
                rerank_top_k: int = 5,
                include_sources: bool = True,
                search_mode: str = "hybrid") -> Dict[str, Any]:
                
        self.logger.info(f"Retrieving documents for query: {query}")
        
        query_type = self._detect_query_type(query)
        self.logger.info(f"Detected query type: {query_type}")
        
        retrieve_k = k * 3 if rerank else k
        
        if search_mode == "dense":
            retrieved_docs = self.dense_search(query, k=retrieve_k)
        elif search_mode == "sparse":
            retrieved_docs = self.sparse_search(query, k=retrieve_k)
        else:
            retrieved_docs = self.hybrid_search(query, k=retrieve_k)
        
        if rerank and retrieved_docs:
            retrieved_docs = self._rerank_documents(query, retrieved_docs, top_k=rerank_top_k)
        
        context = self._prepare_context(retrieved_docs)
        
        result = {
            "query": query,
            "query_type": query_type,
            "search_mode": search_mode,
            "num_retrieved": len(retrieved_docs),
            "retrieved_documents": retrieved_docs if include_sources else [],
            "context": context
        }
        
        if self.llm:
            answer = self._generate_answer(query, context, query_type)
            result["answer"] = answer
        else:
            result["answer"] = "LLM not available for answer generation"
            self.logger.warning("LLM not configured, returning context only")
        
        self.logger.info("Retrieval completed successfully")
        return result
    
    def _prepare_context(self, documents: List[Tuple[Document, float]]) -> str:
        context_parts = []
        
        for i, (doc, score) in enumerate(documents):
            metadata = doc.metadata
            content = doc.page_content
            
            context_part = f"[Tài liệu {i+1} - Độ tin cậy: {score:.3f}]\n"
            context_part += f"Loại: {metadata.get('document_type', 'unknown')}\n"
            
            math_structures = metadata.get('math_structures', [])
            philosophy_structures = metadata.get('philosophy_structures', [])
            
            if math_structures:
                context_part += f"Cấu trúc Toán: {', '.join(math_structures)}\n"
            if philosophy_structures:
                context_part += f"Cấu trúc Triết: {', '.join(philosophy_structures)}\n"
            
            context_part += f"Nội dung: {content}\n"
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    def _generate_answer(self, query: str, context: str, query_type: str) -> str:
        
        if not self.llm:
            return "LLM is not avaiable"

        try:
            prompt_info = self.prompts.get(query_type, self.prompts["general"])
            prompt = prompt_info["answer"]
            
            formatted_prompt = prompt.format(context=context, question=query)
            
            self.logger.info(f"Generating answer using {self.llm_config.provider.value}")
            answer = self.llm(formatted_prompt)
            
            return answer.strip()
            
        except Exception as e:
            self.logger.error(f"Answer generation failed: {str(e)}")
    
    def get_retrieval_stats(self, query: str) -> Dict[str, Any]:
        retrieved_docs = self.hybrid_search(query, k=10)
        
        stats = {
            "query": query,
            "total_retrieved": len(retrieved_docs),
            "document_types": {},
            "average_confidence": 0,
            "score_distribution": []
        }
        
        if retrieved_docs:
            scores = [score for _, score in retrieved_docs]
            stats["average_confidence"] = sum(scores) / len(scores)
            stats["score_distribution"] = scores
            
            for doc, score in retrieved_docs:
                doc_type = doc.metadata.get("document_type", "unknown")
                stats["document_types"][doc_type] = stats["document_types"].get(doc_type, 0) + 1
        
        return stats

def create_retriever(vector_store, search_type: str = "hybrid") -> HybridRetriever:
    return HybridRetriever(vector_store, search_type)