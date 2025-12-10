import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from app.utils.logger import setup_logging
from app.config.prompts import PROMPT_MAP
from app.config.llm_config import llm_config

logger = setup_logging("retriever")

class HybridRetriever:
    # Hybrid retriever for Qdrant with dense + BM25 sparse search
    
    def __init__(self, 
                 vector_store,  # HybridVectorStore from ingestor
                 search_type: str = "hybrid"):
        
        self.vector_store = vector_store
        self.search_type = search_type
        self.logger = logger

        self.llm_config = llm_config
        self.llm = self.llm_config.get_llm_client()
        
        self.prompts = PROMPT_MAP
    
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
                     k: int = 5,
                     dense_weight: float = 0.7,
                     sparse_weight: float = 0.3) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search using Qdrant with dense + BM25 sparse retrieval
        
        Args:
            query: Search query
            k: Number of results to return
            dense_weight: Weight for dense embeddings (0-1)
            sparse_weight: Weight for sparse BM25 (0-1)
        
        Returns:
            List of (Document, score) tuples
        """
        self.logger.info(f"Performing hybrid search for: {query}")
        
        try:
            # Use HybridVectorStore's hybrid_search method
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
            # Fallback to dense-only search
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
    
    def retrieve(self, 
                query: str, 
                k: int = 5,
                include_sources: bool = True,
                search_mode: str = "hybrid") -> Dict[str, Any]:
        """
        Main retrieval method with answer generation
        
        Args:
            query: User query
            k: Number of documents to retrieve
            include_sources: Include source documents in response
            search_mode: "hybrid", "dense", or "sparse"
        
        Returns:
            Dict with query, answer, context, and metadata
        """
        self.logger.info(f"Retrieving documents for query: {query}")
        
        # Detect query type for prompt selection
        query_type = self._detect_query_type(query)
        self.logger.info(f"Detected query type: {query_type}")
        
        # Perform search based on mode
        if search_mode == "dense":
            retrieved_docs = self.dense_search(query, k=k)
        elif search_mode == "sparse":
            retrieved_docs = self.sparse_search(query, k=k)
        else:  # hybrid
            retrieved_docs = self.hybrid_search(query, k=k)
        
        # Prepare context from retrieved documents
        context = self._prepare_context(retrieved_docs)
        
        result = {
            "query": query,
            "query_type": query_type,
            "search_mode": search_mode,
            "num_retrieved": len(retrieved_docs),
            "retrieved_documents": retrieved_docs if include_sources else [],
            "context": context
        }
        
        # Generate answer if LLM is available
        if self.llm:
            answer = self._generate_answer(query, context, query_type)
            result["answer"] = answer
        else:
            result["answer"] = "LLM not available for answer generation"
            self.logger.warning("LLM not configured, returning context only")
        
        self.logger.info("Retrieval completed successfully")
        return result
    
    def _prepare_context(self, documents: List[Tuple[Document, float]]) -> str:
        # Prepare context from retrieved documents
        context_parts = []
        
        for i, (doc, score) in enumerate(documents):
            metadata = doc.metadata
            content = doc.page_content
            
            context_part = f"[Tài liệu {i+1} - Độ tin cậy: {score:.3f}]\n"
            context_part += f"Loại: {metadata.get('document_type', 'unknown')}\n"
            
            # Add relevant metadata
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
        # Generate answer using LLM
        
        if not self.llm:
            return "LLM is not avaiable"

        try:
            prompt_info = self.prompts.get(query_type, self.prompts["general"])
            prompt = prompt_info["answer"]
            
            # Format the prompt
            formatted_prompt = prompt.format(context=context, question=query)
            
            # Generate answer
            self.logger.info(f"Generating answer using {self.llm_config.provider.value}")
            answer = self.llm(formatted_prompt)
            
            return answer.strip()
            
        except Exception as e:
            self.logger.error(f"Answer generation failed: {str(e)}")
            return f"Không thể tạo câu trả lời: {str(e)}"
    
    def get_retrieval_stats(self, query: str) -> Dict[str, Any]:
        # Get statistics about the retrieval process
        retrieved_docs = self.hybrid_search(query, k=5)
        
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