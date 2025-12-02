# src/core/retriever.py
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from src.utils.logger import setup_logging
from src.config.prompts import PROMPT_MAP
from src.config.llm_config import llm_config

logger = setup_logging("retriever")

class HybridRetriever:
    """Hybrid retriever combining dense and semantic search"""
    
    def __init__(self, 
                 vector_store: Chroma,
                 search_type: str = "hybrid"):
        
        self.vector_store = vector_store
        self.search_type = search_type
        self.logger = logger

        self.llm_config = llm_config
        self.llm = self.llm_config.get_llm_client()
        
        self.prompts = PROMPT_MAP
    
    def _initialize_llm(self, model_path: str) -> LlamaCpp:
        """Initialize LLM for answer generation and re-ranking"""
        try:
            self.logger.info(f"Initializing LLM for retriever: {model_path}")
            
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            llm = LlamaCpp(
                model_path=model_path,
                temperature=0.3,
                max_tokens=512,
                top_p=0.9,
                top_k=40,
                n_ctx=2048,
                callback_manager=callback_manager,
                verbose=False
            )
            
            self.logger.info("Retriever LLM initialized successfully")
            return llm
            
        except Exception as e:
            self.logger.error(f"Retriever LLM initialization failed: {str(e)}")
            return None
    
    def _setup_prompts(self):
        pass
        
    
    def _detect_query_type(self, query: str) -> str:
        """Detect if query is about mathematics, philosophy, or general"""
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
                     score_threshold: float = 0.7) -> List[Tuple[Document, float]]:
        """Perform hybrid search combining multiple retrieval methods"""
        
        self.logger.info(f"Performing hybrid search for: {query}")
        
        try:
            # Method 1: Standard similarity search
            docs1 = self.vector_store.similarity_search_with_score(
                query, k=k * 2
            )
            
            # Method 2: MMR for diversity
            docs2 = self.vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=k * 3
            )
            
            # Convert MMR results to match format
            docs2_with_scores = [(doc, 1.0) for doc in docs2]
            
            # Combine and re-rank results
            all_docs = self._combine_and_rerank(docs1, docs2_with_scores, query, k)
            
            # Filter by score threshold
            filtered_docs = [(doc, score) for doc, score in all_docs if score >= score_threshold]
            
            self.logger.info(f"Hybrid search returned {len(filtered_docs)} documents")
            return filtered_docs[:k]
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {str(e)}")
            # Fallback to simple search
            return self.vector_store.similarity_search_with_score(query, k=k)
    
    def _combine_and_rerank(self, 
                          docs1: List[Tuple[Document, float]], 
                          docs2: List[Tuple[Document, float]],
                          query: str,
                          k: int) -> List[Tuple[Document, float]]:
        """Combine and re-rank search results"""
        
        # Combine all documents
        all_docs = {}
        for doc, score in docs1 + docs2:
            if doc.page_content not in all_docs:
                all_docs[doc.page_content] = (doc, score)
            else:
                # Keep the higher score
                existing_doc, existing_score = all_docs[doc.page_content]
                if score > existing_score:
                    all_docs[doc.page_content] = (doc, score)
        
        # Convert back to list
        unique_docs = list(all_docs.values())
        
        # Simple re-ranking: sort by score
        reranked_docs = sorted(unique_docs, key=lambda x: x[1], reverse=True)
        
        return reranked_docs[:k]
    
    def retrieve(self, 
                query: str, 
                k: int = 5,
                include_sources: bool = True) -> Dict[str, Any]:
        """Main retrieval method"""
        
        self.logger.info(f"Retrieving documents for query: {query}")
        
        # Detect query type for prompt selection
        query_type = self._detect_query_type(query)
        self.logger.info(f"Detected query type: {query_type}")
        
        # Perform hybrid search
        retrieved_docs = self.hybrid_search(query, k=k)
        
        # Prepare context from retrieved documents
        context = self._prepare_context(retrieved_docs)
        
        result = {
            "query": query,
            "query_type": query_type,
            "retrieved_documents": retrieved_docs if include_sources else [],
            "context": context
        }
        
        # Generate answer if LLM is available
        if self.llm:
            answer = self._generate_answer(query, context, query_type)
            result["answer"] = answer
        else:
            result["answer"] = "LLM not available for answer generation"
        
        self.logger.info("Retrieval completed successfully")
        return result
    
    def _prepare_context(self, documents: List[Tuple[Document, float]]) -> str:
        """Prepare context from retrieved documents"""
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
        """Generate answer using LLM"""
        
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
        """Get statistics about the retrieval process"""
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

# Factory function
def create_retriever(vector_store: Chroma, llm_model_path: Optional[str] = None) -> HybridRetriever:
    """Create a hybrid retriever instance"""
    return HybridRetriever(vector_store, llm_model_path)