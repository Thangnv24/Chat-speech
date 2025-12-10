"""
RAG Pipeline - Orchestrates the complete RAG workflow
Handles: Document Ingestion → Vector Storage → Retrieval → Generation
"""

import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from app.utils.logger import setup_logging
from app.service.RAG.ingestor import DocumentIngestor, HybridVectorStore
from app.service.RAG.retriever import HybridRetriever, create_retriever

logger = setup_logging("rag_pipeline")


class RAGPipeline:
    """Complete RAG pipeline with ingestion, retrieval, and generation"""
    
    def __init__(self,
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "math_philosophy",
                 dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize RAG Pipeline
        
        Args:
            qdrant_url: Qdrant server URL
            collection_name: Name of the collection in Qdrant
            dense_model: HuggingFace model for dense embeddings
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.dense_model = dense_model
        
        self.ingestor = None
        self.vector_store = None
        self.retriever = None
        
        self.logger = logger
        self.logger.info(f"RAG Pipeline initialized for collection: {collection_name}")
    
    def ingest_documents(self, 
                        document_paths: List[str],
                        force_reingest: bool = False) -> Dict[str, Any]:
        """
        Ingest documents into vector database
        
        Args:
            document_paths: List of paths to PDF/TXT files
            force_reingest: Force re-ingestion even if collection exists
        
        Returns:
            Dict with ingestion statistics
        """
        self.logger.info(f"Starting document ingestion for {len(document_paths)} files")
        start_time = time.time()
        
        # Initialize ingestor
        self.ingestor = DocumentIngestor(
            qdrant_url=self.qdrant_url,
            collection_name=self.collection_name,
            dense_model=self.dense_model
        )
        
        # Check if collection already exists
        if not force_reingest:
            existing_store = self.ingestor.load_existing_vector_store()
            if existing_store:
                self.logger.info("Using existing vector store")
                self.vector_store = existing_store
                
                stats = self.ingestor.get_vector_store_stats()
                stats["ingestion_time"] = 0
                stats["status"] = "loaded_existing"
                return stats
        
        # Validate document paths
        valid_paths = []
        for path in document_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                self.logger.warning(f"File not found: {path}")
        
        if not valid_paths:
            raise ValueError("No valid document paths provided")
        
        # Ingest documents
        try:
            self.vector_store = self.ingestor.ingest_documents(valid_paths)
            
            ingestion_time = time.time() - start_time
            
            # Get statistics
            stats = self.ingestor.get_vector_store_stats()
            stats["ingestion_time"] = ingestion_time
            stats["status"] = "success"
            stats["documents_processed"] = len(valid_paths)
            
            self.logger.info(f"Ingestion completed in {ingestion_time:.2f}s")
            return stats
            
        except Exception as e:
            self.logger.error(f"Ingestion failed: {str(e)}")
            raise
    
    def load_existing_store(self) -> bool:
        """
        Load existing vector store from Qdrant
        
        Returns:
            True if loaded successfully, False otherwise
        """
        self.logger.info("Loading existing vector store")
        
        try:
            self.ingestor = DocumentIngestor(
                qdrant_url=self.qdrant_url,
                collection_name=self.collection_name,
                dense_model=self.dense_model
            )
            
            self.vector_store = self.ingestor.load_existing_vector_store()
            
            if self.vector_store:
                self.logger.info("Vector store loaded successfully")
                return True
            else:
                self.logger.warning("No existing vector store found")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load vector store: {str(e)}")
            return False
    
    def initialize_retriever(self, search_type: str = "hybrid") -> HybridRetriever:
        """
        Initialize retriever with loaded vector store
        
        Args:
            search_type: Type of search ("hybrid", "dense", "sparse")
        
        Returns:
            HybridRetriever instance
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Run ingest_documents() or load_existing_store() first")
        
        self.retriever = create_retriever(self.vector_store, search_type)
        self.logger.info(f"Retriever initialized with search type: {search_type}")
        
        return self.retriever
    
    def query(self,
             query: str,
             k: int = 5,
             search_mode: str = "hybrid",
             include_sources: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            query: User query
            k: Number of documents to retrieve
            search_mode: "hybrid", "dense", or "sparse"
            include_sources: Include source documents in response
        
        Returns:
            Dict with answer, context, and metadata
        """
        if not self.retriever:
            self.initialize_retriever()
        
        self.logger.info(f"Processing query: {query}")
        start_time = time.time()
        
        try:
            result = self.retriever.retrieve(
                query=query,
                k=k,
                include_sources=include_sources,
                search_mode=search_mode
            )
            
            query_time = time.time() - start_time
            result["query_time"] = query_time
            
            self.logger.info(f"Query completed in {query_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            raise
    
    def batch_query(self,
                   queries: List[str],
                   k: int = 5,
                   search_mode: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch
        
        Args:
            queries: List of queries
            k: Number of documents per query
            search_mode: Search mode for all queries
        
        Returns:
            List of results for each query
        """
        self.logger.info(f"Processing batch of {len(queries)} queries")
        
        results = []
        for i, query in enumerate(queries):
            self.logger.info(f"Processing query {i+1}/{len(queries)}")
            try:
                result = self.query(query, k=k, search_mode=search_mode)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Query {i+1} failed: {str(e)}")
                results.append({
                    "query": query,
                    "error": str(e),
                    "status": "failed"
                })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        if not self.ingestor:
            return {"error": "System not initialized"}
        
        return self.ingestor.get_vector_store_stats()
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all components"""
        health = {
            "status": "healthy",
            "components": {}
        }
        
        # Check Qdrant connection
        try:
            if self.vector_store:
                health["components"]["qdrant"] = "connected"
            else:
                health["components"]["qdrant"] = "not_initialized"
        except Exception as e:
            health["components"]["qdrant"] = f"error: {str(e)}"
            health["status"] = "unhealthy"
        
        # Check retriever
        health["components"]["retriever"] = "initialized" if self.retriever else "not_initialized"
        
        # Check LLM
        if self.retriever and self.retriever.llm:
            health["components"]["llm"] = "available"
        else:
            health["components"]["llm"] = "not_available"
        
        return health


# Convenience function
def create_pipeline(qdrant_url: str = "http://localhost:6333",
                   collection_name: str = "math_philosophy") -> RAGPipeline:
    """Create a RAG pipeline instance"""
    return RAGPipeline(qdrant_url, collection_name)
