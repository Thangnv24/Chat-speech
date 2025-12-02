# src/core/ingestor.py
import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from rank_bm25 import BM25Okapi
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.utils.logger import setup_logging
from src.utils.chunking import get_chunker
from src.config.llm_config import llm_config

logger = setup_logging("ingestor")

class BM25Embeddings:
    """BM25 sparse embeddings for hybrid search"""
    
    def __init__(self):
        self.tokenized_corpus = []
        self.bm25 = None
        self.vocab = {}
        self.documents = []
    
    def fit(self, texts: List[str]):
        """Fit BM25 on the corpus"""
        from underthesea import word_tokenize
        
        # Tokenize texts for Vietnamese
        self.tokenized_corpus = [word_tokenize(text.lower()) for text in texts]
        self.documents = texts
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Build vocabulary
        for tokens in self.tokenized_corpus:
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
    
    def embed_document(self, text: str, doc_id: int) -> List[float]:
        """Create sparse BM25 embedding for a document"""
        from underthesea import word_tokenize
        
        if not self.bm25:
            raise ValueError("BM25 not fitted. Call fit() first.")
        
        # Get BM25 scores for all documents
        tokenized_query = word_tokenize(text.lower())
        scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize scores
        if max(scores) > 0:
            scores = scores / max(scores)
        
        # Create sparse vector (only store non-zero values for efficiency)
        sparse_vector = []
        for i, score in enumerate(scores):
            if score > 0.01:  # Threshold to reduce noise
                sparse_vector.append((i, float(score)))
        
        return sparse_vector
    
    def embed_query(self, text: str) -> Dict[str, Any]:
        """Create sparse BM25 embedding for a query"""
        from underthesea import word_tokenize
        
        if not self.bm25:
            raise ValueError("BM25 not fitted. Call fit() first.")
        
        tokenized_query = word_tokenize(text.lower())
        
        # Get scores for query against all documents
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top documents
        top_n = min(10, len(doc_scores))
        top_indices = np.argsort(doc_scores)[-top_n:][::-1]
        
        # Create query vector representation
        query_terms = {}
        for token in tokenized_query:
            if token in query_terms:
                query_terms[token] += 1
            else:
                query_terms[token] = 1
        
        # Create sparse vector format
        sparse_vector = []
        for idx in top_indices:
            if doc_scores[idx] > 0:
                sparse_vector.append((int(idx), float(doc_scores[idx])))
        
        return {
            "sparse_vector": sparse_vector,
            "top_docs": [(int(idx), float(doc_scores[idx])) for idx in top_indices],
            "query_terms": list(query_terms.keys())
        }

class HybridVectorStore:
    """Hybrid vector store with dense and sparse embeddings"""
    
    def __init__(self, 
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "math_philosophy",
                 dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.dense_embeddings = HuggingFaceEmbeddings(model_name=dense_model)
        self.bm25 = BM25Embeddings()
        self.logger = logger
        
    def create_collection(self, vector_size: int = 384):
        """Create Qdrant collection with support for dense vectors only"""
        
        try:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            self.logger.info(f"Created collection {self.collection_name}")
        except Exception as e:
            if "already exists" in str(e):
                self.logger.info(f"Collection {self.collection_name} already exists")
            else:
                raise
    
    def add_documents(self, documents: List[Document], texts: List[str]):
        """Add documents with dense embeddings, sparse indices stored separately"""
        
        # Create dense embeddings
        dense_vectors = self.dense_embeddings.embed_documents([doc.page_content for doc in documents])
        
        # Prepare points for Qdrant
        points = []
        for idx, (doc, vector) in enumerate(zip(documents, dense_vectors)):
            # Prepare metadata payload
            payload = {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "chunk_id": idx,
                "document_type": doc.metadata.get("document_type", "unknown"),
                "source": doc.metadata.get("source", "unknown"),
                "math_structures": doc.metadata.get("math_structures", []),
                "philosophy_structures": doc.metadata.get("philosophy_structures", []),
                "ingestion_time": datetime.now().isoformat()
            }
            
            point = models.PointStruct(
                id=idx,
                vector=vector,
                payload=payload
            )
            points.append(point)
        
        # Upload to Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        # Train BM25 on texts
        self.bm25.fit(texts)
        
        self.logger.info(f"Added {len(documents)} documents to Qdrant")
        
        # Save BM25 data
        self._save_bm25_data()
        
        return len(documents)
    
    def _save_bm25_data(self):
        """Save BM25 data for later use"""
        bm25_data = {
            "vocab": self.bm25.vocab,
            "documents": self.bm25.documents,
            "tokenized_corpus": self.bm25.tokenized_corpus
        }
        
        os.makedirs("./vector_data", exist_ok=True)
        with open("./vector_data/bm25_data.pkl", "wb") as f:
            pickle.dump(bm25_data, f)
        
        self.logger.info("BM25 data saved")
    
    def _load_bm25_data(self):
        """Load BM25 data"""
        try:
            with open("./vector_data/bm25_data.pkl", "rb") as f:
                bm25_data = pickle.load(f)
            
            self.bm25.vocab = bm25_data["vocab"]
            self.bm25.documents = bm25_data["documents"]
            self.bm25.tokenized_corpus = bm25_data["tokenized_corpus"]
            
            # Recreate BM25 instance
            self.bm25.bm25 = BM25Okapi(self.bm25.tokenized_corpus)
            
            self.logger.info("BM25 data loaded")
            return True
        except FileNotFoundError:
            self.logger.warning("BM25 data not found")
            return False
    
    def hybrid_search(self, 
                     query: str, 
                     top_k: int = 5,
                     dense_weight: float = 0.7,
                     sparse_weight: float = 0.3) -> List[Tuple[Document, float]]:
        """Perform hybrid search combining dense and sparse retrieval"""
        
        # 1. Dense search
        dense_results = self._dense_search(query, top_k * 2)
        
        # 2. Sparse search (BM25)
        sparse_results = self._sparse_search(query, top_k * 2)
        
        # 3. Combine results using Reciprocal Rank Fusion (RRF)
        combined_results = self._combine_results_rrf(dense_results, sparse_results, top_k)
        
        return combined_results
    
    def _dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Perform dense vector search"""
        query_vector = self.dense_embeddings.embed_query(query)
        
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        
        results = []
        for result in search_results:
            doc = Document(
                page_content=result.payload["text"],
                metadata=result.payload.get("metadata", {})
            )
            results.append((doc, result.score))
        
        return results
    
    def _sparse_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Perform sparse BM25 search"""
        if not self.bm25.bm25:
            if not self._load_bm25_data():
                return []
        
        # Get BM25 results
        query_embedding = self.bm25.embed_query(query)
        top_docs = query_embedding.get("top_docs", [])
        
        results = []
        for doc_idx, score in top_docs[:top_k]:
            # Get document from Qdrant by ID
            try:
                points = self.qdrant_client.retrieve(
                    collection_name=self.collection_name,
                    ids=[doc_idx],
                    with_payload=True
                )
                
                if points:
                    point = points[0]
                    doc = Document(
                        page_content=point.payload["text"],
                        metadata=point.payload.get("metadata", {})
                    )
                    results.append((doc, float(score)))
            except Exception as e:
                self.logger.error(f"Error retrieving document {doc_idx}: {str(e)}")
        
        return results
    
    def _combine_results_rrf(self, 
                           dense_results: List[Tuple[Document, float]], 
                           sparse_results: List[Tuple[Document, float]],
                           top_k: int) -> List[Tuple[Document, float]]:
        """Combine results using Reciprocal Rank Fusion"""
        
        # Create maps for quick lookup
        dense_map = {doc.page_content: (doc, score, i+1) 
                    for i, (doc, score) in enumerate(dense_results)}
        sparse_map = {doc.page_content: (doc, score, i+1) 
                     for i, (doc, score) in enumerate(sparse_results)}
        
        all_docs = set(list(dense_map.keys()) + list(sparse_map.keys()))
        
        # Calculate RRF scores
        rrf_scores = {}
        k = 60  # RRF constant
        
        for doc_content in all_docs:
            rrf_score = 0
            
            # Add dense ranking if exists
            if doc_content in dense_map:
                _, _, rank = dense_map[doc_content]
                rrf_score += 1.0 / (k + rank)
            
            # Add sparse ranking if exists
            if doc_content in sparse_map:
                _, _, rank = sparse_map[doc_content]
                rrf_score += 1.0 / (k + rank)
            
            # Get the document
            if doc_content in dense_map:
                doc, dense_score, _ = dense_map[doc_content]
            else:
                doc, sparse_score, _ = sparse_map[doc_content]
            
            rrf_scores[doc_content] = (doc, rrf_score)
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.values(), 
                              key=lambda x: x[1], 
                              reverse=True)
        
        return sorted_results[:top_k]

class DocumentIngestor:
    """Handles document ingestion with Qdrant and hybrid embeddings"""
    
    def __init__(self, 
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "math_philosophy",
                 dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.chunker = get_chunker()
        self.vector_store = HybridVectorStore(qdrant_url, collection_name, dense_model)
        self.logger = logger
    
    def ingest_documents(self, document_paths: List[str]) -> HybridVectorStore:
        """Ingest documents into Qdrant with hybrid embeddings"""
        
        self.logger.info(f"Starting ingestion of {len(document_paths)} documents")
        
        all_chunks = []
        all_texts = []
        
        for path in document_paths:
            if not os.path.exists(path):
                self.logger.warning(f"Document path does not exist: {path}")
                continue
            
            self.logger.info(f"Processing document: {path}")
            
            try:
                # Use chunker to process documents
                if path.endswith('.pdf'):
                    chunks = self.chunker.process_pdf(path)
                elif path.endswith('.txt'):
                    chunks = self.chunker.process_text_file(path)
                else:
                    self.logger.warning(f"Unsupported file format: {path}")
                    continue
                
                # Add source metadata
                for chunk in chunks:
                    chunk.metadata["source"] = os.path.basename(path)
                    chunk.metadata["source_path"] = path
                
                all_chunks.extend(chunks)
                all_texts.extend([chunk.page_content for chunk in chunks])
                
                self.logger.info(f"Document {path} processed into {len(chunks)} chunks")
                
            except Exception as e:
                self.logger.error(f"Failed to process document {path}: {str(e)}")
                continue
        
        if not all_chunks:
            self.logger.error("No chunks were created from documents")
            raise ValueError("No valid documents processed")
        
        self.logger.info(f"Total chunks created: {len(all_chunks)}")
        
        try:
            # Create collection
            self.vector_store.create_collection()
            
            # Add documents to Qdrant
            self.vector_store.add_documents(all_chunks, all_texts)
            
            # Save ingestion metadata
            self._save_ingestion_metadata(all_chunks)
            
            self.logger.info(f"Documents ingested successfully into Qdrant collection: {self.collection_name}")
            
            return self.vector_store
            
        except Exception as e:
            self.logger.error(f"Vector store creation failed: {str(e)}")
            raise
    
    def _save_ingestion_metadata(self, chunks: List[Document]):
        """Save metadata about the ingestion process"""
        metadata = {
            "total_chunks": len(chunks),
            "document_types": {},
            "chunk_sizes": [],
            "math_structures_count": 0,
            "philosophy_structures_count": 0,
            "ingestion_time": datetime.now().isoformat(),
            "collection_name": self.collection_name,
            "qdrant_url": self.qdrant_url
        }
        
        for chunk in chunks:
            doc_type = chunk.metadata.get("document_type", "unknown")
            metadata["document_types"][doc_type] = metadata["document_types"].get(doc_type, 0) + 1
            
            metadata["chunk_sizes"].append(chunk.metadata.get("chunk_size", 0))
            
            math_structures = chunk.metadata.get("math_structures", [])
            philosophy_structures = chunk.metadata.get("philosophy_structures", [])
            
            metadata["math_structures_count"] += len(math_structures)
            metadata["philosophy_structures_count"] += len(philosophy_structures)
        
        # Save metadata to file
        os.makedirs("./vector_data", exist_ok=True)
        metadata_path = "./vector_data/ingestion_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        self.logger.info(f"Ingestion metadata saved: {metadata}")
    
    def load_existing_vector_store(self) -> Optional[HybridVectorStore]:
        """Load existing vector store from Qdrant"""
        try:
            # Check if collection exists
            collections = self.vector_store.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                # Load BM25 data
                self.vector_store._load_bm25_data()
                self.logger.info(f"Existing vector store loaded from Qdrant: {self.collection_name}")
                return self.vector_store
            else:
                self.logger.warning(f"Collection {self.collection_name} does not exist")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load existing vector store: {str(e)}")
            return None
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            # Get collection info from Qdrant
            collection_info = self.vector_store.qdrant_client.get_collection(
                collection_name=self.collection_name
            )
            
            # Load ingestion metadata
            metadata_path = "./vector_data/ingestion_metadata.pkl"
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
            
            return {
                "collection_name": self.collection_name,
                "vector_count": collection_info.vectors_count,
                "qdrant_url": self.qdrant_url,
                "indexed_vectors": collection_info.indexed_vectors_count,
                "pending_vectors": collection_info.pending_vectors_count,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": str(collection_info.config.params.vectors.distance)
                },
                **metadata
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get vector store stats: {str(e)}")
            return {"error": str(e)}

# Factory function
def create_ingestor(qdrant_url: str = "http://localhost:6333",
                   collection_name: str = "math_philosophy") -> DocumentIngestor:
    """Create a document ingestor instance"""
    return DocumentIngestor(qdrant_url, collection_name)