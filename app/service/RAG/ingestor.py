import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client import models
import hashlib
import uuid

# from qdrant_client.http import models
from app.utils.logger import setup_logging
from app.service.RAG.chunking import get_chunker
from app.config.llm_config import llm_config

from underthesea import word_tokenize

logger = setup_logging("ingestor")

class HybridVectorStore:
    # Hybrid vector store with dense and sparse
    def __init__(self, 
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "math_philosophy",
                 dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.dense_embeddings = HuggingFaceEmbeddings(model_name=dense_model)
        self.logger = logger
        
    def create_collection(self, vector_size: int = 384):
        """Tạo collection với sparse vector support"""
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size, 
                distance=models.Distance.COSINE
            ),
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    index=models.SparseIndexParams()
                )
            }
        )
        self.logger.info(f"Created collection {self.collection_name} with Sparse support")

    def _generate_id(self, text: str) -> str:
        """Tạo UUID từ text để tránh collision"""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))

    def _get_sparse_vector(self, text: str) -> models.SparseVector:
        """
        Tạo sparse vector từ text với indices UNIQUE
        FIX: Merge duplicate indices bằng cách cộng dồn values
        """
        tokens = word_tokenize(text.lower())
        token_counts = {}
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1
        
        # Giới hạn chỉ lấy top 100 tokens quan trọng nhất
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:100]
        
        # Dùng dict để tự động merge duplicate indices
        index_value_map = {}
        for token, count in sorted_tokens:
            idx = abs(hash(token)) % 10000
            # Nếu index đã tồn tại, cộng dồn giá trị (merge)
            if idx in index_value_map:
                index_value_map[idx] += float(count)
            else:
                index_value_map[idx] = float(count)
        
        # Chuyển sang lists - indices đã unique
        indices = list(index_value_map.keys())
        values = list(index_value_map.values())
        
        return models.SparseVector(indices=indices, values=values)

    def add_documents(self, documents: List[Document], texts: List[str]):
        """Thêm documents với cả dense và sparse vectors"""
        dense_vectors = self.dense_embeddings.embed_documents([doc.page_content for doc in documents])
        
        points = []
        for idx, (doc, vector) in enumerate(zip(documents, dense_vectors)):
            point_id = self._generate_id(doc.page_content)
            
            # Tạo sparse vector với unique indices
            sparse_vector = self._get_sparse_vector(doc.page_content)

            points.append(models.PointStruct(
                id=point_id,
                vector={
                    "": vector,          # Dense vector mặc định
                    "bm25": sparse_vector # Sparse vector với unique indices
                },
                payload={"text": doc.page_content, "metadata": doc.metadata}
            ))

        self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
        self.logger.info(f"Ingested {len(points)} points with synced IDs and Sparse vectors")
    
    def _dense_search(self, query: str, top_k: int):
        """Tìm kiếm dense vector"""
        query_vector = self.dense_embeddings.embed_query(query)
        
        response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )
        return [(Document(page_content=p.payload["text"], metadata=p.payload.get("metadata", {})), p.score) 
                for p in response.points]
            
    def _sparse_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """Tìm kiếm sparse vector"""
        sparse_vec = self._get_sparse_vector(query)
        
        # SỬA: Dùng query_points với using parameter
        response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=sparse_vec,  # Truyền trực tiếp SparseVector
            using="bm25",      # Chỉ định sử dụng sparse vector "bm25"
            limit=top_k,
            with_payload=True
        )
        return [(Document(page_content=p.payload["text"], metadata=p.payload.get("metadata", {})), p.score) 
                for p in response.points]

    def hybrid_search(self, 
                    query: str, 
                    top_k: int = 5,
                    dense_weight: float = 0.7,
                    sparse_weight: float = 0.3) -> List[Tuple[Document, float]]:
        """
        Hybrid search kết hợp dense và sparse với RRF
        """
        dense_vec = self.dense_embeddings.embed_query(query)
        sparse_vec = self._get_sparse_vector(query)

        try:
            # SỬA: Sử dụng Prefetch với using parameter thay vì NamedSparseVector
            response = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    # Dense search
                    models.Prefetch(
                        query=dense_vec,
                        limit=top_k
                    ),
                    # Sparse search với using parameter
                    models.Prefetch(
                        query=sparse_vec,
                        using="bm25",  # ← ĐÂY LÀ KEY FIX!
                        limit=top_k
                    )
                ],
                # Hợp nhất kết quả bằng RRF
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True
            )

            return [(Document(page_content=p.payload["text"], metadata=p.payload.get("metadata", {})), p.score) 
                    for p in response.points]
        
        except Exception as e:
            self.logger.warning(f"Native hybrid search failed: {e}. Falling back to manual RRF.")
            # Fallback: Manual RRF nếu native không hoạt động
            dense_results = self._dense_search(query, top_k)
            sparse_results = self._sparse_search(query, top_k)
            return self._combine_results_rrf(dense_results, sparse_results, top_k)

    def _combine_results_rrf(self, 
                           dense_results: List[Tuple[Document, float]], 
                           sparse_results: List[Tuple[Document, float]],
                           top_k: int) -> List[Tuple[Document, float]]:
        """Manual RRF fusion"""
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
    # Handles document ingestion with Qdrant and hybrid embeddings
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
        # Ingest documents into Qdrant with hybrid embeddings
        
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
        try:
            collections = self.vector_store.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                self.logger.info(f"Successfully connected to existing Qdrant collection: {self.collection_name}")
                return self.vector_store
            else:
                self.logger.warning(f"Collection {self.collection_name} does not exist on Qdrant server")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {str(e)}")
            return None
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        # Get statistics about the vector store
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
                "vector_count": collection_info.points_count,
                "qdrant_url": self.qdrant_url,
                "indexed_vectors": collection_info.indexed_vectors_count if hasattr(collection_info, 'indexed_vectors_count') else None,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": str(collection_info.config.params.vectors.distance)
                },
                **metadata
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get vector store stats: {str(e)}")
            return {"error": str(e)}

def create_ingestor(qdrant_url: str = "http://localhost:6333",
                   collection_name: str = "math_philosophy") -> DocumentIngestor:
    return DocumentIngestor(qdrant_url, collection_name)