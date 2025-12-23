import os
from typing import List, Tuple, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
from underthesea import word_tokenize
import uuid
import pickle
import threading


from app.utils.logger import setup_logging
from app.service.RAG.chunking import get_chunker
from ingestion_pipeline import IngestionPipeline

logger = setup_logging("Ingestor")
qdrant_url = os.getenv("QDRANT_URL")


class ThreadSafeTokenizer:
    _lock = threading.Lock()
    
    @classmethod
    def tokenize(cls, text: str):
        with cls._lock:
            return word_tokenize(text.lower())

class HybridVectorStore:
    def __init__(self, qdrant_url: str = qdrant_url,
                 collection_name: str = "math_philosophy",
                 dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.dense_embeddings = HuggingFaceEmbeddings(model_name=dense_model)
        self.logger = logger
        
    def create_collection(self, vector_size: int = 384):
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            sparse_vectors_config={"bm25": models.SparseVectorParams(index=models.SparseIndexParams())}
        )
        self.logger.info(f"Created collection {self.collection_name}")

    def _get_sparse_vector(self, text: str) -> models.SparseVector:
        # tokens = word_tokenize(text.lower())
        tokens = ThreadSafeTokenizer().tokenize(text)
        token_counts = {}
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1
        
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:100]
        
        index_value_map = {}
        for token, count in sorted_tokens:
            idx = abs(hash(token)) % 10000
            index_value_map[idx] = index_value_map.get(idx, 0.0) + float(count)
        
        return models.SparseVector(
            indices=list(index_value_map.keys()),
            values=list(index_value_map.values())
        )

    def add_points(self, points: List[models.PointStruct]):
        self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
        self.logger.info(f"Ingested {len(points)} points")
    
    def _dense_search(self, query: str, top_k: int):
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
        sparse_vec = self._get_sparse_vector(query)
        response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=sparse_vec,
            using="bm25",
            limit=top_k,
            with_payload=True
        )
        return [(Document(page_content=p.payload["text"], metadata=p.payload.get("metadata", {})), p.score) 
                for p in response.points]

    def hybrid_search(self, query: str, top_k: int = 5, 
                    dense_weight: float = 0.7,
                    sparse_weight: float = 0.3) -> List[Tuple[Document, float]]:
        dense_vec = self.dense_embeddings.embed_query(query)
        sparse_vec = self._get_sparse_vector(query)

        try:
            response = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(query=dense_vec, limit=top_k),
                    models.Prefetch(query=sparse_vec, using="bm25", limit=top_k)
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True
            )
            return [(Document(page_content=p.payload["text"], metadata=p.payload.get("metadata", {})), p.score) 
                    for p in response.points]
        
        except Exception as e:
            self.logger.warning(f"Hybrid search failed: {e}. Using fallback.")
            return self._combine_results_rrf(
                self._dense_search(query, top_k),
                self._sparse_search(query, top_k),
                top_k
            )

    def _combine_results_rrf(self, dense_results: List[Tuple[Document, float]], 
                           sparse_results: List[Tuple[Document, float]],
                           top_k: int) -> List[Tuple[Document, float]]:
        dense_map = {doc.page_content: (doc, score, i+1) for i, (doc, score) in enumerate(dense_results)}
        sparse_map = {doc.page_content: (doc, score, i+1) for i, (doc, score) in enumerate(sparse_results)}
        all_docs = set(list(dense_map.keys()) + list(sparse_map.keys()))
        
        rrf_scores = {}
        k = 60
        
        for doc_content in all_docs:
            rrf_score = 0
            if doc_content in dense_map:
                rrf_score += 1.0 / (k + dense_map[doc_content][2])
            if doc_content in sparse_map:
                rrf_score += 1.0 / (k + sparse_map[doc_content][2])
            
            doc = dense_map[doc_content][0] if doc_content in dense_map else sparse_map[doc_content][0]
            rrf_scores[doc_content] = (doc, rrf_score)
        
        return sorted(rrf_scores.values(), key=lambda x: x[1], reverse=True)[:top_k]

class DocumentIngestor:
    def __init__(self, qdrant_url: str = qdrant_url,
                 collection_name: str = "math_philosophy",
                 dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 32,
                 max_workers: int = 4):
        
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.vector_store = HybridVectorStore(qdrant_url, collection_name, dense_model)
        self.pipeline = IngestionPipeline(
            chunker=get_chunker(),
            batch_size=batch_size,
            max_workers=max_workers
        )
        self.logger = logger
    
    def ingest_documents(self, document_paths: List[str]) -> HybridVectorStore:
        self.logger.info(f"Starting ingestion: {len(document_paths)} documents")
        
        chunks = self.pipeline.process_documents(document_paths)
        self.logger.info(f"Total chunks: {len(chunks)}")
        
        self.vector_store.create_collection()
        
        points = self.pipeline.prepare_vectors(
            chunks,
            self.vector_store.dense_embeddings,
            self.vector_store._get_sparse_vector
        )
        
        self.vector_store.add_points(points)
        self.pipeline.save_metadata(chunks, self.collection_name, self.qdrant_url)
        
        self.logger.info("Ingestion completed")
        return self.vector_store
    
    def load_existing_vector_store(self) -> Optional[HybridVectorStore]:
        try:
            collections = self.vector_store.qdrant_client.get_collections()
            if self.collection_name in [col.name for col in collections.collections]:
                self.logger.info(f"Connected to collection: {self.collection_name}")
                return self.vector_store
            else:
                self.logger.warning(f"Collection not found: {self.collection_name}")
                return None
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return None
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        try:
            collection_info = self.vector_store.qdrant_client.get_collection(self.collection_name)
            
            metadata = {}
            metadata_path = "./vector_data/ingestion_metadata.pkl"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
            
            return {
                "collection_name": self.collection_name,
                "vector_count": collection_info.points_count,
                "qdrant_url": self.qdrant_url,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": str(collection_info.config.params.vectors.distance)
                },
                **metadata
            }
        except Exception as e:
            self.logger.error(f"Stats failed: {e}")
            return {"error": str(e)}

def create_ingestor(qdrant_url: str = qdrant_url,
                   collection_name: str = "math_philosophy",
                   batch_size: int = 32,
                   max_workers: int = 4) -> DocumentIngestor:
    return DocumentIngestor(qdrant_url, collection_name, batch_size=batch_size, max_workers=max_workers)