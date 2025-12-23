import os
import pickle
from typing import List, Dict, Any
from datetime import datetime
from langchain_core.documents import Document
from qdrant_client import models
import uuid

from app.utils.logger import setup_logging
from app.service.RAG.chunking import get_chunker
from batch_processor import BatchProcessor

logger = setup_logging("ingestion_pipeline")

class IngestionPipeline:
    def __init__(self, chunker=None, batch_size: int = 32, max_workers: int = 4):
        self.chunker = chunker or get_chunker()
        self.batch_processor = BatchProcessor(batch_size, max_workers)
        self.logger = logger
    
    def process_documents(self, document_paths: List[str]) -> List[Document]:
        all_chunks = []
        
        for path in document_paths:
            if not os.path.exists(path):
                self.logger.warning(f"Path not found: {path}")
                continue
            
            try:
                if path.endswith('.pdf'):
                    chunks = self.chunker.process_pdf(path)
                elif path.endswith('.txt'):
                    chunks = self.chunker.process_text_file(path)
                else:
                    self.logger.warning(f"Unsupported format: {path}")
                    continue
                
                for chunk in chunks:
                    chunk.metadata.update({
                        "source": os.path.basename(path),
                        "source_path": path
                    })
                
                all_chunks.extend(chunks)
                self.logger.info(f"Processed {path}: {len(chunks)} chunks")
                
            except Exception as e:
                self.logger.error(f"Failed {path}: {e}")
                continue
        
        if not all_chunks:
            raise ValueError("No valid documents processed")
        
        return all_chunks
    
    def prepare_vectors(self, documents: List[Document], embedder, sparse_fn) -> List[Dict]:
        texts = [doc.page_content for doc in documents]
        
        dense_vectors = self.batch_processor.embed_documents_parallel(embedder, texts)
        
        sparse_vectors = self.batch_processor.process_parallel(
            texts, 
            lambda batch: [sparse_fn(text) for text in batch]
        )
        
        points = []
        for doc, dense_vec, sparse_vec in zip(documents, dense_vectors, sparse_vectors):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content))
            points.append(models.PointStruct(
                id=point_id,
                vector={"": dense_vec, "bm25": sparse_vec},
                payload={"text": doc.page_content, "metadata": doc.metadata}
            ))
        
        return points
    
    def save_metadata(self, chunks: List[Document], collection_name: str, qdrant_url: str):
        metadata = {
            "total_chunks": len(chunks),
            "document_types": {},
            "chunk_sizes": [],
            "math_structures_count": 0,
            "philosophy_structures_count": 0,
            "ingestion_time": datetime.now().isoformat(),
            "collection_name": collection_name,
            "qdrant_url": qdrant_url
        }
        
        for chunk in chunks:
            doc_type = chunk.metadata.get("document_type", "unknown")
            metadata["document_types"][doc_type] = metadata["document_types"].get(doc_type, 0) + 1
            metadata["chunk_sizes"].append(chunk.metadata.get("chunk_size", 0))
            metadata["math_structures_count"] += len(chunk.metadata.get("math_structures", []))
            metadata["philosophy_structures_count"] += len(chunk.metadata.get("philosophy_structures", []))
        
        os.makedirs("./vector_data", exist_ok=True)
        with open("./vector_data/ingestion_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        self.logger.info(f"Metadata saved: {metadata['total_chunks']} chunks")
        return metadata