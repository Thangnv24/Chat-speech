import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.service.RAG.rag_pipeline import create_pipeline
from app.utils.logger import setup_logging

logger = setup_logging("rag_simple")
QDRANT_URL = os.getenv("QDRANT_URL")

class SimpleRAG:
    def __init__(self, qdrant_url=QDRANT_URL, collection_name="math_philosophy"):
        self.pipeline = create_pipeline(qdrant_url=qdrant_url, collection_name=collection_name)
        self.is_ready = False
    
    def ingest(self, document_paths):
        stats = self.pipeline.ingest_documents(document_paths)
        print(f"Processed {stats.get('documents_processed', 0)} tai lieu")
        print(f"Chunks: {stats.get('total_chunks', 0)}")
        print(f"Time: {stats.get('ingestion_time', 0):.2f}s")
        return True
    
    def setup(self, search_type="hybrid"):
        if not self.pipeline.load_existing_store():
            print("Vector store not found")
            return False
        self.pipeline.initialize_retriever(search_type=search_type)
        self.is_ready = True
        return True
    
    def query(self, question, k=5, search_mode="hybrid"):
        if not self.is_ready:
            return "Run setup() first"
        
        result = self.pipeline.query(
            query=question,
            k=k,
            search_mode=search_mode,
            include_sources=True
        )
        return result.get('answer', 'There is no answer')
    
    def check_health(self):
        health = self.pipeline.health_check()
        print(f"Trang thai: {health['status']}")
        for component, status in health['components'].items():
            print(f"  {component}: {status}")
        return health['status'] == 'healthy'
    
    # Lay thong tin thong ke
    def get_info(self):
        if self.pipeline.load_existing_store():
            stats = self.pipeline.get_stats()
            print(f"Collection: {stats.get('collection_name', 'N/A')}")
            print(f"Vectors: {stats.get('vector_count', 0)}")
            print(f"Chunks: {stats.get('total_chunks', 0)}")
            return stats
        return None


def chat_terminal():
    rag = SimpleRAG()
    if not rag.setup():
        print("Do you want to ingest document? (y/n): ", end="")
        choice = input().strip().lower()
        
        if choice == 'y':
            print("\nPath(commo):")
            paths = input().strip().split(',')
            paths = [p.strip() for p in paths]
            
            if rag.ingest(paths):
                rag.setup()
            else:
                print("Can not ingest - Exit")
                return
        else:
            return
    
    while True:
        try:
            question = input("\nInput: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit']:
                break
            
            if question.lower() == 'info':
                print()
                rag.get_info()
                continue
            
            if question.lower() == 'health':
                print()
                rag.check_health()
                continue
            
            answer = rag.query(question)
            
            print(f"\nAnswer: {answer}")    
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    chat_terminal()