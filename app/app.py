# # main.py - Luồng hoạt động đầy đủ
# import os
# from typing import List
# from src.core.ingestor import create_ingestor
# from src.core.retriever import create_retriever
# from src.utils.logger import setup_logging
# from src.utils.chunking import get_chunker

# logger = setup_logging("main")

# def setup_environment():
#     """Setup environment variables"""
#     os.environ["GEMINI_API_KEY"] = "your_gemini_api_key_here"  # Optional
#     os.environ["QWEN_API_KEY"] = "your_qwen_api_key_here"  # Optional
    
#     # Create necessary directories
#     os.makedirs("./data", exist_ok=True)
#     os.makedirs("./vector_data", exist_ok=True)
#     os.makedirs("./logs", exist_ok=True)

# def chunk_documents(document_paths: List[str]) -> List:
#     """Chunk documents using advanced chunker"""
#     logger.info("Starting document chunking...")
    
#     chunker = get_chunker()
#     all_chunks = []
    
#     for path in document_paths:
#         if not os.path.exists(path):
#             logger.warning(f"Document not found: {path}")
#             continue
        
#         logger.info(f"Chunking document: {path}")
        
#         try:
#             if path.endswith('.pdf'):
#                 chunks = chunker.process_pdf(path)
#             elif path.endswith('.txt'):
#                 chunks = chunker.process_text_file(path)
#             else:
#                 logger.warning(f"Unsupported format: {path}")
#                 continue
            
#             all_chunks.extend(chunks)
#             logger.info(f"Chunked {path} into {len(chunks)} chunks")
            
#         except Exception as e:
#             logger.error(f"Chunking failed for {path}: {str(e)}")
    
#     logger.info(f"Total chunks created: {len(all_chunks)}")
#     return all_chunks

# def ingest_to_qdrant(document_paths: List[str]) -> object:
#     """Ingest documents to Qdrant"""
#     logger.info("Starting document ingestion...")
    
#     # Create ingestor with Qdrant
#     ingestor = create_ingestor(
#         qdrant_url="http://localhost:6333",
#         collection_name="math_philosophy_docs"
#     )
    
#     try:
#         # Check if vector store already exists
#         existing_store = ingestor.load_existing_vector_store()
        
#         if existing_store:
#             logger.info("Using existing vector store")
#             vector_store = existing_store
#         else:
#             # Ingest new documents
#             logger.info("Ingesting new documents...")
#             vector_store = ingestor.ingest_documents(document_paths)
        
#         # Get stats
#         stats = ingestor.get_vector_store_stats()
#         logger.info(f"Vector store stats: {stats}")
        
#         return vector_store
        
#     except Exception as e:
#         logger.error(f"Ingestion failed: {str(e)}")
#         raise

# def create_retrieval_pipeline(vector_store) -> HybridRetriever:
#     """Create retrieval pipeline"""
#     logger.info("Creating retrieval pipeline...")
    
#     retriever = create_retriever(vector_store)
#     logger.info("Retrieval pipeline created successfully")
    
#     return retriever

# def process_queries(retriever: HybridRetriever, queries: List[str]):
#     """Process queries through the retrieval pipeline"""
#     logger.info("Processing queries...")
    
#     results = {}
    
#     for query in queries:
#         logger.info(f"Processing query: '{query}'")
        
#         try:
#             # Retrieve relevant documents
#             result = retriever.retrieve(query, top_k=5, include_sources=True)
            
#             # Display results
#             print(f"\n{'='*60}")
#             print(f"Query: {query}")
#             print(f"Query Type: {result['query_type']}")
#             print(f"Documents Found: {result['retrieved_count']}")
#             print(f"\nAnswer:\n{result['answer']}")
            
#             if result['retrieved_documents']:
#                 print(f"\nTop 3 Sources:")
#                 for i, (doc, score) in enumerate(result['retrieved_documents'][:3]):
#                     print(f"\n{i+1}. Score: {score:.3f}")
#                     print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
#                     print(f"   Type: {doc.metadata.get('document_type', 'Unknown')}")
#                     print(f"   Preview: {doc.page_content[:200]}...")
            
#             results[query] = result
            
#             # Get retrieval stats
#             stats = retriever.get_retrieval_stats(query)
#             logger.info(f"Retrieval stats for '{query}': {stats}")
            
#         except Exception as e:
#             logger.error(f"Query processing failed for '{query}': {str(e)}")
#             results[query] = {"error": str(e)}
    
#     return results

# def main():
#     setup_environment()
    
#     document_paths = [
#         "./data/toan_cao_cap.pdf",
#         "./data/triet_hoc_mac_lenin.pdf",
#         "./data/triet_hoc_phuong_dong.txt",
#     ]
    
#     # Step 1: Chunk documents (optional - can be done separately)
#     chunks = chunk_documents(document_paths)
    
#     # Step 2: Ingest to Qdrant
#     logger.info("=== STEP 1: INGESTION ===")
#     vector_store = ingest_to_qdrant(document_paths)
    
#     # Step 3: Create retrieval pipeline
#     logger.info("\n=== STEP 2: RETRIEVAL PIPELINE ===")
#     retriever = create_retrieval_pipeline(vector_store)
    
#     # Step 4: Process queries
#     logger.info("\n=== STEP 3: QUERY PROCESSING ===")
    
#     # Example queries
#     queries = [
#         "Định lý Pythagoras là gì và công thức của nó?",
#         "Phân tích quan điểm duy vật biện chứng của Marx",
#         "So sánh tư tưởng của Plato và Aristotle",
#         "Cách tính đạo hàm của hàm số bậc 3?",
#         "Khái niệm 'tồn tại' trong triết học hiện sinh"
#     ]
    
#     results = process_queries(retriever, queries)
    
#     # Summary
#     logger.info("\n=== SUMMARY ===")
#     logger.info(f"Total documents processed: {len(document_paths)}")
#     logger.info(f"Total queries processed: {len(queries)}")
#     logger.info(f"Successful retrievals: {sum(1 for r in results.values() if 'error' not in r)}")
    
#     return results

# if __name__ == "__main__":
#     main()

import speech_recognition as sr

r = sr.Recognizer()
# with sr.Microphone() as source:
#     print("Say something!")
#     audio = r.listen(source)
with sr.AudioFile("voice.wav") as source:
    print("Loading soung")
    audio = r.listen(source)

try:
    text = r.recognize_google(audio)
    print(f"You said: {text}")
except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")