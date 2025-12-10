import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.service.RAG.rag_pipeline import create_pipeline
from app.service.RAG.rag_metrics import create_metrics
from app.utils.logger import setup_logging

logger = setup_logging("rag_main")


def example_1_basic_ingestion():
    """Example 1: Basic document ingestion"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Document Ingestion")
    print("="*60 + "\n")
    
    # Create pipeline
    pipeline = create_pipeline(
        qdrant_url="http://localhost:6333",
        collection_name="math_philosophy"
    )
    
    # Ingest documents
    document_paths = [
        "data/doc_1.pdf",
        # Add more documents here
    ]
    
    try:
        stats = pipeline.ingest_documents(document_paths)
        
        print("✓ Ingestion completed!")
        print(f"  - Documents processed: {stats.get('documents_processed', 0)}")
        print(f"  - Total chunks: {stats.get('total_chunks', 0)}")
        print(f"  - Ingestion time: {stats.get('ingestion_time', 0):.2f}s")
        print(f"  - Document types: {stats.get('document_types', {})}")
        
    except Exception as e:
        print(f"Ingestion failed: {e}")
        return None
    
    return pipeline


def example_2_query_system():
    """Example 2: Query the RAG system"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Query RAG System")
    print("="*60 + "\n")
    
    # Create pipeline and load existing store
    pipeline = create_pipeline()
    
    if not pipeline.load_existing_store():
        print("No existing vector store found. Run example_1 first.")
        return
    
    # Initialize retriever
    pipeline.initialize_retriever(search_type="hybrid")
    
    # Example queries
    queries = [
        "Định lý Pythagore là gì?",
        "Giải thích khái niệm biện chứng",
        "Công thức tính diện tích hình tròn"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        try:
            result = pipeline.query(
                query=query,
                k=3,
                search_mode="hybrid",
                include_sources=True
            )
            
            print(f"✓ Query completed in {result.get('query_time', 0):.2f}s")
            print(f"  Query type: {result.get('query_type', 'unknown')}")
            print(f"  Documents retrieved: {result.get('num_retrieved', 0)}")
            print(f"\n💡 Answer:\n{result.get('answer', 'No answer generated')}")
            
            # Show sources
            if result.get('retrieved_documents'):
                print(f"\n📚 Sources:")
                for i, (doc, score) in enumerate(result['retrieved_documents'][:2], 1):
                    print(f"  {i}. Score: {score:.3f}")
                    print(f"     {doc.page_content[:150]}...")
            
        except Exception as e:
            print(f"Query failed: {e}")


def example_3_compare_search_modes():
    """Example 3: Compare different search modes"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Compare Search Modes")
    print("="*60 + "\n")
    
    pipeline = create_pipeline()
    
    if not pipeline.load_existing_store():
        print("No existing vector store found.")
        return
    
    pipeline.initialize_retriever()
    
    query = "Định lý Pythagore"
    search_modes = ["hybrid", "dense", "sparse"]
    
    print(f"Query: {query}\n")
    
    for mode in search_modes:
        print(f"\n🔍 Search mode: {mode.upper()}")
        print("-" * 40)
        
        try:
            result = pipeline.query(
                query=query,
                k=3,
                search_mode=mode,
                include_sources=False
            )
            
            print(f"  Time: {result.get('query_time', 0):.3f}s")
            print(f"  Retrieved: {result.get('num_retrieved', 0)} docs")
            print(f"  Answer length: {len(result.get('answer', '').split())} words")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")


def example_4_batch_processing():
    """Example 4: Batch query processing"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Query Processing")
    print("="*60 + "\n")
    
    pipeline = create_pipeline()
    
    if not pipeline.load_existing_store():
        print("❌ No existing vector store found.")
        return
    
    pipeline.initialize_retriever()
    
    # Batch queries
    queries = [
        "Định lý Pythagore",
        "Khái niệm biện chứng",
        "Công thức diện tích",
        "Tư tưởng triết học",
        "Phương trình bậc hai"
    ]
    
    print(f"Processing {len(queries)} queries in batch...\n")
    
    results = pipeline.batch_query(queries, k=3, search_mode="hybrid")
    
    # Summary
    successful = sum(1 for r in results if "error" not in r)
    total_time = sum(r.get("query_time", 0) for r in results)
    
    print(f"\n✓ Batch processing completed!")
    print(f"  - Successful: {successful}/{len(queries)}")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Avg time per query: {total_time/len(queries):.2f}s")


def example_5_evaluation():
    """Example 5: Evaluate RAG system"""
    print("\n" + "="*60)
    print("EXAMPLE 5: RAG System Evaluation")
    print("="*60 + "\n")
    
    pipeline = create_pipeline()
    
    if not pipeline.load_existing_store():
        print("❌ No existing vector store found.")
        return
    
    pipeline.initialize_retriever()
    
    # Create metrics evaluator
    metrics = create_metrics()
    
    # Test queries with ground truth (example)
    test_cases = [
        {
            "query": "Định lý Pythagore",
            "relevant_docs": ["doc_1_chunk_5", "doc_1_chunk_6"],  # Example IDs
        },
        {
            "query": "Khái niệm biện chứng",
            "relevant_docs": ["doc_2_chunk_3", "doc_2_chunk_4"],
        }
    ]
    
    print("Running evaluation on test queries...\n")
    
    query_times = []
    answers = []
    queries = []
    contexts = []
    
    for test in test_cases:
        query = test["query"]
        
        result = pipeline.query(query, k=5, include_sources=True)
        
        query_times.append(result.get("query_time", 0))
        answers.append(result.get("answer", ""))
        queries.append(query)
        contexts.append(result.get("context", ""))
        
        # Calculate precision/recall (simplified)
        retrieved_ids = [f"doc_{i}" for i in range(result.get("num_retrieved", 0))]
        precision = metrics.precision_at_k(retrieved_ids, test["relevant_docs"], k=5)
        
        print(f"Query: {query}")
        print(f"  Precision@5: {precision:.3f}")
        print(f"  Query time: {result.get('query_time', 0):.3f}s")
    
    # Latency metrics
    latency = metrics.latency_metrics(query_times)
    print(f"\n📊 Latency Metrics:")
    print(f"  Mean: {latency.get('mean', 0):.3f}s")
    print(f"  Median: {latency.get('median', 0):.3f}s")
    print(f"  P95: {latency.get('p95', 0):.3f}s")
    
    # Generation metrics
    gen_metrics = metrics.evaluate_generation(answers, queries, contexts)
    print(f"\n📊 Generation Metrics:")
    print(f"  Avg Relevance: {gen_metrics['metrics']['avg_relevance']:.3f}")
    print(f"  Avg Faithfulness: {gen_metrics['metrics']['avg_faithfulness']:.3f}")


def example_6_health_check():
    """Example 6: System health check"""
    print("\n" + "="*60)
    print("EXAMPLE 6: System Health Check")
    print("="*60 + "\n")
    
    pipeline = create_pipeline()
    
    # Health check
    health = pipeline.health_check()
    
    print(f"System Status: {health['status'].upper()}")
    print(f"\nComponents:")
    for component, status in health['components'].items():
        icon = "✓" if "error" not in status.lower() else "❌"
        print(f"  {icon} {component}: {status}")
    
    # Get stats if available
    if pipeline.load_existing_store():
        stats = pipeline.get_stats()
        print(f"\n📊 Vector Store Stats:")
        print(f"  Collection: {stats.get('collection_name', 'N/A')}")
        print(f"  Vectors: {stats.get('vector_count', 0)}")
        print(f"  Total chunks: {stats.get('total_chunks', 0)}")
        print(f"  Document types: {stats.get('document_types', {})}")


def main():
    """Main function - Run all examples"""
    print("\n" + "="*70)
    print(" "*20 + "RAG SYSTEM DEMO")
    print("="*70)
    
    print("\nAvailable Examples:")
    print("1. Basic Document Ingestion")
    print("2. Query RAG System")
    print("3. Compare Search Modes")
    print("4. Batch Query Processing")
    print("5. System Evaluation")
    print("6. Health Check")
    print("7. Run All Examples")
    
    choice = input("\nSelect example (1-7): ").strip()
    
    if choice == "1":
        example_1_basic_ingestion()
    elif choice == "2":
        example_2_query_system()
    elif choice == "3":
        example_3_compare_search_modes()
    elif choice == "4":
        example_4_batch_processing()
    elif choice == "5":
        example_5_evaluation()
    elif choice == "6":
        example_6_health_check()
    elif choice == "7":
        # Run all examples
        pipeline = example_1_basic_ingestion()
        if pipeline:
            example_2_query_system()
            example_3_compare_search_modes()
            example_4_batch_processing()
            example_5_evaluation()
            example_6_health_check()
    else:
        print("Invalid choice")
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
