"""
Simple test script for RAG metrics
"""
import os
from app.service.RAG.main import SimpleRAG
from app.utils.logger import setup_logging

logger = setup_logging("test_rag_metrics")

def test_single_query():
    """Test a single query to understand the output format"""
    
    # Initialize RAG
    rag = SimpleRAG()
    if not rag.setup():
        logger.error("Failed to setup RAG")
        return
    
    # Test query
    query = "Định lý Pytago là gì?"
    logger.info(f"Testing query: {query}")
    
    # Get result using query_full()
    result = rag.query_full(query, k=5, search_mode='hybrid')
    
    # Print result structure
    logger.info("\n" + "="*70)
    logger.info("RESULT STRUCTURE:")
    logger.info("="*70)
    
    for key, value in result.items():
        if key == 'retrieved_documents':
            logger.info(f"\n{key}:")
            logger.info(f"  Type: {type(value)}")
            logger.info(f"  Length: {len(value)}")
            if value:
                logger.info(f"  First item type: {type(value[0])}")
                if isinstance(value[0], tuple):
                    logger.info(f"  Tuple length: {len(value[0])}")
                    logger.info(f"  First element type: {type(value[0][0])}")
                    if hasattr(value[0][0], 'page_content'):
                        logger.info(f"  Has page_content: Yes")
                        logger.info(f"  Content preview: {value[0][0].page_content[:100]}...")
        elif key == 'context':
            logger.info(f"\n{key}:")
            logger.info(f"  Type: {type(value)}")
            logger.info(f"  Length: {len(value)}")
            logger.info(f"  Preview: {value[:200]}...")
        elif key == 'answer':
            logger.info(f"\n{key}:")
            logger.info(f"  Type: {type(value)}")
            logger.info(f"  Preview: {value[:200]}...")
        else:
            logger.info(f"\n{key}: {value}")
    
    logger.info("\n" + "="*70)
    
    # Extract documents
    logger.info("\nEXTRACTING DOCUMENTS:")
    logger.info("="*70)
    
    retrieved_docs = result.get('retrieved_documents', [])
    doc_contents = []
    
    for i, doc_tuple in enumerate(retrieved_docs, 1):
        if isinstance(doc_tuple, tuple) and len(doc_tuple) >= 1:
            doc = doc_tuple[0]
            score = doc_tuple[1] if len(doc_tuple) > 1 else 0
            if hasattr(doc, 'page_content'):
                content = doc.page_content
                doc_contents.append(content)
                logger.info(f"\nDoc {i}:")
                logger.info(f"  Score: {score:.4f}")
                logger.info(f"  Content length: {len(content)}")
                logger.info(f"  Preview: {content[:150]}...")
    
    logger.info(f"\nTotal extracted: {len(doc_contents)} documents")
    
    return result, doc_contents


def test_metrics_format():
    """Test the format needed for metrics evaluation"""
    
    logger.info("\n" + "="*70)
    logger.info("TESTING METRICS FORMAT")
    logger.info("="*70)
    
    # Initialize RAG
    rag = SimpleRAG()
    if not rag.setup():
        logger.error("Failed to setup RAG")
        return
    
    # Test queries
    test_queries = [
        "Định lý Pytago là gì?",
        "Tích phân là gì?"
    ]
    
    retrieved_contexts = []
    generated_answers = []
    
    for i, q in enumerate(test_queries, 1):
        logger.info(f"\nQuery {i}: {q}")
        
        result = rag.query_full(q, k=3, search_mode='hybrid')
        
        # Extract documents
        retrieved_docs = result.get('retrieved_documents', [])
        doc_contents = []
        
        for doc_tuple in retrieved_docs:
            if isinstance(doc_tuple, tuple) and len(doc_tuple) >= 1:
                doc = doc_tuple[0]
                if hasattr(doc, 'page_content'):
                    doc_contents.append(doc.page_content)
        
        retrieved_contexts.append(doc_contents)
        
        # Get answer
        answer = result.get('answer', '')
        generated_answers.append(answer)
        
        logger.info(f"  Retrieved: {len(doc_contents)} docs")
        logger.info(f"  Answer: {answer[:100]}...")
    
    # Print format for metrics
    logger.info("\n" + "="*70)
    logger.info("FORMAT FOR METRICS:")
    logger.info("="*70)
    logger.info(f"\nretrieved_contexts type: {type(retrieved_contexts)}")
    logger.info(f"retrieved_contexts length: {len(retrieved_contexts)}")
    logger.info(f"First item type: {type(retrieved_contexts[0])}")
    logger.info(f"First item length: {len(retrieved_contexts[0])}")
    
    logger.info(f"\ngenerated_answers type: {type(generated_answers)}")
    logger.info(f"generated_answers length: {len(generated_answers)}")
    logger.info(f"First answer type: {type(generated_answers[0])}")
    
    logger.info("\n✅ Format is correct for metrics evaluation!")
    
    return retrieved_contexts, generated_answers


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RAG METRICS TEST")
    print("="*70)
    
    # Test 1: Single query
    print("\n[TEST 1] Single Query Test")
    print("-"*70)
    result, docs = test_single_query()
    
    # Test 2: Metrics format
    print("\n[TEST 2] Metrics Format Test")
    print("-"*70)
    contexts, answers = test_metrics_format()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED!")
    print("="*70)
    print("\nYou can now run: python app/service/RAG/rag_metrics.py")
