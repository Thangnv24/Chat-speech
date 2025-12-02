# main.py - ĐƠN GIẢN HƠN
from src.core.ingestor import DocumentIngestor
from src.core.retriever import create_retriever

def main():
    # Setup environment variables (trong .env file)
    # GEMINI_API_KEY=your_key_here
    # hoặc QWEN_API_KEY=your_key_here
    
    # Khởi tạo - không cần chỉ đường dẫn model
    ingestor = DocumentIngestor("./vector_store")
    vector_store = ingestor.ingest_documents(["data/toan.pdf", "data/triet.txt"])
    
    retriever = create_retriever(vector_store)
    
    # Tự động dùng Gemini/Qwen nếu có API key, không thì dùng Ollama local