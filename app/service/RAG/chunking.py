import re
import numpy as np
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_core.documents import Document
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import fitz  
from app.utils.logger import setup_logging

"""
Chunk theo context, semantic, rule và recursive
Cho toán học:
    + rule - để bảo toàn định lý/ chứng minh
    + struture_preserving - để giữ công thức
    + Context - để nhận diện cấu trúc
Cho triết học:
    + Semantic clustering - nhóm chủ đề
    + Recursive - xử lý văn bản dài
    + Rule-based - giữ luận điểm
"""

logger = setup_logging("chunking")

class AdvancedTextChunker:
    """Advanced chunker specialized for Mathematics and Philosophy texts"""
    
    def __init__(self):
        self.logger = logger
        # self.nlp = self._load_spacy_model()
        
        # Special patterns for Math and Philosophy
        self.math_patterns = {
            'theorem': r'Định lý|Định lí|Theorem|Theorem\s*[\d\.]+',
            'proof': r'Chứng minh|Proof|Chứng minh rằng',
            'definition': r'Định nghĩa|Definition|Định nghĩa\s*[\d\.]+',
            'equation': r'\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)',
            'formula': r'[A-Za-z]+\([^)]+\)|=.*?=|→|⇔|⇒|∀|∃|∈|⊂|∪|∩',
            'example': r'Ví dụ|Example|Ví dụ\s*[\d\.]+'
        }
        
        self.philosophy_patterns = {
            'concept': r'Khái niệm|Concept|Quan niệm|Tư tưởng',
            'argument': r'Luận điểm|Argument|Luận chứng|Luận cứ',
            'critique': r'Phê phán|Critique|Phản biện|Bác bỏ',
            'doctrine': r'Học thuyết|Doctrine|Thuyết|Chủ nghĩa',
            'dialectic': r'Biện chứng|Dialectic|Phép biện chứng'
        }
    
    # Use spacy model for tokenizing
    # def _load_spacy_model(self):
    #     try:
    #         return spacy.load("vi_core_news_lg")
    #     except OSError:
    #         try:
    #             return spacy.load("en_core_web_sm")
    #         except OSError:
    #             logger.warning("spaCy model not found, using basic tokenization")
    #             return None
    
    # Detect if text is Math, Philosophy or Mixed
    """
    Tìm tất cả pattern trong tài liệu, bỏ qua chữ in hoa hay thường
    """
    def detect_document_type(self, text: str) -> str:
        math_score = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                        for pattern in self.math_patterns.values())
        philo_score = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                             for pattern in self.philosophy_patterns.values())
        
        total_score = math_score + philo_score
        
        if total_score == 0:
            return "general"
        
        math_ratio = math_score / total_score
        
        if math_ratio > 0.7:
            return "mathematics"
        elif math_ratio < 0.3:
            return "philosophy"
        else:
            return "mixed"
    
    # Extract mathematical structures from text (theorem, proof,equation)
    def extract_mathematical_structures(self, text: str) -> List[Dict[str, Any]]:
        structures = []
        # Theorems
        theorems = re.finditer(self.math_patterns['theorem'], text, re.IGNORECASE)
        for match in theorems:
            structures.append({
                'type': 'theorem',
                'content': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        # Proofs
        proofs = re.finditer(self.math_patterns['proof'], text, re.IGNORECASE)
        for match in proofs:
            structures.append({
                'type': 'proof',
                'content': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        # equations
        equations = re.finditer(self.math_patterns['equation'], text, re.DOTALL)
        for match in equations:
            structures.append({
                'type': 'equation',
                'content': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        return structures
    
    # Extract philo structures from text (concept, argument, )
    def extract_philosophical_structures(self, text: str) -> List[Dict[str, Any]]:
        structures = []
        # concepts
        concepts = re.finditer(self.philosophy_patterns['concept'], text, re.IGNORECASE)
        for match in concepts:
            structures.append({
                'type': 'concept',
                'content': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        # arguments
        arguments = re.finditer(self.philosophy_patterns['argument'], text, re.IGNORECASE)
        for match in arguments:
            structures.append({
                'type': 'argument',
                'content': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        return structures
    
    # Semantic and tf-idf
    def semantic_chunking(self, texts: List[str], num_clusters: int = 10) -> List[List[str]]:
        if not texts:
            return []
        
        """
        tf-idf
        Tần suất thuật ngữ - tần suất tài liệu nghịch đảo
        Càng cao -> càng quan trọng
        Sử dụng 1000 từ quan trọng nhất
        """
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  
            min_df=2,
            max_df=0.8
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # k-mean
            kmeans = KMeans(n_clusters=min(num_clusters, len(texts)), random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            # group texts =
            clustered_chunks = [[] for _ in range(max(clusters) + 1)]
            for i, cluster_id in enumerate(clusters):
                clustered_chunks[cluster_id].append(texts[i])
            
            logger.info(f"Semantic chunking created {len(clustered_chunks)} clusters")
            return clustered_chunks
            
        except Exception as e:
            logger.error(f"Semantic chunking failed: {str(e)}", {"error": str(e)})
            return [texts]  
    
    # Document type and context
    def context_aware_chunking(self, text: str, doc_type: str, 
                             chunk_size: int = 1000, 
                             chunk_overlap: int = 200) -> List[Document]:
        
        logger.info(f"Starting context-aware chunking for {doc_type} document")
        
        if doc_type == "mathematics":
            return self._chunk_mathematics(text, chunk_size, chunk_overlap)
        elif doc_type == "philosophy":
            return self._chunk_philosophy(text, chunk_size, chunk_overlap)
        elif doc_type == "mixed":
            return self._chunk_mixed(text, chunk_size, chunk_overlap)
        else:
            return self._chunk_general(text, chunk_size, chunk_overlap)

    # Math    
    def _chunk_mathematics(self, text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
        # First, extract important mathematical structures
        math_structures = self.extract_mathematical_structures(text)
        
        # Split by mathematical sections
        sections = self._split_by_mathematical_sections(text)
        
        chunks = []
        for section in sections:
            if len(section) > chunk_size:
                # Use recursive splitting for large sections, but preserve math content
                sub_chunks = self._split_preserving_math(section, chunk_size, chunk_overlap)
                chunks.extend(sub_chunks)
            else:
                chunks.append(section)
        
        # Create documents with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            # Find mathematical structures in this chunk
            chunk_structures = [s for s in math_structures 
                              if s['start'] >= text.find(chunk) and s['end'] <= text.find(chunk) + len(chunk)]
            
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "document_type": "mathematics",
                    "math_structures": [s['type'] for s in chunk_structures],
                    "contains_equations": any(s['type'] == 'equation' for s in chunk_structures),
                    "chunk_size": len(chunk)
                }
            )
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} chunks for mathematics document")
        return documents
    
    def _chunk_philosophy(self, text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Specialized chunking for philosophical texts"""
        
        # Extract philosophical structures
        philosophy_structures = self.extract_philosophical_structures(text)
        
        # Split by paragraphs and philosophical concepts
        paragraphs = self._split_by_philosophical_paragraphs(text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Create documents with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            # Find philosophical structures in this chunk
            chunk_structures = [s for s in philosophy_structures 
                              if s['start'] >= text.find(chunk) and s['end'] <= text.find(chunk) + len(chunk)]
            
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "document_type": "philosophy",
                    "philosophy_structures": [s['type'] for s in chunk_structures],
                    "chunk_size": len(chunk)
                }
            )
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} chunks for philosophy document")
        return documents
    
    def _chunk_mixed(self, text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Chunking for mixed Mathematics and Philosophy texts"""
        
        math_structures = self.extract_mathematical_structures(text)
        philosophy_structures = self.extract_philosophical_structures(text)
        
        # recursive
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
        )
        
        chunks = text_splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_math = [s for s in math_structures 
                         if s['start'] >= text.find(chunk) and s['end'] <= text.find(chunk) + len(chunk)]
            chunk_philo = [s for s in philosophy_structures 
                          if s['start'] >= text.find(chunk) and s['end'] <= text.find(chunk) + len(chunk)]
            
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "document_type": "mixed",
                    "math_structures": [s['type'] for s in chunk_math],
                    "philosophy_structures": [s['type'] for s in chunk_philo],
                    "contains_equations": any(s['type'] == 'equation' for s in chunk_math),
                    "chunk_size": len(chunk)
                }
            )
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} chunks for mixed document")
        return documents
    
    def _chunk_general(self, text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
        """General chunking for other document types"""
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        chunks = text_splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "document_type": "general",
                    "chunk_size": len(chunk)
                }
            )
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} chunks for general document")
        return documents
    
    def _split_by_mathematical_sections(self, text: str) -> List[str]:
        """Split mathematical text by sections (theorems, proofs, definitions, etc.)"""
        
        # Pattern to match mathematical sections
        section_patterns = [
            self.math_patterns['theorem'],
            self.math_patterns['proof'], 
            self.math_patterns['definition'],
            self.math_patterns['example']
        ]
        
        combined_pattern = '|'.join(f'({pattern})' for pattern in section_patterns)
        
        sections = []
        last_end = 0
        
        for match in re.finditer(combined_pattern, text, re.IGNORECASE | re.DOTALL):
            if match.start() > last_end:
                sections.append(text[last_end:match.start()])
            sections.append(match.group())
            last_end = match.end()
        
        if last_end < len(text):
            sections.append(text[last_end:])
        
        return [s.strip() for s in sections if s.strip()]
    
    def _split_by_philosophical_paragraphs(self, text: str) -> List[str]:
        """Split philosophical text by paragraphs and concepts"""
        
        # Split by multiple newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        refined_paragraphs = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Further split by philosophical concepts if paragraph is too long
            if len(paragraph) > 500:
                concept_split = re.split(
                    '|'.join(self.philosophy_patterns.values()),
                    paragraph,
                    flags=re.IGNORECASE
                )
                refined_paragraphs.extend([p.strip() for p in concept_split if p.strip()])
            else:
                refined_paragraphs.append(paragraph)
        
        return refined_paragraphs
    
    def _split_preserving_math(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text while preserving mathematical content"""
        
        # First, protect equations and formulas
        protected_text = text
        equation_placeholders = {}
        
        # Replace equations with placeholders
        equations = re.finditer(self.math_patterns['equation'], text, re.DOTALL)
        for i, match in enumerate(equations):
            placeholder = f"__EQUATION_{i}__"
            equation_placeholders[placeholder] = match.group()
            protected_text = protected_text.replace(match.group(), placeholder)
        
        # Now split the protected text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        chunks = text_splitter.split_text(protected_text)
        
        # Restore equations in chunks
        final_chunks = []
        for chunk in chunks:
            for placeholder, equation in equation_placeholders.items():
                chunk = chunk.replace(placeholder, equation)
            final_chunks.append(chunk)
        
        return final_chunks
    
    def process_pdf(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                full_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"
                # print(f"Page {page_num+1} length: {len(text)}")
                # print(f"First 200 chars: {text[:200]}")
            
            doc.close()
            
            # Detect document type
            doc_type = self.detect_document_type(full_text)
            logger.info(f"Detected document type: {doc_type}")
            
            # Perform chunking
            chunks = self.context_aware_chunking(full_text, doc_type, chunk_size, chunk_overlap)
            
            logger.info(f"Successfully processed PDF into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}", {"file": pdf_path, "error": str(e)})
            return []
    
    def process_text_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]: 
        logger.info(f"Processing text file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Detect document type
            doc_type = self.detect_document_type(text)
            logger.info(f"Detected document type: {doc_type}")
            
            # Perform chunking
            chunks = self.context_aware_chunking(text, doc_type, chunk_size, chunk_overlap)
            
            logger.info(f"Successfully processed text file into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Text file processing failed: {str(e)}", {"file": file_path, "error": str(e)})
            return []

chunker = AdvancedTextChunker()

def get_chunker() -> AdvancedTextChunker:
    return chunker