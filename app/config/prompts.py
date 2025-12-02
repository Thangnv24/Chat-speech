from langchain.prompts import PromptTemplate

MATH_ANSWER_PROMPT = PromptTemplate(
    template="""Bạn là trợ lý chuyên gia Toán học. Hãy sử dụng thông tin dưới đây để trả lời câu hỏi.

Ngữ cảnh: {context}

Câu hỏi: {question}

Hướng dẫn:
- Trả lời chính xác dựa trên thông tin được cung cấp
- Giải thích rõ ràng các bước lập luận
- Nếu có công thức toán học, trình bày đầy đủ
- Nếu không đủ thông tin, hãy nói rõ

Câu trả lời:""",
    input_variables=["context", "question"]
)

MATH_SUMMARY_PROMPT = PromptTemplate(
    template="""Tóm tắt ngắn gọn nội dung toán học sau, tập trung vào định lý, công thức và phương pháp:

{text}

Tóm tắt:""",
    input_variables=["text"]
)

# === PROMPTS FOR PHILOSOPHY ===
PHILOSOPHY_ANSWER_PROMPT = PromptTemplate(
    template="""Bạn là trợ lý chuyên gia Triết học. Hãy sử dụng thông tin dưới đây để trả lời câu hỏi.

Ngữ cảnh: {context}

Câu hỏi: {question}

Hướng dẫn:
- Phân tích sâu sắc vấn đề triết học
- Trình bày các quan điểm khác nhau nếu có
- Liên hệ với các khái niệm triết học liên quan
- Giữ tính khách quan trong phân tích

Câu trả lời:""",
    input_variables=["context", "question"]
)

PHILOSOPHY_SUMMARY_PROMPT = PromptTemplate(
    template="""Tóm tắt các luận điểm triết học chính trong văn bản sau:

{text}

Tóm tắt:""",
    input_variables=["text"]
)


GENERAL_ANSWER_PROMPT = PromptTemplate(
    template="""Sử dụng thông tin sau để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết thay vì bịa đáp án.

Ngữ cảnh: {context}

Câu hỏi: {question}

Câu trả lời:""",
    input_variables=["context", "question"]
)

QUERY_EXPANSION_PROMPT = PromptTemplate(
    template="""Mở rộng và làm rõ câu hỏi sau để tìm kiếm thông tin hiệu quả hơn:

Câu hỏi gốc: {query}

Câu hỏi mở rộng (trả về 2-3 biến thể):""",
    input_variables=["query"]
)

# Export all prompts
PROMPT_MAP = {
    "mathematics": {
        "answer": MATH_ANSWER_PROMPT,
        "summary": MATH_SUMMARY_PROMPT
    },
    "philosophy": {
        "answer": PHILOSOPHY_ANSWER_PROMPT,
        "summary": PHILOSOPHY_SUMMARY_PROMPT
    },
    "general": {
        "answer": GENERAL_ANSWER_PROMPT,
        "expansion": QUERY_EXPANSION_PROMPT
    }
}