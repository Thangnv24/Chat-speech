# Chat-speech

README tiếng Việt cho dự án Chat-speech — một hệ thống RAG (Retrieval-Augmented Generation) tích hợp ingest tài liệu, lưu vector, và phục vụ API trả lời/thuật toán chat.

## Tính năng (Features)
- Ingest tài liệu và tạo chunk (chấm nhỏ) phục vụ cho retrieval.
- Lưu trữ vector embedding vào vector store (cấu hình được: Chroma/FAISS/Pinecone...).
- RAG pipeline: lấy ngữ cảnh từ vector store và sinh câu trả lời bằng mô hình gen (ví dụ Google Generative API thông qua LangChain).
- API HTTP (FastAPI + Uvicorn) để phục vụ endpoint chat/truy vấn.
- Migration database với Alembic (cho phần metadata / user / logs nếu sử dụng DB).
- Hỗ trợ chạy bằng Docker Compose hoặc môi trường ảo Python local.
- Scripts tiện ích để load env, ingest dữ liệu, và chạy server.

## Tech stack (chi tiết)
(Dưới đây là danh sách các công nghệ/thư viện mà dự án sử dụng hoặc tương thích; chỉnh sửa .env / requirements theo repo nếu khác)
- Ngôn ngữ:
  - HTML cho frontend (nếu có giao diện đơn giản)
- Web / API:
  - FastAPI — web framework chính để expose API.
  - Uvicorn — ASGI server (phát triển / reload).
- Vector search / RAG:
  - LangChain — orchestration RAG, prompt management, retry logic.
  - Embeddings provider — Google Generative Embeddings / OpenAI / hoặc local (tùy cấu hình).
  - Vector stores (configurable):
    - Chroma (local) hoặc FAISS (local)
    - Pinecone, Milvus (dịch vụ) — nếu bạn muốn dùng cloud vector DB.
- Migrations & ORM:
  - SQLAlchemy — ORM (nếu dùng DB quan hệ)
  - Alembic — migration scripts (đã có lệnh `alembic upgrade head`)
- Container / Orchestration:
  - Docker, Docker Compose — để chạy DB/Redis/phiên bản service nếu repo có docker-compose.yml
- Các tiện ích / khác:
  - python-dotenv hoặc script PowerShell (`load_env.ps1`) để load biến môi trường.
  - Thư viện xử lý văn bản: nltk, sentencepiece, transformers,... (xem `requirements.txt` để biết chính xác).
  - Các package bổ sung: requests, httpx, tqdm, langchain, chromadb, faiss-cpu, alembic, uvicorn, fastapi, pydantic,...
- CI / DevOps:
  - (Có thể cấu hình GitHub Actions / CI để test và deploy)


## Cách chạy dự án (Quick start)
Dưới đây có cả bước chạy bằng Docker Compose và chạy local bằng venv. Điều chỉnh theo OS (Windows / macOS / Linux).

1) Bằng Docker (nếu repo chứa docker-compose.yml)
- Khởi động background:
```bash
docker-compose up -d
```
- Bắt đầu services (nếu đã có containers nhưng ở trạng thái stopped):
```bash
docker-compose start
```

2) Chạy local (Python venv)
- Tạo virtual environment:
```bash
python -m venv venv
```
- Kích hoạt virtualenv:
  - Trên macOS / Linux:
    ```bash
    source venv/bin/activate
    ```
  - Trên Windows (PowerShell):
    ```powershell
    .\venv\Scripts\Activate.ps1
    ```
  - Trên Windows (cmd):
    ```
    .\venv\Scripts\activate.bat
    ```
- Cài dependencies:
  - (sửa lỗi chính tả từ `uv pip` trong ví dụ)  
  ```bash
  pip install -r requirements.txt
  ```
- Load biến môi trường (nếu có script PowerShell cung cấp):
  - Trên Windows PowerShell:
    ```powershell
    .\load_env.ps1
    ```
  - Nếu bạn dùng `.env` file, có thể chạy:
    ```bash
    export $(cat .env | xargs)          # macOS / Linux (tùy shell)
    # hoặc dùng python-dotenv trong code để load tự động
    ```

3) Migration DB (nếu dùng Alembic)
```bash
alembic upgrade head
```

4) Ingest tài liệu (RAG)
- Chạy script chunk & ingest (đảm bảo biến môi trường cho embeddings/vector store đã được cấu hình):
```bash
python app/service/RAG/chunk_and_ingest.py
```

5) Chạy server dev
```bash
uvicorn app.main:app --reload
```
Sau đó truy cập API tại http://localhost:8000/docs (Để test swagger)
Truy cập vào http://localhost:8000/static/chat.html để test chức năng chat và speech
## Kết quả / Metrics (RAG EVALUATION)
Kết quả đánh giá RAG (k=10) thu được trong quá trình kiểm thử:

---------------

RAG EVALUATION (k=10)
--------------------------------------
Precision@10               | 0.85  
Recall@10                  | 0.72  
F1 Score                  | 0.78  
--------------------------------------
Context Precision         | 0.81  
Faithfulness              | 0.92  
Answer Relevance          | 0.88
