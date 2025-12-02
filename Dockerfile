# ----- builder -----
    FROM python:3.11-slim AS runtime
    WORKDIR /app
    ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
    COPY . .
    RUN pip install --upgrade pip
    # Cài từ pyproject nếu có, nếu thất bại thì fallback cài thủ công
    RUN pip install -e . || pip install fastapi "uvicorn[standard]" pydantic-settings sqlalchemy asyncpg alembic psycopg[binary] \
        loguru python-multipart python-jose[cryptography] passlib[bcrypt]
    EXPOSE 8000
    CMD ["uvicorn", "app.main:app", "--host","0.0.0.0", "--port","8000"]
    