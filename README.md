# FastAPI Todo Application

FastAPI-based Todo application with PostgreSQL database and future RAG capabilities.

## Features

- ✅ RESTful API for Todo CRUD operations
- ✅ PostgreSQL database with async SQLAlchemy
- ✅ Alembic migrations
- ✅ Docker & Docker Compose setup
- ✅ Pydantic v2 for validation
- 🚧 RAG (Retrieval-Augmented Generation) - In Development

## Tech Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL with asyncpg
- **ORM**: SQLAlchemy 2.0 (async)
- **Migrations**: Alembic
- **Validation**: Pydantic v2
- **Containerization**: Docker & Docker Compose

## Project Structure

```
mini_pj/
├── app/
│   ├── core/           # Core configurations
│   │   ├── config.py   # Settings management
│   │   └── database.py # Database setup
│   ├── models/         # SQLAlchemy models
│   │   └── todo.py
│   ├── schemas/        # Pydantic schemas
│   │   └── todo.py
│   ├── routers/        # API routes
│   │   └── todo.py
│   ├── service/        # RAG services (in development)
│   ├── config/         # RAG configs (in development)
│   └── main.py         # FastAPI application
├── alembic/            # Database migrations
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── requirements.txt    # FastAPI dependencies
└── requirements-rag.txt # RAG dependencies (optional)
```

## Quick Start

### 1. Clone and Setup

```bash
# Clone repository
git clone <your-repo>
cd mini_pj

# Copy environment file
cp .env.example .env
```

### 2. Run with Docker

```bash
# Start all services
make up
# or
docker compose up -d --build

# View logs
make logs
# or
docker compose logs -f
```

### 3. Access Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **PgAdmin**: http://localhost:8080
  - Email: admin@example.com
  - Password: admin

## Development Setup

### Local Development (without Docker)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Set environment variables
export POSTGRES_HOST=localhost
# ... other variables from .env

# Run migrations
alembic upgrade head

# Start server
uvicorn app.main:app --reload
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## API Endpoints

### Health Check
- `GET /` - Basic health check
- `GET /health` - Detailed health check

### Todos
- `POST /api/v1/todos` - Create todo
- `GET /api/v1/todos` - List todos (with pagination)
- `GET /api/v1/todos/{id}` - Get todo by ID
- `PATCH /api/v1/todos/{id}` - Update todo
- `DELETE /api/v1/todos/{id}` - Delete todo

## API Examples

### Create Todo
```bash
curl -X POST "http://localhost:8000/api/v1/todos" \
  -H "Content-Type: application/json" \
  -d '{"title": "Learn FastAPI", "completed": false}'
```

### List Todos
```bash
curl "http://localhost:8000/api/v1/todos?skip=0&limit=10"
```

### Update Todo
```bash
curl -X PATCH "http://localhost:8000/api/v1/todos/1" \
  -H "Content-Type: application/json" \
  -d '{"completed": true}'
```

## Environment Variables

See `.env.example` for all available configuration options.

Key variables:
- `PROJECT_NAME` - Application name
- `POSTGRES_HOST` - Database host
- `POSTGRES_DB` - Database name
- `POSTGRES_USER` - Database user
- `POSTGRES_PASSWORD` - Database password

## RAG Features (Coming Soon)

The project includes infrastructure for RAG capabilities:
- Document ingestion and chunking
- Vector store with ChromaDB
- LLM integration (Gemini, Qwen, Ollama)
- Semantic search and retrieval

To install RAG dependencies:
```bash
pip install -r requirements-rag.txt
# or
pip install -e ".[rag]"
```

## Makefile Commands

```bash
make up        # Start services
make down      # Stop services and remove volumes
make logs      # View logs
make build     # Rebuild containers
make shell     # Access API container shell
make migrate   # Create and apply migration
```

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=app tests/
```

## Code Quality

```bash
# Format code
black .
isort .

# Lint
ruff check .

# Type check
mypy app/
```

## License

MIT

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request
