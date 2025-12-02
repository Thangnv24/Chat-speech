.PHONY: up down logs build shell test fmt lint migrate

up:
\tdocker compose up -d --build

down:
\tdocker compose down -v

logs:
\tdocker compose logs -f

build:
\tdocker compose build --no-cache

shell:
\tdocker compose exec api /bin/sh

test:
\tpytest -q

fmt:
\tblack . && isort . && ruff --fix .

migrate:
\tdocker compose exec api alembic revision -m "auto" --autogenerate && docker compose exec api alembic upgrade head
