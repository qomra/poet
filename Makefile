# ============================================================================
# الشاعر — dev commands
# ============================================================================

export UV_PROJECT_ENVIRONMENT := $(HOME)/.venvs/alshaer
export PATH := $(HOME)/.local/bin:$(PATH)

.DEFAULT_GOAL := help

.PHONY: help up down ps logs install migrate etl-ashaar etl-all build

help:
	@echo ""
	@echo "  make up           Start Postgres + Qdrant"
	@echo "  make down         Stop and remove containers"
	@echo "  make ps           Show container status"
	@echo "  make logs         Tail container logs"
	@echo "  make install      Install all packages into venv"
	@echo "  make migrate      Run Alembic migrations"
	@echo "  make etl-dwianai  Run dwianai ETL pipeline"
	@echo "  make etl-ashaar   Run ashaar ETL pipeline"
	@echo ""

# ── Infrastructure ────────────────────────────────────────────────────────────

build: ## Build all Docker images
	docker compose -f infra/docker-compose.yml build

up: ## Start all services (Postgres, Qdrant, tools)
	docker compose -f infra/docker-compose.yml up -d
	@echo ""
	@echo "✓  Postgres          →  localhost:5433"
	@echo "✓  Qdrant            →  http://localhost:6333"
	@echo "✓  Qafiya Annotator  →  http://localhost:8501"

down:
	docker compose -f infra/docker-compose.yml down

ps:
	docker compose -f infra/docker-compose.yml ps

logs:
	docker compose -f infra/docker-compose.yml logs -f

# ── Python environment ─────────────────────────────────────────────────────────

install:
	uv sync

# ── Database ───────────────────────────────────────────────────────────────────

migrate:
	uv run --package db alembic -c packages/db/alembic.ini upgrade head

# ── ETL ───────────────────────────────────────────────────────────────────────

etl-ashaar:
	ALSHAER_ROOT=$(PWD) uv run etl ashaar

etl-all:
	ALSHAER_ROOT=$(PWD) uv run etl all
