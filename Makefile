# ============================================================
# Cerebrum — Developer Makefile
# ============================================================

.PHONY: help up down dev-setup lint test test-unit test-int test-cov test-load \
        migrate seed clean logs ps build push docs

DOCKER_COMPOSE = docker compose
DOCKER_COMPOSE_DEV = docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml
DOCKER_COMPOSE_TEST = docker compose -f docker/docker-compose.test.yml

# Colors
GREEN  = \033[0;32m
YELLOW = \033[0;33m
NC     = \033[0m

help: ## Show this help message
	@echo "$(GREEN)Cerebrum — Developer Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

# ============================================================
# Docker
# ============================================================
up: ## Start the full stack (production-like)
	@echo "$(GREEN)Starting Cerebrum stack...$(NC)"
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml up -d
	@echo "$(GREEN)✓ Stack running at http://localhost:3000$(NC)"

down: ## Stop all containers
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml down

dev: ## Start in development mode (with hot reload)
	@echo "$(GREEN)Starting Cerebrum in development mode...$(NC)"
	$(DOCKER_COMPOSE_DEV) up

dev-down: ## Stop development containers
	$(DOCKER_COMPOSE_DEV) down

logs: ## Tail logs from all containers
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml logs -f

ps: ## Show running containers
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml ps

build: ## Build all Docker images
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml build

push: ## Push images to registry
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml push

# ============================================================
# Development Setup
# ============================================================
dev-setup: ## Setup full development environment
	@echo "$(GREEN)Setting up development environment...$(NC)"
	@which python3.13 || (echo "Python 3.13 required" && exit 1)
	python3.13 -m pip install --upgrade pip poetry
	poetry install
	cd apps/web && npm install
	pre-commit install
	cp .env.example .env
	@echo "$(GREEN)✓ Development environment ready$(NC)"
	@echo "$(YELLOW)  Next: Update .env with your API keys, then run: make dev$(NC)"

# ============================================================
# Code Quality
# ============================================================
lint: ## Run all linters
	@echo "$(GREEN)Running linters...$(NC)"
	ruff check . --fix
	ruff format .
	mypy apps/api agents services core models --ignore-missing-imports
	@echo "$(GREEN)✓ Linting complete$(NC)"

lint-check: ## Check linting without auto-fix (for CI)
	ruff check .
	ruff format --check .
	mypy apps/api agents services core models --ignore-missing-imports

format: ## Format code
	ruff format .

# ============================================================
# Testing
# ============================================================
test: ## Run all tests
	@echo "$(GREEN)Running all tests...$(NC)"
	pytest -v

test-unit: ## Run unit tests only (no Docker required)
	pytest -v -m "unit" tests/unit/

test-int: ## Run integration tests (requires Docker)
	@echo "$(YELLOW)Starting test services...$(NC)"
	$(DOCKER_COMPOSE_TEST) up -d
	pytest -v -m "integration" tests/integration/
	$(DOCKER_COMPOSE_TEST) down

test-cov: ## Run tests with coverage report
	pytest --cov=. --cov-report=term-missing --cov-report=html --cov-fail-under=80
	@echo "$(GREEN)Coverage report: htmlcov/index.html$(NC)"

test-e2e: ## Run end-to-end tests (requires running stack)
	pytest -v tests/e2e/

test-load: ## Run load tests
	locust -f tests/load/locustfile.py --headless -u 100 -r 10 --run-time 60s

test-agent: ## Run agent-specific tests
	pytest -v -m "agent" tests/unit/agents/

test-ml: ## Run ML pipeline tests
	pytest -v -m "ml" tests/unit/ml/

# ============================================================
# Database
# ============================================================
migrate: ## Run database migrations
	cd apps/api && alembic upgrade head

migrate-create: ## Create a new migration (usage: make migrate-create MSG="description")
	cd apps/api && alembic revision --autogenerate -m "$(MSG)"

migrate-down: ## Rollback one migration
	cd apps/api && alembic downgrade -1

seed: ## Seed database with sample data
	cd apps/api && python scripts/seed.py

# ============================================================
# Documentation
# ============================================================
docs: ## Build documentation
	mkdocs build

docs-serve: ## Serve documentation locally
	mkdocs serve

docs-deploy: ## Deploy documentation to GitHub Pages
	mkdocs gh-deploy

# ============================================================
# Cleanup
# ============================================================
clean: ## Remove generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ site/
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-docker: ## Remove all Docker containers and volumes (DANGEROUS)
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml down -v --remove-orphans
	docker system prune -f

# ============================================================
# Security
# ============================================================
security-scan: ## Run security scans
	trivy fs . --security-checks vuln,secret
	bandit -r apps/api agents services core -ll

audit: ## Audit Python dependencies
	pip-audit

# ============================================================
# Benchmarks
# ============================================================
benchmark: ## Run performance benchmarks
	pytest benchmarks/ -v --benchmark-only

# ============================================================
# CI helpers
# ============================================================
ci-lint: lint-check ## Lint check for CI
ci-test: test-cov ## Test with coverage for CI
ci-security: security-scan ## Security scan for CI
ci-build: build ## Docker build for CI

.DEFAULT_GOAL := help
