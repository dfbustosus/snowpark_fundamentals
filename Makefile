.PHONY: help install install-dev install-notebooks install-all \
       format lint typecheck security test test-cov \
       check ci clean pre-commit pre-commit-install notebook \
       dbt-setup dbt-run dbt-test dbt-docs dbt-clean

# ──────────────────────────────────────────────
# Default
# ──────────────────────────────────────────────

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

install: ## Install production dependencies
	pip install -e .

install-dev: ## Install dev dependencies (tests, lint, types, security)
	pip install -e ".[dev]"

install-notebooks: ## Install notebook dependencies (jupyter, matplotlib)
	pip install -e ".[notebooks]"

install-all: ## Install everything (production + dev + notebooks)
	pip install -e ".[dev,notebooks]"

pre-commit-install: ## Install pre-commit hooks into .git/hooks
	pre-commit install

# ──────────────────────────────────────────────
# Code Quality
# ──────────────────────────────────────────────

format: ## Auto-format code with black + isort
	black src/ tests/
	isort src/ tests/

lint: ## Run flake8 linter
	flake8 src/ tests/

typecheck: ## Run mypy static type checker
	mypy src/

# ──────────────────────────────────────────────
# Security
# ──────────────────────────────────────────────

security: ## Run bandit security scan
	bandit -r src/ -s B101,B105,B608

# ──────────────────────────────────────────────
# Testing
# ──────────────────────────────────────────────

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=snowpark_fundamentals --cov-report=term-missing

# ──────────────────────────────────────────────
# Pre-commit
# ──────────────────────────────────────────────

pre-commit: ## Run all pre-commit hooks on all files
	pre-commit run --all-files

# ──────────────────────────────────────────────
# Composite
# ──────────────────────────────────────────────

check: format lint typecheck security ## Run all quality checks (format + lint + typecheck + security)

ci: lint typecheck security test-cov ## Run full CI pipeline locally (lint + typecheck + security + tests)

# ──────────────────────────────────────────────
# Notebooks
# ──────────────────────────────────────────────

notebook: ## Launch Jupyter with the tutorial notebooks
	jupyter notebook notebooks/

# ──────────────────────────────────────────────
# dbt
# ──────────────────────────────────────────────

dbt-setup: ## Install dbt deps (dbt-snowflake already in main venv)
	cd dbt_feature_store && dbt deps

dbt-run: ## Run dbt models
	cd dbt_feature_store && dbt run --profiles-dir .

dbt-test: ## Run dbt tests
	cd dbt_feature_store && dbt test --profiles-dir .

dbt-docs: ## Generate and serve dbt docs
	cd dbt_feature_store && dbt docs generate --profiles-dir .
	cd dbt_feature_store && dbt docs serve --profiles-dir .

dbt-clean: ## Clean dbt artifacts
	rm -rf dbt_feature_store/target dbt_feature_store/dbt_packages dbt_feature_store/logs

# ──────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────

clean: ## Remove build artifacts, caches, and temp files
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -f bandit-report.json .coverage
