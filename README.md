# Snowpark ML Fundamentals

[![Python Tests](https://github.com/dfbustosus/snowpark_fundamentals/actions/workflows/python-tests.yml/badge.svg)](https://github.com/dfbustosus/snowpark_fundamentals/actions/workflows/python-tests.yml)
[![Lint](https://github.com/dfbustosus/snowpark_fundamentals/actions/workflows/python-lint.yml/badge.svg)](https://github.com/dfbustosus/snowpark_fundamentals/actions/workflows/python-lint.yml)
[![Type Check](https://github.com/dfbustosus/snowpark_fundamentals/actions/workflows/python-typecheck.yml/badge.svg)](https://github.com/dfbustosus/snowpark_fundamentals/actions/workflows/python-typecheck.yml)
[![Security Scan](https://github.com/dfbustosus/snowpark_fundamentals/actions/workflows/security-scan.yml/badge.svg)](https://github.com/dfbustosus/snowpark_fundamentals/actions/workflows/security-scan.yml)
[![CodeQL](https://github.com/dfbustosus/snowpark_fundamentals/actions/workflows/codeql.yml/badge.svg)](https://github.com/dfbustosus/snowpark_fundamentals/actions/workflows/codeql.yml)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/imports-isort-1674b1.svg)](https://pycqa.github.io/isort/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
[![Coverage](https://img.shields.io/badge/coverage-%E2%89%A570%25-brightgreen.svg)](pyproject.toml)

**Lunch & Learn Series: Snowpark ML for Python**

A production-grade, modular tutorial covering Snowpark ML fundamentals,
feature engineering, and the Snowflake Feature Store.

---

## Week 1: Snowpark ML Fundamentals (1 Hour)

| Time | Module | Notebook | Topics |
|------|--------|----------|--------|
| 0-5 min | **Presentation** | `00_presentation` | Overview, objectives, architecture, environment check |
| 5-15 min | **DataFrames** | `01_snowpark_dataframes` | Connect, load data, select, filter, aggregate, join, lazy evaluation |
| 15-25 min | **Preprocessing** | `02_data_preprocessing` | StandardScaler, OneHotEncoder, OrdinalEncoder, feature engineering |
| 25-40 min | **Model Training** | `03_model_training` | XGBoost, RandomForest, LogisticRegression, metrics, feature importance |
| 40-50 min | **Model Registry** | `04_model_registry` | Register, version, load, and score with the Snowflake Model Registry |
| 50-60 min | **End-to-End** | `05_end_to_end_pipeline` | Pipeline API, Snowpark vs Stored Procedures, architecture recap |

## Week 2: Feature Engineering in Snowflake (1 Hour)

| Time | Module | Resource | Topics |
|------|--------|----------|--------|
| 0-10 min | **Concepts** | `06_feature_store_concepts` | What is a Feature Store, architecture, synthetic temporal data |
| 10-20 min | **SQL Engineering** | `07_feature_engineering_sql` | Deduplication, time-windowed aggregations, ratios, bucketing |
| 20-30 min | **Feature Views** | `08_feature_views` | Entity, managed vs external FeatureView, registration |
| 30-40 min | **Training Sets** | `09_training_sets` | Point-in-time joins, `generate_training_set()`, inference retrieval |
| 40-60 min | **dbt Feature Store** | `dbt_feature_store/README.md` | Full dbt command reference, model walkthrough, macros, how to build your own features |

## Project Structure

```
snowpark_fundamentals/
├── src/snowpark_fundamentals/       # Modular Python package
│   ├── config.py                    # Connection configuration (12-factor)
│   ├── session.py                   # Session factory
│   ├── data/
│   │   ├── loader.py                # DataFrame loading utilities
│   │   └── sample_data.py           # Synthetic dataset generation
│   ├── preprocessing/
│   │   ├── transformers.py          # Snowpark ML scalers & encoders
│   │   └── feature_engineering.py   # Derived & interaction features
│   ├── modeling/
│   │   ├── trainer.py               # Model training (XGB, RF, LR)
│   │   ├── evaluation.py            # Metrics & feature importance
│   │   └── pipeline.py              # ML Pipeline construction
│   ├── registry/
│   │   └── model_registry.py        # Snowflake Model Registry ops
│   └── feature_store/               # Week 2: Feature Store operations
│       ├── entities.py              # Entity setup & registration
│       ├── feature_data.py          # Synthetic temporal data generation
│       ├── feature_views.py         # Managed & external FeatureViews
│       └── training_sets.py         # Training set & feature retrieval
├── notebooks/                       # Tutorial Jupyter notebooks (0-11)
├── tests/                           # Unit tests with mocking
├── dbt_feature_store/               # dbt project for feature engineering
│   ├── models/                      # staging → features → marts
│   ├── macros/                      # Reusable SQL helpers
│   └── tests/                       # Data quality tests
├── .github/workflows/               # CI/CD (lint, test, type-check, security)
├── pyproject.toml                   # Project config & tool settings
├── requirements.txt                 # Dependencies
└── .pre-commit-config.yaml          # Pre-commit hooks
```

## Quick Start

### Prerequisites

- Python 3.10, 3.11, or 3.12
- A Snowflake account with `SNOWFLAKE_SAMPLE_DATA` access
- A warehouse with sufficient compute (MEDIUM recommended)

### Setup

```bash
# Clone the repo
git clone <repo-url>
cd snowpark_fundamentals

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -e ".[dev,notebooks]"

# Configure credentials
cp .env.example .env
# Edit .env with your Snowflake credentials
```

### Run the Tutorial

```bash
# Start Jupyter
jupyter notebook notebooks/

# Or run individual notebooks
jupyter notebook notebooks/01_snowpark_dataframes.ipynb
```

### Run Tests

```bash
pytest tests/ -v --cov=snowpark_fundamentals
```

### dbt Feature Store (Week 2)

```bash
# Install dbt deps (dbt-snowflake is in the main project requirements)
make dbt-setup

# Run dbt models (requires source tables from notebooks 06-07)
make dbt-run

# Run dbt data quality tests
make dbt-test
```

## Key Technologies

| Package | Purpose |
|---------|---------|
| `snowflake-snowpark-python` | DataFrame API, session management |
| `snowflake-ml-python` | ML preprocessing, training, registry, Feature Store |
| `dbt-snowflake` | SQL feature engineering (separate venv) |
| `pytest` | Unit testing with mocking |
| `black` + `isort` + `flake8` | Code formatting and linting |
| `mypy` | Static type checking |

## Architecture

```
Snowflake Warehouse (all computation runs here)
    │
    ├── Snowpark DataFrames     → Lazy, distributed SQL execution
    ├── ML Preprocessing        → StandardScaler, Encoders (server-side)
    ├── Feature Store           → Entities, Feature Views, training sets
    ├── Model Training          → XGBoost, RandomForest (warehouse compute)
    ├── Model Registry          → Versioned, governed model storage
    └── Inference               → Feature retrieval + batch scoring
```

## Best Practices

This project follows production-grade patterns applicable to any Snowpark ML project.

### Configuration & Security

- **12-Factor App configuration** — all credentials loaded from environment variables via `.env`, never hardcoded
- **Immutable config** — `SnowflakeConfig` uses a frozen dataclass to prevent accidental mutation at runtime
- **Secret detection** — pre-commit hooks block private keys and large files from being committed
- **Security scanning** — Bandit and CodeQL run on every push to catch vulnerabilities early
- **`.env.example` template** — provides a safe, credential-free reference for new contributors

### Code Quality

- **Strict formatting** — Black (100-char line length) + isort enforce a consistent style across the entire codebase
- **Linting** — Flake8 catches common errors and anti-patterns before they reach review
- **Static type checking** — mypy validates type annotations to prevent runtime surprises
- **Pre-commit hooks** — automated checks run locally before every commit (formatting, linting, trailing whitespace, YAML validation, merge conflict detection)

### Testing

- **Mock-based unit tests** — all tests run without a live Snowflake connection, enabling fast CI feedback
- **Minimum 70% coverage** — enforced via `pytest-cov` in both local runs and CI
- **Multi-version matrix** — CI tests against Python 3.10, 3.11, and 3.12 to ensure compatibility
- **Isolated fixtures** — `conftest.py` provides reusable `mock_session`, `mock_dataframe`, and `sample_env_vars` fixtures

### Architecture & Design

- **Server-side computation** — all heavy operations (preprocessing, training, inference) execute inside the Snowflake warehouse, not locally
- **Lazy evaluation** — Snowpark DataFrames defer execution until an action is called, minimizing unnecessary data transfers
- **Modular package structure** — each concern (config, data, preprocessing, modeling, registry) lives in its own subpackage with clear interfaces
- **Factory pattern** — `create_session()` is the single entry point for session creation, ensuring consistent configuration
- **Scikit-learn compatible API** — Snowpark ML mirrors the familiar `fit` / `transform` / `predict` interface, reducing the learning curve
- **Reusable transformers** — fitted preprocessing objects are returned alongside transformed data for consistent inference on new data

### CI/CD

- **Five independent pipelines** — tests, lint, type check, security scan, and CodeQL run in parallel on every push and pull request
- **Dependabot** — automated dependency update PRs keep packages current and secure
- **Pull request template** — standardized PR descriptions ensure reviewers have the context they need
- **Issue templates** — separate templates for bugs and feature requests streamline triage

### Dependency Management

- **Pinned critical versions** — `xgboost` and `numpy` are pinned to versions available on the Snowflake conda channel, preventing runtime mismatches
- **Optional extras** — `[dev]` and `[notebooks]` extras keep production dependencies lean
- **Single source of truth** — `pyproject.toml` manages all project metadata, tool configs, and dependency specs in one place

## Series Roadmap

| Week | Topic | Status |
|------|-------|--------|
| 1 | **Snowpark ML Fundamentals** | Complete |
| 2 | **Feature Engineering in Snowflake** | Complete |
| 3 | Model Training at Scale | Upcoming |
| 4 | Inference Patterns | Upcoming |
| 5 | Monitoring & Observability | Upcoming |
| 6 | CI/CD for Snowflake ML | Upcoming |
| 7 | Governance & Access Control | Upcoming |

## License

MIT
