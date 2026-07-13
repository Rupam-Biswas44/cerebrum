<div align="center">

# 🧠 Cerebrum

### Enterprise Multi-Agent AI Platform

*Describe your goal. Cerebrum handles the rest.*

[![CI](https://github.com/Rupam-Biswas44/cerebrum/actions/workflows/ci.yml/badge.svg)](https://github.com/Rupam-Biswas44/cerebrum/actions/workflows/ci.yml)
[![CD](https://github.com/Rupam-Biswas44/cerebrum/actions/workflows/cd.yml/badge.svg)](https://github.com/Rupam-Biswas44/cerebrum/actions/workflows/cd.yml)
[![Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen)](https://github.com/Rupam-Biswas44/cerebrum)
[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=nextdotjs)](https://nextjs.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[**Documentation**](docs/) · [**API Reference**](apps/api/) · [**Contributing**](CONTRIBUTING.md)

</div>

---

## 🚀 What is Cerebrum?

Cerebrum is a **production-grade, open-source enterprise AI platform** that autonomously ingests data, coordinates specialized AI agents, trains machine learning models, generates insights, and delivers results — all from a single natural language goal.

**You say:**
> *"Analyze our Q3 sales data, find why revenue dropped, build a forecast for Q4, and prepare an executive report."*

**Cerebrum does:**
1. 📥 Ingests and validates your data
2. 🧹 Cleans and profiles it automatically
3. 📊 Runs statistical analysis and EDA
4. 🤖 Trains forecasting models with AutoML
5. 🔍 Detects anomalies and root causes
6. 📈 Generates interactive visualizations
7. 📝 Writes an evidence-linked executive report
8. 📤 Exports to PDF, PowerPoint, or dashboard

**No code required. Fully auditable. Production ready.**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          User Interfaces                         │
│              Web App │ CLI │ REST API │ Python SDK              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                    API Gateway (Traefik)                          │
│              Rate Limiting │ Auth │ Load Balancing               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                   FastAPI Backend (Python 3.13)                   │
│         REST API │ WebSocket │ OpenTelemetry │ Structured Logs   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                  Task Orchestrator (LangGraph)                    │
│              Goal Decomposition │ Agent Routing                  │
└────────────┬────────────────────────────────────────┬───────────┘
             │                                        │
┌────────────▼────────────────────────────────────────▼──────────┐
│                         Agent Layer                              │
│  🧠 Planner │ 🔧 Data Engineer │ 📊 Statistician │ 🤖 ML Eng.  │
│  📈 Visualizer │ ✍️ Writer │ 🔍 Critic │ 💾 Memory │ 🛡️ Security │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                         Data Layer                               │
│  PostgreSQL │ Redis │ MinIO │ Qdrant │ Neo4j │ DuckDB           │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      Observability Stack                          │
│           Prometheus │ Grafana │ Loki │ Tempo │ Alerts           │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🤖 Multi-Agent Intelligence
- **Planner Agent** — Decomposes complex goals into executable subtasks
- **Data Engineer Agent** — Cleans, validates, and transforms raw data
- **Statistician Agent** — Runs hypothesis tests, correlation analysis, EDA
- **ML Engineer Agent** — AutoML with XGBoost, LightGBM, PyTorch, Prophet
- **Visualization Agent** — Creates interactive charts and dashboards
- **Writer Agent** — Generates evidence-linked executive reports
- **Critic Agent** — Detects hallucinations and validates conclusions
- **Memory Agent** — Maintains context across sessions

### 📊 Data Intelligence
- Upload CSV, Excel, JSON, Parquet, SQL databases
- Automatic schema inference and data profiling
- Natural language queries → SQL
- Anomaly detection and outlier analysis
- Time-series forecasting

### 🧪 ML Pipeline
- AutoML with hyperparameter optimization (Optuna)
- MLflow experiment tracking and model registry
- SHAP explainability for every prediction
- ONNX export for fast inference
- Reproducible experiments

### 🏭 Production Grade
- JWT + OAuth2 authentication (Google, GitHub)
- Role-based access control (Admin, Analyst, Viewer)
- OpenTelemetry distributed tracing
- Prometheus metrics + Grafana dashboards
- Docker Compose + Kubernetes Helm charts
- Full CI/CD with GitHub Actions

---

## 🛠️ Quick Start

### Prerequisites
- Docker 24+
- Docker Compose v2
- 8GB RAM minimum (16GB recommended)

### Run Locally

```bash
# Clone the repository
git clone https://github.com/Rupam-Biswas44/cerebrum.git
cd cerebrum

# Copy environment variables
cp .env.example .env

# Start the full stack
make up

# Open the app
open http://localhost:3000
```

### Using the CLI

```bash
pip install cerebrum-cli

cerebrum login
cerebrum analyze --file sales_data.csv --goal "Find why revenue dropped in Q3"
```

---

## 📦 Stack

| Category | Technologies |
|---|---|
| **Backend** | Python 3.13, FastAPI, SQLAlchemy, Alembic, Celery |
| **AI/ML** | PyTorch, scikit-learn, XGBoost, LightGBM, MLflow, ONNX |
| **LLM** | LangGraph, Sentence Transformers, Ollama, OpenAI API |
| **Data** | DuckDB, Polars, Apache Arrow |
| **Vector DB** | Qdrant |
| **Graph DB** | Neo4j |
| **Frontend** | Next.js 15, React, ShadCN, Tailwind, Recharts |
| **Database** | PostgreSQL 16, Redis 7 |
| **Storage** | MinIO (S3-compatible) |
| **Infra** | Docker, Kubernetes, Helm |
| **Observability** | Prometheus, Grafana, Loki, Tempo, OpenTelemetry |
| **CI/CD** | GitHub Actions, CodeQL, Trivy, Dependabot |

---

## 🗺️ Roadmap

- [x] **v0.1** — Repository foundation and engineering standards
- [ ] **v0.2** — Core backend and authentication
- [ ] **v0.3** — Data pipeline and storage services
- [ ] **v0.4** — Multi-agent framework
- [ ] **v0.5** — ML pipeline and experiment tracking
- [ ] **v0.6** — Frontend application
- [ ] **v0.7** — Observability stack
- [ ] **v0.8** — Security hardening
- [ ] **v0.9** — CI/CD and Kubernetes deployment
- [ ] **v1.0** — Production release

---

## 🧪 Testing

```bash
make test          # Run all tests
make test-unit     # Unit tests only
make test-int      # Integration tests (requires Docker)
make test-cov      # With coverage report
make test-load     # Load tests
```

---

## 🤝 Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

```bash
make dev-setup  # Setup development environment
make lint       # Run pre-commit hooks
make test       # Run tests before submitting PR
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ by <strong>Rupam Biswas</strong> · MSc Data Science, TU Dortmund
</div>
