# ============================================================
# Cerebrum API — Multi-stage Docker Build
# ============================================================

# ── Stage 1: Base ────────────────────────────────────────────
FROM python:3.13-slim AS base

LABEL maintainer="Rupam Biswas <rupam@cerebrum.ai>"
LABEL org.opencontainers.image.title="Cerebrum API"
LABEL org.opencontainers.image.source="https://github.com/yourusername/cerebrum"
LABEL org.opencontainers.image.licenses="MIT"

# Security: run as non-root user
RUN groupadd -r cerebrum && useradd -r -g cerebrum -d /app -s /bin/bash cerebrum

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set pip cache
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONASYNCIODEBUG=0

WORKDIR /app

# ── Stage 2: Dependencies ─────────────────────────────────────
FROM base AS dependencies

COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -e ".[dev]" --no-deps --only-binary :all: || \
    pip install -e "."

# ── Stage 3: Development ──────────────────────────────────────
FROM dependencies AS development

# Install dev dependencies
RUN pip install pytest pytest-asyncio pytest-cov httpx

COPY . .
RUN chown -R cerebrum:cerebrum /app

USER cerebrum

EXPOSE 8000

CMD ["uvicorn", "cerebrum.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]

# ── Stage 4: Production ───────────────────────────────────────
FROM base AS production

# Copy pre-installed dependencies from builder
COPY --from=dependencies /usr/local/lib/python3.13 /usr/local/lib/python3.13
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application source
COPY --chown=cerebrum:cerebrum . .

# Set proper permissions
RUN mkdir -p /app/logs && chown -R cerebrum:cerebrum /app

USER cerebrum

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["gunicorn", "cerebrum.main:app", \
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--workers", "4", \
    "--bind", "0.0.0.0:8000", \
    "--timeout", "120", \
    "--keepalive", "5", \
    "--max-requests", "1000", \
    "--max-requests-jitter", "100", \
    "--access-logfile", "-", \
    "--error-logfile", "-", \
    "--log-level", "info"]
