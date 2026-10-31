# syntax=docker/dockerfile:1

# ---- builder: resolve and install runtime dependencies into /app/.venv ----
FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=2.5.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry-cache

WORKDIR /app

# Pinned to the Poetry that generated poetry.lock (see its header).
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-ansi \
    && rm -rf "$POETRY_CACHE_DIR" /root/.cache

# ---- runtime: slim base, no build tooling, no Poetry, non-root ----
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH

# curl is required by the docker-compose healthcheck (absent from slim).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app \
    && chown app:app /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
# src/ssl_trust.py loads ../certs/*.crt at runtime, so certs must ship.
COPY --chown=app:app src ./src
COPY --chown=app:app certs ./certs
# Default dev .env uses MODELS_CONFIG=./data/models.json; production mounts /data.
COPY --chown=app:app data/models.json ./data/models.json

USER app

EXPOSE 8000

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "10", "--timeout-graceful-shutdown", "600", "--loop", "uvloop", "--limit-concurrency", "500"]
