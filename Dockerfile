# syntax=docker/dockerfile:1

# --------------------------------------------------------------------------- #
# Stage 1 — builder: resolve and install deps into a self-contained .venv      #
# --------------------------------------------------------------------------- #
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies first, from the lockfile only, so this layer is cached
# and rebuilt only when pyproject.toml / uv.lock change (not on every code edit).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Now bring in the application source and finish the sync.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# --------------------------------------------------------------------------- #
# Stage 2 — runtime: slim image with just the venv + source, run as non-root   #
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser
COPY --from=builder --chown=appuser:appuser /app /app
USER appuser

EXPOSE 8000

# Container-native liveness check that hits the /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
