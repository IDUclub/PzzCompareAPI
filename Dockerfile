FROM python:3.11-slim

# uv installs the locked dependencies into /app/.venv with the image's Python.
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /uvx /bin/
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --no-install-project --no-cache
ENV PATH="/app/.venv/bin:$PATH"

COPY . /app

# Run as non-root user — Celery and uvicorn refuse/warn when run as root
RUN useradd --no-create-home --shell /bin/false appuser \
    && mkdir -p /app/outputs /app/task_inputs /app/data \
    && chown -R appuser:appuser /app/outputs /app/task_inputs /app/data
USER appuser

ENV PYTHONUNBUFFERED=1
