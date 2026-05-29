FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Run as non-root user — Celery and uvicorn refuse/warn when run as root
RUN useradd --no-create-home --shell /bin/false appuser \
    && mkdir -p /app/outputs /app/task_inputs /app/data \
    && chown -R appuser:appuser /app/outputs /app/task_inputs /app/data
USER appuser

ENV PYTHONUNBUFFERED=1