# Minimal image for the GenAI Categorizer web service (CPU-only).
#
#   docker build -t genai-categorizer .
#   docker run --rm -p 8000:8000 genai-categorizer
#
# The multilingual embedding model downloads on first request into HF_HOME;
# mount a volume there to cache it across runs.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install the package with web extras (FastAPI + uvicorn + python-multipart).
# Copy only what the build needs first so layers cache well.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[web]"

EXPOSE 8000

# The FastAPI app object lives at src.app:app. Bind to all interfaces so the
# service is reachable from outside the container.
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
