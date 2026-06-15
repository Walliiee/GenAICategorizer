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

# Install dependencies first, against a stub package, so the heavy layer
# (torch, sentence-transformers) stays cached when only src/ changes.
COPY pyproject.toml README.md ./
RUN mkdir -p src && touch src/__init__.py \
    && pip install --upgrade pip \
    && pip install ".[web]"

# Add the real source and (re)install just the package itself — deps are
# already present, so this stays fast on code-only changes.
COPY src ./src
RUN pip install --no-deps ".[web]"

# Run as a non-root user; ensure /app (incl. the HF cache dir) is writable.
RUN useradd -m -u 8888 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# The FastAPI app object lives at src.app:app. Bind to all interfaces so the
# service is reachable from outside the container.
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
