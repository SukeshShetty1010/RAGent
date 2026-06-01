# ============================================================
# RAGent — Dockerfile (Render Free Tier)
# ============================================================
# Entry point: FastAPI backend (uvicorn)
# LLM inference via Groq API (no GPU needed)
# ============================================================

FROM python:3.11-slim

# System deps for building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (Docker cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Render uses PORT env var; fall back to Render's default web port locally
EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

# Start FastAPI server
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
