# ============================================================
# RAGent — Dockerfile (Render Free Tier)
# ============================================================
# Entry point: Streamlit streaming UI
# GPU inference offloaded to Modal (not in this container)
# ============================================================

FROM python:3.11-slim

# System deps for building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (Docker cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Render uses PORT env var; Streamlit binds to it
ENV PORT=8501

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/_stcore/health || exit 1

# Streamlit config: headless mode, no file watcher, CORS disabled
CMD ["sh", "-c", "streamlit run ui/app_streaming.py \
    --server.port=${PORT} \
    --server.headless=true \
    --server.fileWatcherType=none \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false"]
