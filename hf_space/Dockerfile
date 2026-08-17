# ============================================================
# RAGent Reranker — Hugging Face Docker Space (free CPU Basic)
# ============================================================
# Spaces run the container as UID 1000, so a user is created before any
# COPY and every COPY sets --chown=user (a recursive chown afterwards
# would duplicate every file into a new layer).
# ============================================================

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Bake the ONNX model into the image. A Space's disk is ephemeral across
# restarts, so without this every cold start would re-download the model
# on the request path — the exact failure mode the eager load in app.py
# exists to prevent.
ENV FASTEMBED_CACHE_PATH=$HOME/app/.fastembed_cache
RUN python -c "\
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')"

COPY --chown=user app.py .

# Spaces expose port 7860 (see app_port in README.md's YAML block).
EXPOSE 7860

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
