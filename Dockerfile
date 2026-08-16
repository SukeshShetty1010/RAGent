# ============================================================
# RAGent — Dockerfile (Render Free Tier)
# ============================================================
# Multi-stage build:
# 1. Build Next.js static frontend
# 2. Run FastAPI backend serving static frontend
# ============================================================

# ---------- Stage 1: Build Frontend ----------
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
# Copy frontend dependencies and code
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
# Build static export (generates /app/frontend/out)
RUN npm run build

# ---------- Stage 2: Build & Run Backend ----------
FROM python:3.11-slim

# System deps for building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (Docker cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the BM25 and cross-encoder ONNX models into a fixed cache
# path baked into the image, so a cold container start never triggers an
# on-demand download (both models are loaded eagerly at module import in
# retriever/rag_retriever.py — a failure here must surface as a build
# error, not a runtime hang on the first live request).
ENV FASTEMBED_CACHE_PATH=/app/.fastembed_cache
RUN python -c "\
from fastembed import SparseTextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
SparseTextEmbedding(model_name='Qdrant/bm25'); \
TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')"

# Copy backend application code
COPY . .

# Copy static frontend build from Stage 1
COPY --from=frontend-builder /app/frontend/out ./frontend_build

# Render uses PORT env var; fall back to Render's default web port locally
EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

# Start FastAPI server
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
