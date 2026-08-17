---
title: RAGent Reranker
emoji: 🛡️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# RAGent Reranker

Cross-encoder reranking service for [RAGent](https://rag-ent.onrender.com), running
`Xenova/ms-marco-MiniLM-L-6-v2` via fastembed's ONNX runtime.

RAGent's web service runs on Render's free tier (0.1 vCPU / 512MB), where this model
in-process measured ~106-122s per query and needed `batch_size=1` to avoid OOM. Here it gets
2 vCPU / 16GB, so it runs at full batch size in seconds.

The model is deliberately identical to RAGent's in-process fallback: scores stay on the same
raw-logit scale, so the retrieval quality gate's refusal thresholds stay valid without
re-calibration.

## API

### `GET /health`
Unauthenticated liveness check. Also the keepalive target — free Spaces sleep after 48h idle,
and RAGent's GitHub Actions cron pings this every 10 minutes.

```json
{ "status": "ok", "model": "Xenova/ms-marco-MiniLM-L-6-v2" }
```

### `POST /rerank`

```json
{ "query": "When was Far Cry 5 released?", "documents": ["...", "..."] }
```

```json
{ "scores": [7.21, -4.03], "model": "Xenova/ms-marco-MiniLM-L-6-v2", "elapsed_ms": 412 }
```

`scores` are returned **in input order**, one per document, as **raw logits** (roughly
-8..+11) — not normalized. The caller zips them directly onto its candidate list.

Max 200 documents per request (`413` beyond that). RAGent's largest real shape is 40.

## Auth

Set a `RERANK_SECRET` in **Settings → Secrets**; requests must then carry it as the
`X-Rerank-Key` header. Leave it unset to disable that check. If the Space itself is private,
callers additionally need `Authorization: Bearer <HF token>`.
