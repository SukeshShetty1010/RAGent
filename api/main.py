import os
import sys
import json
import asyncio
import queue
import threading
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.execution_engine_streaming import StreamingRageEngine
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAGent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazily-built, cached engine instance (owns connections like Qdrant).
# Built on first request rather than at import time, so importing this
# module (e.g. for test collection) never touches Qdrant.
_engine: StreamingRageEngine | None = None


def get_engine() -> StreamingRageEngine:
    global _engine
    if _engine is None:
        _engine = StreamingRageEngine()
    return _engine

class ChatRequest(BaseModel):
    query: str

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ragent"}

@app.get("/ping")
def ping():
    """
    Keep-alive endpoint for cron-job.org (or similar).
    Set up a free cron job to GET this URL every 10 minutes
    to prevent Render free-tier cold starts.
    """
    from datetime import datetime, timezone
    return {"pong": True, "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/api/usage")
def usage_dashboard():
    """
    Today's Gemini/Groq request counts, by provider and by surface
    (chat, decision, embedding, ragas_judge, ablation), aggregated
    across Render chat traffic and local RAGAS/eval runs via the
    shared Qdrant-backed UsageCounter. Fail-soft: returns in-process
    numbers with "degraded": true on any Qdrant error, never a 500.
    """
    from utils.usage_counter import UsageCounter
    return UsageCounter.get().read_today()


@app.post("/api/chat")
async def chat_endpoint(request: Request, body: ChatRequest):
    """
    Streaming chat endpoint. Yields SSE events:
    - 'token': Incremental LLM text.
    - 'stage': Pipeline progress events.
    - 'done': Final JSON with KPIs, evidence, and raw_metrics.
    """
    q = queue.Queue()
    
    def on_token(token: str):
        q.put({"type": "token", "data": token})
        
    def on_stage(stage):
        stage_dict = {
            "name": stage.name,
            "status": stage.status,
            "duration_ms": stage.duration_ms,
            "data": stage.data
        }
        q.put({"type": "stage", "data": stage_dict})

    def run_engine():
        try:
            result = get_engine().run_streaming(
                query=body.query,
                on_token_callback=on_token,
                on_stage_callback=on_stage
            )
            # Make sure evidence is cleanly serializable
            safe_evidence = []
            for ev in result.evidence:
                safe_ev = {k: v for k, v in ev.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                safe_evidence.append(safe_ev)

            q.put({"type": "done", "data": {
                "final_answer": result.final_answer,
                "kpis": result.kpis,
                "evidence": safe_evidence,
                "agent_decisions": result.agent_decisions
            }})
        except Exception as e:
            q.put({"type": "error", "data": str(e)})
        finally:
            # Force-deliver this request's trace before the daemon thread
            # exits — a Render container idling out between requests must
            # not silently drop the last trace's spans.
            from utils import tracing
            tracing.flush()

            from utils.usage_counter import UsageCounter
            UsageCounter.get().flush()

    threading.Thread(target=run_engine, daemon=True).start()

    async def event_generator():
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break

            # Await the blocking q.get in a thread pool
            item = await asyncio.get_event_loop().run_in_executor(None, q.get)
            
            if item["type"] == "token":
                yield {"event": "token", "data": json.dumps({"text": item["data"]})}
            elif item["type"] == "stage":
                yield {"event": "stage", "data": json.dumps(item["data"])}
            elif item["type"] == "done":
                yield {"event": "done", "data": json.dumps(item["data"])}
                break
            elif item["type"] == "error":
                yield {"event": "error", "data": json.dumps({"error": item["data"]})}
                break

    return EventSourceResponse(event_generator())

# Serve static frontend files if they exist (for unified deployment)
STATIC_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend_build')
if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
