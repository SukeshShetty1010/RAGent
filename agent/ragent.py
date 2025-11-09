# agent/ragent.py — FINAL UNIVERSAL RAG AGENT (November 09, 2025)
import json
import time
import logging
import threading
import itertools
import sys
from datetime import datetime, UTC
from typing import List, Dict, Any
import concurrent.futures
from pydantic import BaseModel, Field

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "deploy"))

from agent.tools import search_knowledge_base, fetch_news, search_igdb
from agent.ragent_client import chat_completion_remote

try:
    from ingest.ragent_ingestor import _clean_content
except:
    def _clean_content(x): return x.replace("[Upgrade subscription plan]", "").replace("Premium content", "").replace("Subscribe to read", "").strip()

try:
    from vector.index_manager import client as weaviate_client
    import atexit
    atexit.register(weaviate_client.close)
except:
    pass

from agent.constants import SOURCE_NEWS, SOURCE_IGDB, DEFAULT_TOP_K

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class ToolCall(BaseModel):
    tools: List[str] = Field(default=["kb", "news", "igdb"])
    query: str
    max_results: int = Field(default=15, ge=1, le=30)

FINAL_ANSWER_SYSTEM = """You are a gaming RAG agent. Answer ONLY using the context below.
NO hallucinations. NO old games. NO chit-chat.

If no relevant context → "No fresh info right now."

Else: 4-7 short bullets. Newest first.
Every bullet ends with [Source: NAME, YYYY-MM-DD]

Context:
{context}

Question: {query}

Answer now:"""

def _spinner_task(stop_event):
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if stop_event.is_set(): break
        sys.stdout.write(f'\rThinking {c} '); sys.stdout.flush(); time.sleep(0.1)
    sys.stdout.write('\r' + ' ' * 20 + '\r'); sys.stdout.flush()

class RAGAgent:
    def __init__(self):
        self.temperature = 0.0
        self.max_tokens = 512

    def _route(self, query: str) -> ToolCall:
        return ToolCall(query=query, max_results=15)

    @staticmethod
    def _run_kb(q: str, k: int) -> List[Dict]:
        try:
            docs = search_knowledge_base.invoke({"query": q, "top_k": k})
            return [{"content": d.page_content, "source": d.metadata.get("source","KB"), "date": str(d.metadata.get("created_at","2025-11-09")).split("T")[0]} for d in docs]
        except: return []

    @staticmethod
    def _run_news(q: str, limit: int) -> List[Dict]:
        try:
            raw = fetch_news.invoke({"query": q, "limit": limit*2})
            items = raw.get("headlines", {}).get("results", []) + raw.get("news", {}).get("results", [])
            results = []
            for i in items:
                content = _clean_content(f"{i.get('title','')}\n{i.get('description','')}\n{i.get('body','')}\n{i.get('content','')}")
                if len(content) < 70: continue
                date = (i.get("publishedAt") or "2025-11-09").split("T")[0]
                if int(date[:4]) < 2025: continue
                results.append({"content": content, "source": i.get("source","GNEWS"), "date": date})
            results.sort(key=lambda x: x["date"], reverse=True)
            return results[:limit]
        except: return []

    @staticmethod
    def _run_igdb(q: str, limit: int) -> List[Dict]:
        try:
            raw = search_igdb.invoke({"query": q, "limit": limit})
            items = raw.get("recent_games", []) + raw.get("searched_games", [])
            results = []
            for i in items:
                content = f"{i.get('name','')} — {i.get('summary','')[:400]}"
                if len(content) < 70: continue
                ts = i.get("first_release_date")
                date = datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d") if ts else "2025-11-09"
                if int(date[:4]) < 2025: continue
                results.append({"content": content, "source": "IGDB", "date": date})
            results.sort(key=lambda x: x["date"], reverse=True)
            return results
        except: return []

    def _execute_tools(self, call: ToolCall) -> List[Dict]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            futures = {
                "kb": ex.submit(self._run_kb, call.query, call.max_results),
                "news": ex.submit(self._run_news, call.query, call.max_results),
                "igdb": ex.submit(self._run_igdb, call.query, call.max_results)
            }
            return [{"tool": k, "data": f.result()} for k, f in futures.items()]

    def _synthesize(self, query: str, tool_results: List[Dict]) -> str:
        query_words = [w.lower().replace("'", "").replace("’", "") for w in query.split() if len(w) > 2]

        context_lines = []
        for r in tool_results:
            for item in r["data"]:
                content_clean = item["content"].lower().replace("’", "'").replace("'", "")
                if any(word in content_clean for word in query_words) or len(set(query_words) & set(content_clean.split())) >= 2:
                    cite = f"[Source: {item['source']}, {item['date']}]"
                    line = item["content"].replace("\n", " ").strip()[:600]
                    context_lines.append(f"- {line} {cite}")

        if not context_lines:
            all_items = [item for r in tool_results for item in r["data"]]
            all_items.sort(key=lambda x: x["date"], reverse=True)
            for item in all_items[:7]:
                cite = f"[Source: {item['source']}, {item['date']}]"
                context_lines.append(f"- {item['content'].replace('\n', ' ')[:600]} {cite}")

        context = "\n".join(context_lines[:14]) if context_lines else "No fresh info right now."
        prompt = FINAL_ANSWER_SYSTEM.format(context=context, query=query)

        stop_event = threading.Event()
        spinner = threading.Thread(target=_spinner_task, args=(stop_event,))
        spinner.start()

        try:
            answer = chat_completion_remote.remote(prompt=prompt, max_tokens=512, temperature=0.0)
            stop_event.set()
            spinner.join()
            return answer.strip()
        except Exception as e:
            stop_event.set()
            spinner.join()
            return f"LLM error. Context:\n{context}"

    def answer_query(self, query: str) -> Dict[str, Any]:
        start = time.time()
        call = self._route(query)
        results = self._execute_tools(call)
        answer = self._synthesize(query, results)
        return {
            "answer": answer,
            "tools_used": call.tools,
            "latencies": {"total": round(time.time() - start, 2)},
            "timestamp": datetime.now(UTC).isoformat() + "Z",
        }

if __name__ == "__main__":
    agent = RAGAgent()
    print("\nRAG Agent READY! (LIVE Modal Llama-3-8B)")
    print("Type 'quit' to exit\n")
    while True:
        q = input("> ").strip()
        if q.lower() in ("quit", "exit", "q"): break
        if not q: continue
        resp = agent.answer_query(q)
        print("\n" + "="*60)
        print(resp["answer"])
        print(f"\nTools: {resp['tools_used']} | Total: {resp['latencies']['total']}s")
        print("="*60 + "\n")