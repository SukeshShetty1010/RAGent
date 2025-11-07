# agent/ragent.py
"""
FINAL WORKING RAG AGENT – November 07, 2025
→ Local Weaviate + Tools
→ LIVE Modal Llama-3-8B on T4 ($0.0012/query)
"""

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

# Add deploy folder to path
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "deploy"))

# Local tools
from agent.tools import search_knowledge_base, fetch_news, search_igdb

# LIVE MODAL CONNECTION (NO MORE HYDRATION ERROR!)
from agent.ragent_client import chat_completion_remote

# Clean news content
try:
    from ingest.ragent_ingestor import _clean_content
except:
    def _clean_content(x): return x

# Close Weaviate on exit
try:
    from vector.index_manager import client as weaviate_client
    import atexit
    atexit.register(weaviate_client.close)
except:
    pass

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class ToolCall(BaseModel):
    tools: List[str] = Field(..., description="['kb','news','igdb']")
    query: str = Field(..., description="Refined query")
    max_results: int = Field(default=3, ge=1, le=5)

FINAL_ANSWER_SYSTEM = """You are a concise gaming assistant.
Answer in 1-3 short paragraphs or bullet points.
Cite every fact inline: [Source: <source>, <YYYY-MM-DD>].

Context:
{context}

Question: {query}

Answer:"""

def _spinner_task(stop_event):
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if stop_event.is_set():
            break
        sys.stdout.write(f'\rThinking {c} ')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r' + ' ' * 20 + '\r')
    sys.stdout.flush()

class RAGAgent:
    def __init__(self):
        self.max_tokens_answer = 512
        self.temperature = 0.1

    def _route(self, query: str) -> ToolCall:
        return ToolCall(tools=["kb", "news", "igdb"], query=query, max_results=3)

    @staticmethod
    def _run_kb(q: str, k: int) -> List[Dict]:
        try:
            docs = search_knowledge_base.invoke({"query": q, "top_k": k})
            return [
                {
                    "content": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                    "source": doc.metadata.get("source", "KB"),
                    "id": doc.metadata.get("article_id"),
                    "date": str(doc.metadata.get("created_at", "2025-11-07")).split(" ")[0],
                }
                for doc in docs
            ]
        except Exception as e:
            log.error(f"KB error: {e}")
            return []

    @staticmethod
    def _run_news(q: str, limit: int) -> List[Dict]:
        try:
            raw = fetch_news.invoke({"query": q, "limit": limit, "country": "us"})
            items = raw.get("headlines", {}).get("results", [])[:limit]
            results = []
            for i in items:
                content = _clean_content(f"{i.get('title','')} – {i.get('description','')}".strip())
                if content:
                    date = (i.get("published_at") or "")[:10] or datetime.now(UTC).strftime("%Y-%m-%d")
                    results.append({
                        "content": content,
                        "source": "APITube.io",
                        "id": i.get("id"),
                        "date": date,
                    })
            return results
        except Exception as e:
            log.error(f"News error: {e}")
            return []

    @staticmethod
    def _run_igdb(q: str, limit: int) -> List[Dict]:
        try:
            raw = search_igdb.invoke({"query": q, "limit": limit})
            games = raw.get("results", [])[:limit]
            return [
                {
                    "content": f"{g.get('name','Unknown')} ({g.get('first_release_date','TBA')})",
                    "source": "IGDB",
                    "id": g.get("id"),
                    "date": (datetime.fromtimestamp(g.get("first_release_date")).strftime("%Y-%m-%d")
                    if isinstance(g.get("first_release_date"), (int, float)) and g.get("first_release_date")
                    else str(g.get("first_release_date", "TBA")).split(" ")[0]),
                }
                for g in games if g.get("name")
            ]
        except Exception as e:
            log.error(f"IGDB error: {e}")
            return []

    def _execute_tools(self, call: ToolCall):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            if "kb" in call.tools:
                futures["kb"] = executor.submit(self._run_kb, call.query, call.max_results)
            if "news" in call.tools:
                futures["news"] = executor.submit(self._run_news, call.query, call.max_results)
            if "igdb" in call.tools:
                futures["igdb"] = executor.submit(self._run_igdb, call.query, call.max_results)

            for name, future in futures.items():
                data = future.result()
                results.append({"tool": name, "data": data})
        return results

    def _synthesize(self, query: str, tool_results: List[Dict]) -> str:
        context_lines = []
        for r in tool_results:
            for item in r["data"]:
                cite = f"[Source: {item['source']}, {item['date']}"
                if item.get("id"): cite += f" ID:{item['id']}"
                cite += "]"
                context_lines.append(f"{item['content']}\n{cite}")
        context = "\n\n".join(context_lines)
        prompt = FINAL_ANSWER_SYSTEM.format(context=context, query=query)

        stop_event = threading.Event()
        spinner = threading.Thread(target=_spinner_task, args=(stop_event,))
        spinner.start()

        try:
            answer = chat_completion_remote.remote(
                prompt=prompt,
                max_tokens=self.max_tokens_answer,
                temperature=self.temperature
            )
            stop_event.set()
            spinner.join()
            return answer.strip()
        except Exception as e:
            stop_event.set()
            spinner.join()
            log.error(f"Modal LLM failed: {e}")
            return "No answer (LLM error). Raw results above."

    def answer_query(self, query: str) -> Dict[str, Any]:
        total_start = time.time()
        call = self._route(query)
        exec_start = time.time()
        results = self._execute_tools(call)
        exec_lat = round(time.time() - exec_start, 2)
        synth_start = time.time()
        answer = self._synthesize(query, results)
        synth_lat = round(time.time() - synth_start, 2)
        total_lat = round(time.time() - total_start, 2)

        sources = [
            {"source": item["source"], "id": item.get("id"), "date": item["date"]}
            for r in results for item in r["data"]
        ]

        return {
            "answer": answer,
            "sources": sources,
            "tools_used": call.tools,
            "latencies": {"tools": exec_lat, "llm": synth_lat, "total": total_lat},
            "timestamp": datetime.now(UTC).isoformat() + "Z",
        }

if __name__ == "__main__":
    agent = RAGAgent()
    print("\nRAG Agent READY! (LIVE Modal Llama-3-8B)")
    print("Type 'quit' to exit\n")
    while True:
        q = input("> ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        resp = agent.answer_query(q)
        print("\n" + "="*60)
        print(resp["answer"])
        print(f"\nTools: {resp['tools_used']} | Total: {resp['latencies']['total']}s")
        print("="*60 + "\n")