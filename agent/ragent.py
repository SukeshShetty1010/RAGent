# agent/ragent.py
"""
Hybrid RAG Agent using a remote LLM hosted via Modal.
"""

import json
import time
import logging
import threading
import itertools
import sys
from datetime import datetime
from typing import List, Dict, Any

from pydantic import BaseModel, Field

from agent.tools import search_knowledge_base, fetch_news, search_igdb
from agent.remote_llm import chat_completion, get_text_from_response, RemoteLLMError

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class ToolCall(BaseModel):
    tools: List[str] = Field(..., description="['kb','news','igdb']")
    query: str = Field(..., description="Refined query for the tools")
    max_results: int = Field(default=3, ge=1, le=5)

TOOL_ROUTER_SYSTEM = """You are a tool-router for a gaming RAG agent.
Decide which tools to call in parallel:
- kb   : vector search on stored knowledge
- news : live news headlines
- igdb : live IGDB game data

Return **only** valid JSON matching the ToolCall schema:
{ "tools": ["kb","news"], "query": "...", "max_results": 3 }
"""

FINAL_ANSWER_SYSTEM = """You are a concise gaming assistant.
Synthesize the tool results into 1-3 short paragraphs or bullet-points.
Cite every fact inline: [Source: <source>, <YYYY-MM-DD>].

Context:
{context}

Question: {query}

Answer:"""

def _spinner_task(stop_event):
    for c in itertools.cycle(['|','/','-','\\']):
        if stop_event.is_set():
            break
        sys.stdout.write('\rProcessing '+c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r'+' '*20+'\r')
    sys.stdout.flush()

class RAGAgent:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model = model_name
        self.temperature = 0.0
        self.max_tokens_router = 128
        self.max_tokens_answer = 512

    def _route(self, query: str) -> ToolCall:
        messages = [
            {"role": "system", "content": TOOL_ROUTER_SYSTEM},
            {"role": "user", "content": query},
        ]
        stop_event = threading.Event()
        spinner = threading.Thread(target=_spinner_task, args=(stop_event,))
        spinner.start()
        try:
            resp = chat_completion(
                messages,
                model=self.model,
                max_tokens=self.max_tokens_router,
                temperature=self.temperature,
                timeout=300
            )
            stop_event.set()
            spinner.join()
            text = get_text_from_response(resp).strip()
            payload = json.loads(text)
            return ToolCall(**payload)
        except (RemoteLLMError, json.JSONDecodeError) as e:
            stop_event.set()
            spinner.join()
            log.warning(f"Router error: {e}")
            log.warning("Defaulting to all tools")
            return ToolCall(tools=["kb","news","igdb"], query=query, max_results=3)

    @staticmethod
    def _run_kb(q: str, k: int) -> List[Dict]:
        docs = search_knowledge_base.invoke({"query": q, "top_k": k})
        return [
            {
                "content": doc.page_content[:250] + ("..." if len(doc.page_content) > 250 else ""),
                "source": doc.metadata.get("source", "KB"),
                "id": doc.metadata.get("article_id"),
                "date": doc.metadata.get("created_at", "")[:10],
            }
            for doc in docs
        ]

    @staticmethod
    def _run_news(q: str, limit: int) -> List[Dict]:
        raw = fetch_news.invoke({"query": q, "limit": limit, "country": "us"})
        items = raw.get("headlines", {}).get("results", [])[:limit]
        return [
            {
                "content": f"{i.get('title','')} – {i.get('description','')}",
                "source": "APITube.io",
                "id": i.get("id"),
                "date": i.get("published_at", "")[:10] or datetime.now().strftime("%Y-%m-d"),
            }
            for i in items
        ]

    @staticmethod
    def _run_igdb(q: str, limit: int) -> List[Dict]:
        raw = search_igdb.invoke({"query": q, "limit": limit})
        games = raw.get("results", [])[:limit]
        return [
            {
                "content": f"{g.get('name','')} ({g.get('first_release_date','')})",
                "source": "IGDB",
                "id": g.get("id"),
                "date": (datetime.utcfromtimestamp(g.get("first_release_date")).strftime("%Y-%m-d") 
                         if g.get("first_release_date") else ""),
            }
            for g in games
        ]

    def _execute_tools(self, call: ToolCall) -> List[Dict[str, Any]]:
        tool_map = {
            "kb": lambda: self._run_kb(call.query, call.max_results),
            "news": lambda: self._run_news(call.query, call.max_results),
            "igdb": lambda: self._run_igdb(call.query, call.max_results),
        }
        results: List[Dict[str, Any]] = []
        for name in call.tools:
            start_t = time.time()
            try:
                data = tool_map[name]()
                latency = round(time.time()-start_t, 2)
                results.append({"tool": name, "data": data, "latency": latency})
            except Exception as e:
                log.error(f"Tool {name} error: {e}")
                results.append({"tool": name, "data": [], "latency": 0})
        return results

    def _synthesize(self, query: str, tool_results: List[Dict]) -> str:
        context = "\n".join(f"[{r['tool'].upper()}] {i['content']}"
                            for r in tool_results for i in r["data"])
        system_prompt = FINAL_ANSWER_SYSTEM.format(context=context, query=query)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        for r in tool_results:
            for item in r["data"]:
                cite = f"[Source: {item['source']}, {item['date']}"
                if item.get("id"):
                    cite += f" (ID:{item['id']})"
                cite += "]"
                messages.append({"role":"assistant","content":f"{item['content']}\n{cite}"})

        stop_event = threading.Event()
        spinner = threading.Thread(target=_spinner_task, args=(stop_event,))
        spinner.start()
        try:
            resp = chat_completion(
                messages,
                model=self.model,
                max_tokens=self.max_tokens_answer,
                temperature=self.temperature,
                timeout=600
            )
            stop_event.set()
            spinner.join()
            answer_text = get_text_from_response(resp).strip()
        except RemoteLLMError as e:
            stop_event.set()
            spinner.join()
            log.error(f"Synthesis error: {e}")
            answer_text = "\n".join(
                f"{r['tool'].upper()}: {', '.join(i['content'][:80] for i in r['data'])}"
                for r in tool_results
            )

        for r in tool_results:
            for item in r["data"]:
                placeholder = f"[{r['tool'].upper()}]"
                citation = f"[Source: {item['source']}, {item['date']}"
                if item.get("id"):
                    citation += f" (ID:{item['id']})"
                citation += "]"
                answer_text = answer_text.replace(placeholder, citation, 1)

        return answer_text

    def answer_query(self, query: str) -> Dict[str, Any]:
        total_start = time.time()

        route_start = time.time()
        call = self._route(query)
        route_lat = time.time() - route_start

        exec_start = time.time()
        results = self._execute_tools(call)
        exec_lat = time.time() - exec_start

        synth_start = time.time()
        answer = self._synthesize(query, results)
        synth_lat = time.time() - synth_start

        total_lat = time.time() - total_start
        log.info(f"Query answered in {round(total_lat,2)}s (route:{round(route_lat,2)}s execute:{round(exec_lat,2)}s synth:{round(synth_lat,2)}s)")

        sources = [
            {"source": item['source'], "id": item.get('id'), "date": item['date']}
            for r in results for item in r["data"]
        ]

        return {
            "answer": answer,
            "sources": sources,
            "tools_used": call.tools,
            "latencies": {
                "total": round(total_lat,2),
                "route": round(route_lat,2),
                "execute": round(exec_lat,2),
                "synthesize": round(synth_lat,2),
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

if __name__ == "__main__":
    agent = RAGAgent()
    print("Enter your query (or 'quit'):")
    while True:
        q = input("> ")
        if q.lower().strip() in ("quit", "exit"):
            break
        response = agent.answer_query(q)
        print("Answer:", response["answer"])
        print("Tools used:", response["tools_used"])
        print("Sources:", response["sources"])
        print("Latency:", response["latencies"]["total"], "s")
