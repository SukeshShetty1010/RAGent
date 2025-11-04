# agent/ragent.py
"""
Hybrid RAG Agent (KB + live news + IGDB)
- LLM: google/gemma-2-2b-it (GPU, less than or equal to 8 GB VRAM)
- Tool selection: LLM-structured Pydantic
- Parallel tool execution
- Inline citations
- Structured JSON output
"""

import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any

import torch
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_huggingface import HuggingFacePipeline

from utils.gpu_utils import get_device
from .tools import search_knowledge_base, fetch_news, search_igdb

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# LLM (lazy load, GPU-first, CPU-fallback)
# --------------------------------------------------------------------------- #
_llm: HuggingFacePipeline | None = None

class ToolCall(BaseModel):
    tools: List[str] = Field(..., description="['kb','news','igdb']")
    query: str = Field(..., description="Refined query for the tools")
    max_results: int = Field(default=3, ge=1, le=5)

def _get_llm() -> HuggingFacePipeline:
    """Load Gemma-2-2B-IT with automatic GPU/CPU placement."""
    global _llm
    if _llm is None:
        device = get_device()

        if device == "cpu":
            log.warning(
                "GPU not detected – falling back to CPU. Latency >5 s expected. "
                "Use docker-compose.gpu.yml with `--gpus all` for GPU."
            )

        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        model_id = "google/gemma-2-2b-it"          # <-- NEW MODEL

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,          # Gemma-2 works best in fp16
            device_map="auto",            # requires `accelerate`
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            device_map="auto",
            dtype=torch.float16,
        )
        # Gemma uses eos_token as pad token
        pipe.model.generation_config.pad_token_id = tokenizer.eos_token_id

        _llm = HuggingFacePipeline(pipeline=pipe)
        log.info(f"Gemma-2-2B-IT loaded on {device.upper()}")

    return _llm

# --------------------------------------------------------------------------- #
# Prompt templates
# --------------------------------------------------------------------------- #
TOOL_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a tool-router for a gaming RAG agent.
Decide which tools to call in parallel:
- kb   : vector search on stored knowledge
- news : live news headlines
- igdb : live IGDB game data

Return **only** valid JSON matching the ToolCall schema."""),
    ("human", "{query}")
])

FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a concise gaming assistant.
Synthesize the tool results into 1-3 short paragraphs or bullet points.
Cite every fact inline: [Source: <source>, <YYYY-MM-DD>].
If a fact comes from a document, include its article_id when available.

Context:
{context}

Question: {query}

Answer:"""),
    ("placeholder", "{messages}")
])

# --------------------------------------------------------------------------- #
# Agent class
# --------------------------------------------------------------------------- #
class RAGAgent:
    def __init__(self):
        self.llm = _get_llm()
        self.router_chain = TOOL_ROUTER_PROMPT | self.llm | ToolCall
        self.answer_chain = FINAL_ANSWER_PROMPT | self.llm

    # ------------------- tool routing -------------------
    def _route(self, query: str) -> ToolCall:
        try:
            return self.router_chain.invoke({"query": query})
        except Exception as e:
            log.warning(f"Router failed ({e}); using all tools")
            return ToolCall(tools=["kb", "news", "igdb"], query=query, max_results=3)

    # ------------------- tool helpers -------------------
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
                "date": i.get("published_at", "")[:10] or datetime.now().strftime("%Y-%m-%d"),
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
                "date": (datetime.utcfromtimestamp(g.get("first_release_date"))
                         .strftime("%Y-%m-%d") if g.get("first_release_date") else ""),
            }
            for g in games
        ]

    # ------------------- parallel execution -------------------
    def _execute_tools(self, call: ToolCall) -> List[Dict[str, Any]]:
        tool_map = {
            "kb": lambda: self._run_kb(call.query, call.max_results),
            "news": lambda: self._run_news(call.query, call.max_results),
            "igdb": lambda: self._run_igdb(call.query, call.max_results),
        }
        results = []
        for name in call.tools:
            start = time.time()
            try:
                data = tool_map[name]()
                results.append({"tool": name, "data": data, "latency": round(time.time() - start, 2)})
            except Exception as e:
                log.error(f"Tool {name} error: {e}")
                results.append({"tool": name, "data": [], "latency": 0})
        return results

    # ------------------- answer synthesis -------------------
    def _synthesize(self, query: str, tool_results: List[Dict]) -> str:
        ctx_parts = []
        for r in tool_results:
            for item in r["data"]:
                ctx_parts.append(f"[{r['tool'].upper()}] {item['content']}")
        context = "\n".join(ctx_parts)

        messages = [HumanMessage(content=query)]
        for r in tool_results:
            for item in r["data"]:
                cite = f"[Source: {item['source']}, {item['date']}"
                if item.get("id"):
                    cite += f" (ID:{item['id']})"
                cite += "]"
                messages.append(ToolMessage(content=item["content"],
                                            tool_call_id=f"{r['tool']}_{item.get('id','')}",
                                            name=r["tool"]))

        try:
            raw = self.answer_chain.invoke({"query": query, "context": context, "messages": messages})
            answer = raw.strip()
        except Exception as e:
            log.error(f"Synthesis error: {e}")
            answer = "\n".join([f"{r['tool'].upper()}: {', '.join([i['content'][:80] for i in r['data']])}"
                                for r in tool_results])

        # inline citation post-process
        for r in tool_results:
            for item in r["data"]:
                placeholder = f"[{r['tool'].upper()}]"
                citation = f"[Source: {item['source']}, {item['date']}"
                if item.get("id"):
                    citation += f" (ID:{item['id']})"
                citation += "]"
                answer = answer.replace(placeholder, citation, 1)
        return answer

    # ------------------- public API -------------------
    def answer_query(self, query: str) -> Dict[str, Any]:
        total_start = time.time()

        route_start = time.time()
        tool_call = self._route(query)
        route_lat = time.time() - route_start

        exec_start = time.time()
        tool_results = self._execute_tools(tool_call)
        exec_lat = time.time() - exec_start

        synth_start = time.time()
        answer = self._synthesize(query, tool_results)
        synth_lat = time.time() - synth_start

        total_lat = time.time() - total_start
        log.info(f"Query answered in {total_lat:.2f}s "
                 f"(route:{route_lat:.2f}s exec:{exec_lat:.2f}s synth:{synth_lat:.2f}s)")

        sources = []
        for r in tool_results:
            for item in r["data"]:
                sources.append({
                    "source": item["source"],
                    "id": item.get("id"),
                    "date": item["date"],
                })

        return {
            "answer": answer,
            "sources": sources,
            "tools_used": tool_call.tools,
            "latencies": {
                "total": round(total_lat, 2),
                "route": round(route_lat, 2),
                "execute": round(exec_lat, 2),
                "synthesize": round(synth_lat, 2),
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


# --------------------------------------------------------------------------- #
# CLI test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    agent = RAGAgent()
    tests = [
        "What's the latest on GTA VI release date?",
        "List the newest shooter games released this month.",
        "Summarize top gaming trends from recent news.",
    ]
    for q in tests:
        print("\n" + "=" * 80)
        print(f"Q: {q}")
        res = agent.answer_query(q)
        print(f"A: {res['answer']}")
        print(f"Tools: {', '.join(res['tools_used'])} | Total: {res['latencies']['total']}s")
        print(f"Sources: {len(res['sources'])}")