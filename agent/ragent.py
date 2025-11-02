#!/usr/bin/env python3
"""
RAG_ENT Agent - ReAct RAG Agent for Gaming Knowledge
Integrates: Weaviate RAG + IGDB + APITube News + Local HF LLM (<3B)
"""

import json
import time
import logging
import os
import torch
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Local imports (your project)
from retriever.retriever import retrieve_similar
from api.apitube_client import APITubeClient
from api.igdb_client import igdb_request
from utils.gpu_utils import get_device  # Assuming you have this
from vector.embed import get_embedding_model

# Load env
load_dotenv()

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format=f"[RAGENT] [GPU:{get_device().upper()}] [%(asctime)s] [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

class ReActAgent:
    def __init__(self):
        self.device = get_device()
        self.llm = self._init_llm()
        self.news_client = APITubeClient()
        self.conversation_history = []
        
        log.info(f"🚀 ReAct Agent initialized on {self.device.upper()}")

    def _init_llm(self):
        """Load <3B Gemma-2-2B (best local model for reasoning)"""
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        
        model_name = "google/gemma-2-2b-it"  # 2B params - perfect for local
        log.info(f"Loading {model_name} on {self.device.upper()}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        log.info("✅ LLM loaded successfully!")
        return pipe

    def _react_prompt(self, query: str) -> str:
        """ReAct prompt template with RAG context"""
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history[-3:]])
        
        return f"""<s>[INST] You are a gaming expert agent. Use tools when needed, think step-by-step.

Recent conversation:
{history}

Current question: {query}

Available tools:
1. search_rag - Search knowledge base (games + news)
2. search_igdb - Live IGDB game search
3. fetch_news - Live gaming news

Respond in this EXACT format:
Thought: [your reasoning]
Action: [tool_name]
Action Input: [exact input for tool]

Or to finish:
Final Answer: [your answer]

Tools MUST return data in this format:
{{"tool_name": "search_rag", "result": "data here"}}

Think carefully before acting. [/INST]"""

    def search_rag(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Tool: Search Weaviate RAG (your retriever)"""
        try:
            docs = retrieve_similar(query=query, top_k=top_k)
            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "unknown")
                content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                results.append(f"{i}. [{source}] {content}")
            
            return {
                "tool_name": "search_rag",
                "result": f"Found {len(docs)} relevant chunks:\n" + "\n".join(results),
                "doc_count": len(docs)
            }
        except Exception as e:
            return {"tool_name": "search_rag", "result": f"Search failed: {str(e)}"}

    def search_igdb(self, query: str) -> Dict[str, Any]:
        """Tool: Live IGDB search"""
        try:
            igdb_query = (
                f'fields id,name,summary,first_release_date,genres.name,platforms.name;'
                f'search "{query}"; limit 5;'
            )
            data = igdb_request("games", igdb_query)
            
            results = []
            for game in data[:3]:
                name = game.get("name", "Unknown")
                summary = game.get("summary", "")[:100] + "..." if game.get("summary") else ""
                genres = ", ".join(g["name"] for g in game.get("genres", []))
                results.append(f"- {name} | Genres: {genres} | {summary}")
            
            return {
                "tool_name": "search_igdb", 
                "result": f"Top IGDB matches for '{query}':\n" + "\n".join(results),
                "count": len(data)
            }
        except Exception as e:
            return {"tool_name": "search_igdb", "result": f"IGDB error: {str(e)}"}

    def fetch_news(self, query: str) -> Dict[str, Any]:
        """Tool: Live gaming news"""
        try:
            headlines = self.news_client.get_top_headlines(q=query, category="gaming")
            news_items = headlines.get("articles", [])[:3]
            
            results = []
            for item in news_items:
                title = item.get("title", "No title")
                source = item.get("source", {}).get("name", "Unknown")
                results.append(f"- {title} [{source}]")
            
            return {
                "tool_name": "fetch_news",
                "result": f"Latest news for '{query}':\n" + "\n".join(results),
                "count": len(news_items)
            }
        except Exception as e:
            return {"tool_name": "fetch_news", "result": f"News error: {str(e)}"}

    def _parse_action(self, response: str) -> tuple[Optional[str], Optional[str]]:
        """Parse ReAct response -> Action + Input"""
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        
        action = None
        action_input = None
        
        for line in lines:
            if line.startswith("Action:"):
                action = line.split("Action:")[1].strip().lower()
            elif line.startswith("Action Input:"):
                action_input = line.split("Action Input:")[1].strip().strip('"\'')
            elif line.startswith("Final Answer:"):
                return "FINAL", line.split("Final Answer:")[1].strip().strip('"\'')
        
        return action, action_input

    def run(self, query: str, max_steps: int = 5) -> Dict[str, Any]:
        """Main ReAct loop"""
        start_time = time.time()
        self.conversation_history.append({"role": "user", "content": query})
        
        step = 0
        tool_results = []
        
        while step < max_steps:
            step += 1
            log.info(f"🤔 Step {step}/{max_steps}: Processing '{query[:50]}...'")
            
            # 1. Build ReAct prompt
            prompt = self._react_prompt(query)
            
            # 2. LLM inference
            try:
                response = self.llm(prompt, pad_token_id=self.llm.tokenizer.eos_token_id)[0]["generated_text"]
                response = response.split("[/INST]")[-1].strip()  # Extract after instruction
            except Exception as e:
                log.error(f"LLM error: {e}")
                break
            
            # 3. Parse action
            action, action_input = self._parse_action(response)
            log.info(f"Action: {action} | Input: {action_input}")
            
            if action == "FINAL" or action_input is None:
                final_answer = action_input or "No clear answer found"
                latency = time.time() - start_time
                
                result = {
                    "answer": final_answer,
                    "steps": step,
                    "latency": round(latency, 2),
                    "tool_calls": len(tool_results),
                    "sources": tool_results
                }
                
                self.conversation_history.append({"role": "assistant", "content": final_answer})
                log.info(f"✅ Done in {latency:.2f}s ({step} steps)")
                return result
            
            # 4. Execute tool
            tool_result = None
            if action == "search_rag":
                tool_result = self.search_rag(action_input)
            elif action == "search_igdb":
                tool_result = self.search_igdb(action_input)
            elif action == "fetch_news":
                tool_result = self.fetch_news(action_input)
            else:
                tool_result = {"tool_name": action, "result": f"Unknown tool: {action}"}
            
            tool_results.append(tool_result)
            self.conversation_history.append({"role": "tool", "content": str(tool_result)})
        
        # Timeout fallback
        fallback = "Sorry, I couldn't complete the reasoning in time. Try a simpler query."
        return {"answer": fallback, "steps": max_steps, "tool_calls": 0, "latency": round(time.time() - start_time, 2)}

# === CLI INTERFACE ===
def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG_ENT Gaming Agent")
    parser.add_argument("query", help="Your gaming question")
    parser.add_argument("--steps", type=int, default=5, help="Max reasoning steps")
    args = parser.parse_args()
    
    agent = ReActAgent()
    result = agent.run(args.query, max_steps=args.steps)
    
    print("\n" + "="*80)
    print(f"🤖 ANSWER: {result['answer']}")
    print(f"⏱️  Time: {result['latency']}s | Steps: {result['steps']}")
    print(f"🔧 Tools: {result['tool_calls']}")
    
    if result['sources']:
        print("\n📚 SOURCES:")
        for source in result['sources'][-3:]:  # Last 3
            print(f"  {source['tool_name']}: {source['result'][:100]}...")

if __name__ == "__main__":
    main()