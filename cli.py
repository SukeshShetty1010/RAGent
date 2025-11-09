# RAG_ENT/cli.py
import fire
from agent.ragent import RAGAgent
from ingest.ragent_ingestor import main as ingest_main

agent = RAGAgent()

def ask(question: str, show_sources: bool = True):
    print(f"\nQ: {question}\n")
    result = agent.answer_query(question)
    print(f"{result['answer']}\n")

    if show_sources and result.get("citations"):
        print("Sources:")
        for i, c in enumerate(result["citations"], 1):
            src = c["source"].upper()
            title = c.get("title") or c["text"][:80]
            date = c.get("date", "unknown")
            url = c.get("url", "")
            print(f"[{i}] {title} [{src}] {date}")
            if url:
                print(f"    {url}")

def ingest():
    ingest_main()

def health():
    from vector.index_manager import client
    from utils.gpu_utils import get_device
    print(f"GPU: {get_device().upper()}")
    print(f"Weaviate: {'OK' if client.is_ready() else 'DOWN'}")

if __name__ == "__main__":
    fire.Fire()