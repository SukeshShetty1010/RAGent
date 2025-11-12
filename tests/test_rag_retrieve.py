# tests/test_rag_retrieve.py
import json
from retriever.retriever import retrieve_similar
from retriever.rag import answer_query

def test_basic_retrieval():
    print("=" * 80)
    print("🎯  RAGent Retrieval Sanity Check")
    print("=" * 80)

    # Example queries based on your ingested GameSpot dataset
    queries = [
        "Tell me about the main storyline of Far Cry 6.",
        "What are the main platforms Far Cry 6 is available on?",
        "Who developed Far Cry 6?",
        "What is the release date and rating of Far Cry 6?",
    ]

    for q in queries:
        print(f"\n🔍 Query: {q}")
        results = retrieve_similar(q, top_k=3)
        if not results:
            print("⚠️  No results found.")
            continue

        print(f"✅ Retrieved {len(results)} chunks:")
        for i, doc in enumerate(results, 1):
            meta = doc.metadata
            print(f"  [{i}] {meta.get('title')} — {meta.get('source')}")
            print(f"      Genres: {meta.get('genres')}")
            print(f"      Platforms: {meta.get('platforms')}")
            print(f"      Snippet: {doc.page_content[:120].replace('\\n',' ')}...\n")

    print("=" * 80)
    print("✅ Retrieval test complete")
    print("=" * 80)

def test_rag_generation():
    print("\n" + "=" * 80)
    print("🧠  RAGent Generation Test (CPU)")
    print("=" * 80)

    query = "Summarize the gameplay experience of Far Cry 6."
    response = answer_query(query, filters={"source": "gamespot"}, top_k=5)
    print("\n🔎 Query:", query)
    print("\n💬 Answer:\n", response["answer"])
    print("\n📚 Citations:", response["citations"])
    print("\n📈 Metrics:", json.dumps(response["metrics"], indent=2))

if __name__ == "__main__":
    test_basic_retrieval()
    test_rag_generation()
