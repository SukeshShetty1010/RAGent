# search_result.py
from vector.search import search
from vector.index_manager import client  # <-- 1. Import the client

def main():
    try:
        # Search for "PilotXross" with source filter for IGDB and top 5 results
        results = search("PilotXross", source="IGDB", top_k=5)
        
        if not results:
            print("No results found for 'PilotXross' from source 'IGDB'.")
            return

        for i, doc in enumerate(results, 1):
            # This line is now correct!
            created_str = doc.metadata['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"{i}. Content: {doc.page_content[:100]}... (ID: {doc.metadata['article_id']}, Source: {doc.metadata['source']}, Created: {created_str})")
            
    finally:
        # <-- 2. Ensure the client is always closed when the script exits
        print("\nClosing Weaviate connection...")
        client.close()

if __name__ == "__main__":
    main()