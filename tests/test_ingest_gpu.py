# test_ingest_gpu.py
import time
from ingest.game_ingest import GameIngestor
from api.igdb_client import igdb_client

start = time.time()
ingestor = GameIngestor(api_client=igdb_client, output_path="test.jsonl")
ingestor.run()
print(f"Full ingestion in {time.time() - start:.2f}s")