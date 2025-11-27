# vector/create_schema.py
"""
Create the GameChunk class in Weaviate using direct HTTP calls (no weaviate client).
Usage:
    python -m vector.create_schema
"""
import json
import time
import sys
from pathlib import Path

SCHEMA_PATH = Path("vector/schemas/weaviate_gamechunk_schema.json")
WEAVIATE_URL = "http://localhost:8080"
WAIT_SECONDS = 1
MAX_WAIT = 60  # seconds to wait for readiness

def wait_for_weaviate_ready(url: str, timeout: int = MAX_WAIT) -> bool:
    """Polls Weaviate readiness endpoint until it responds 200 or timeout."""
    try:
        import requests
    except Exception:
        print("[ERROR] 'requests' library is required. Install with: pip install requests")
        return False

    ready_url = f"{url.rstrip('/')}/v1/.well-known/ready"
    started = time.time()
    while True:
        try:
            resp = requests.get(ready_url, timeout=3)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        if time.time() - started > timeout:
            return False
        time.sleep(WAIT_SECONDS)

def get_schema(url: str):
    import requests
    r = requests.get(f"{url.rstrip('/')}/v1/schema", timeout=10)
    r.raise_for_status()
    return r.json()

def create_class(url: str, schema_obj: dict):
    import requests
    r = requests.post(f"{url.rstrip('/')}/v1/schema", json=schema_obj, timeout=30)
    # Weaviate returns 200 or 201 for success; otherwise raise
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Failed to create schema: {r.status_code} {r.text}")
    return r.json()

def main():
    if not SCHEMA_PATH.exists():
        print(f"[ERROR] Schema file not found at: {SCHEMA_PATH.resolve()}")
        sys.exit(2)

    try:
        with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
            schema = json.load(fh)
    except Exception as e:
        print(f"[ERROR] Failed reading schema JSON: {e}")
        sys.exit(3)

    print(f"[INFO] Waiting for Weaviate at {WEAVIATE_URL} to become ready (timeout {MAX_WAIT}s)...")
    if not wait_for_weaviate_ready(WEAVIATE_URL, timeout=MAX_WAIT):
        print(f"[ERROR] Weaviate readiness check failed after {MAX_WAIT}s. Check container logs:")
        print("  docker compose logs --no-color --tail=200 weaviate")
        sys.exit(4)

    try:
        existing = get_schema(WEAVIATE_URL)
    except Exception as e:
        print(f"[ERROR] Could not fetch schema from Weaviate: {e}")
        sys.exit(5)

    existing_classes = [c.get("class") for c in existing.get("classes", [])] if existing else []
    class_name = schema.get("class")
    if not class_name:
        print("[ERROR] Schema JSON missing top-level 'class' key.")
        sys.exit(6)

    if class_name in existing_classes:
        print(f"[INFO] Class '{class_name}' already exists in schema. No action taken.")
        return

    try:
        print(f"[INFO] Creating class '{class_name}' via REST API...")
        resp = create_class(WEAVIATE_URL, schema)
        print(f"[SUCCESS] Created class '{class_name}'. Response: {resp}")
    except Exception as e:
        print(f"[ERROR] Failed to create class '{class_name}': {e}")
        sys.exit(7)

if __name__ == "__main__":
    main()
