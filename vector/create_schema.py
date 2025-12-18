"""
vector/create_schema.py

Create or evolve a Weaviate class using a Weaviate Class Schema JSON.
Usage:
    python -m vector.create_schema
    python -m vector.create_schema --force-recreate
"""

from __future__ import annotations

import json
import time
import sys
import argparse
import hashlib
from pathlib import Path
from typing import Dict, Any

import requests

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SCHEMA_PATH = Path("vector/schemas/weaviate_gamechunk_schema.json")
WEAVIATE_URL = "http://localhost:8080"

WAIT_SECONDS = 1
MAX_WAIT = 60
REQUEST_TIMEOUT = 30


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def stable_hash(obj: Dict[str, Any]) -> str:
    """Create a stable hash for schema comparison."""
    canon = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def wait_for_weaviate_ready(url: str, timeout: int = MAX_WAIT) -> bool:
    ready_url = f"{url.rstrip('/')}/v1/.well-known/ready"
    start = time.time()

    while time.time() - start < timeout:
        try:
            r = requests.get(ready_url, timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(WAIT_SECONDS)

    return False


def get_schema(url: str) -> Dict[str, Any]:
    r = requests.get(f"{url.rstrip('/')}/v1/schema", timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def delete_class(url: str, class_name: str) -> None:
    r = requests.delete(
        f"{url.rstrip('/')}/v1/schema/{class_name}",
        timeout=REQUEST_TIMEOUT,
    )
    if r.status_code not in (200, 204):
        raise RuntimeError(f"Failed to delete class '{class_name}': {r.text}")


def create_class(url: str, class_schema: Dict[str, Any]) -> None:
    r = requests.post(
        f"{url.rstrip('/')}/v1/schema",
        json=class_schema,
        timeout=REQUEST_TIMEOUT,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Failed to create class: {r.status_code} {r.text}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Create or evolve Weaviate class schema")
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Delete and recreate class if schema differs (DESTRUCTIVE)",
    )
    args = parser.parse_args()

    if not SCHEMA_PATH.exists():
        print(f"[ERROR] Schema file not found: {SCHEMA_PATH.resolve()}")
        sys.exit(2)

    try:
        with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
            desired_schema = json.load(fh)
    except Exception as e:
        print(f"[ERROR] Failed to read schema JSON: {e}")
        sys.exit(3)

    class_name = desired_schema.get("class")
    if not class_name:
        print("[ERROR] Weaviate schema missing required top-level 'class' key.")
        sys.exit(4)

    desired_hash = stable_hash(desired_schema)

    print(f"[INFO] Waiting for Weaviate at {WEAVIATE_URL}...")
    if not wait_for_weaviate_ready(WEAVIATE_URL):
        print("[ERROR] Weaviate did not become ready in time.")
        sys.exit(5)

    try:
        current_schema = get_schema(WEAVIATE_URL)
    except Exception as e:
        print(f"[ERROR] Failed to fetch Weaviate schema: {e}")
        sys.exit(6)

    existing_classes = {
        c["class"]: c for c in current_schema.get("classes", [])
    }

    # -----------------------------------------------------------------
    # Class does not exist → create
    # -----------------------------------------------------------------

    if class_name not in existing_classes:
        print(f"[INFO] Class '{class_name}' not found. Creating...")
        create_class(WEAVIATE_URL, desired_schema)
        print(f"[SUCCESS] Class '{class_name}' created.")
        return

    # -----------------------------------------------------------------
    # Class exists → compare schema
    # -----------------------------------------------------------------

    existing_hash = stable_hash(existing_classes[class_name])

    if existing_hash == desired_hash:
        print(f"[INFO] Class '{class_name}' already matches schema. No action taken.")
        return

    print(f"[WARN] Schema drift detected for class '{class_name}'.")
    print(f"       Existing hash: {existing_hash}")
    print(f"       Desired  hash: {desired_hash}")

    if not args.force_recreate:
        print(
            "[ABORT] Schema differs but --force-recreate not provided.\n"
            "        This is expected after adding fields (e.g., GameSpot).\n"
            "        Re-run with --force-recreate to apply changes."
        )
        sys.exit(7)

    # -----------------------------------------------------------------
    # Force recreate
    # -----------------------------------------------------------------

    print(f"[DANGER] Deleting and recreating class '{class_name}'...")
    delete_class(WEAVIATE_URL, class_name)
    create_class(WEAVIATE_URL, desired_schema)
    print(f"[SUCCESS] Class '{class_name}' recreated with updated schema.")


if __name__ == "__main__":
    main()
