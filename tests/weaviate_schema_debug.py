# diagnostics/weaviate_schema_debug.py
import json
import sys
from vector import index_manager
from vector.index_manager import client, COLLECTION_NAME

print("COLLECTION_NAME:", COLLECTION_NAME)
print("client type:", type(client))
print("has attr 'collections'?:", hasattr(client, "collections"))
coll = getattr(client, "collections", None)
print("collections type:", type(coll))

# enumerate some useful attrs on client.collections
if coll is not None:
    attrs = [a for a in dir(coll) if not a.startswith("_")]
    print("collections attrs:", attrs)

# Try collections.get
try:
    print("\n--- Trying collections.get(COLLECTION_NAME) ---")
    c = coll.get(COLLECTION_NAME)
    print("collections.get() returned type:", type(c))
    print("repr(c):", repr(c))
    # Try to access config and properties
    cfg = getattr(c, "config", None)
    print("has config?:", cfg is not None)
    if cfg is not None:
        props = getattr(cfg, "properties", None)
        print("config.properties type:", type(props))
        try:
            print("config.properties repr (first 5):", repr(props[:5]))
        except Exception as e:
            print("Cannot slice/print props:", e)
        # try to read names robustly
        try:
            names = []
            for p in props:
                if isinstance(p, dict):
                    names.append(p.get("name"))
                else:
                    # object maybe; try attribute or mapping access
                    name = getattr(p, "name", None) or (p.get("name") if hasattr(p, "get") else None)
                    names.append(name)
            print("extracted property names:", names)
        except Exception as e:
            print("Error while extracting names from config.properties:", e)
except Exception as e:
    print("collections.get() raised:", repr(e))

# Try collections.list_all()
try:
    print("\n--- Trying collections.list_all() ---")
    allc = coll.list_all()
    print("list_all() returned type:", type(allc))
    # If mapping-like, print keys and one value repr
    if isinstance(allc, dict):
        print("list_all keys sample:", list(allc.keys())[:10])
        sample = next(iter(allc.values()))
        print("sample value type:", type(sample))
        try:
            props = getattr(sample, "properties", None) or (sample.get("properties") if isinstance(sample, dict) else None)
            print("sample.properties type:", type(props))
            # attempt to show first property repr
            try:
                print("sample.properties repr (first):", repr(props[0]))
            except Exception as e:
                print("Cannot index/print sample.properties:", e)
        except Exception as e:
            print("Error inspecting sample properties:", e)
    else:
        # try to iterate if list-like
        try:
            print("list_all length:", len(allc))
            print("list_all item 0 type:", type(allc[0]))
            print("list_all item 0 repr:", repr(allc[0]))
        except Exception as e:
            print("Cannot introspect list_all result:", e)
except Exception as e:
    print("collections.list_all() raised:", repr(e))

# Try legacy schema if present
try:
    print("\n--- Trying client.schema (legacy) ---")
    schema = getattr(client, "schema", None)
    print("has client.schema?:", schema is not None)
    if schema and hasattr(schema, "get"):
        full = schema.get()
        print("schema.get() returned type:", type(full))
        try:
            print("schema keys:", list(full.keys()) if isinstance(full, dict) else "no keys")
        except Exception as e:
            print("Cannot list schema keys:", e)
except Exception as e:
    print("client.schema access raised:", repr(e))

print("\nDone.")
