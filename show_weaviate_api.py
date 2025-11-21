# save as show_weaviate_api.py and run: python show_weaviate_api.py
import pprint
from vector.index_manager import client, COLLECTION_NAME

def safe_dir(obj):
    try:
        return sorted([n for n in dir(obj) if not n.startswith("_")])
    except Exception as e:
        return f"<error listing dir: {e}>"

print("client type:", type(client))
print("\n=== client top-level attrs ===")
print("\n".join(safe_dir(client) if isinstance(safe_dir(client), list) else [safe_dir(client)]))

print("\n\n=== client.batch ===")
batch = getattr(client, "batch", None)
print("present:", batch is not None)
print(safe_dir(batch))

print("\n\n=== client.data_object ===")
data_obj = getattr(client, "data_object", None)
print("present:", data_obj is not None)
print(safe_dir(data_obj))

print("\n\n=== client.collections.get(COLLECTION_NAME) ===")
try:
    coll = client.collections.get(COLLECTION_NAME)
    print("collection type:", type(coll))
    print(safe_dir(coll))
    print("\n\n=== collection.batch.fixed_size callable? ===")
    try:
        ctx = coll.batch.fixed_size(4)
        print("fixed_size returned object type:", type(ctx))
        print("dir(ctx):", safe_dir(ctx))
        # If ctx is context manager, try entering it and show the inner object
        try:
            with ctx as real_batch:
                print("Inside context manager, real batch type:", type(real_batch))
                print("dir(real_batch):", safe_dir(real_batch))
        except Exception as e:
            print("Could not enter context manager or inspect real batch:", e)
    except Exception as e:
        print("coll.batch.fixed_size(...) raised:", e)
except Exception as e:
    print("Could not get collection or inspect it:", e)
