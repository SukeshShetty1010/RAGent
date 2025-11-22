# diagnostics/check_chunk_uuid_duplicates.py
import json, sys, collections
path = "all_docs.json"   # adjust if your run writes to different path
data = json.load(open(path, "r", encoding="utf-8"))
uu = [d.get("metadata", {}).get("chunk_uuid") for d in data if d.get("metadata", {}).get("chunk_uuid")]
cnt = collections.Counter(uu)
dups = {k:v for k,v in cnt.items() if v>1}
print("total chunks:", len(uu))
print("duplicates found:", len(dups))
if dups:
    print("Sample duplicates:", list(dups.items())[:10])
else:
    print("No duplicate chunk_uuid found.")
