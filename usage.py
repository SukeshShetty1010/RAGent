from vector.index_manager import client, COLLECTION_NAME

col = client.collections.get(COLLECTION_NAME)

print("COUNT:", col.aggregate.over_all(total_count=True).total_count)

res = col.query.get(
    limit=50,
    return_properties=["unified_game_id", "title", "doc_type"]
)

for obj in res.objects:
    print(obj.properties)
