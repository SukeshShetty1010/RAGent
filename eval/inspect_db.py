# eval/inspect_db.py (Final)
from vector.index_manager import client
from weaviate.classes.query import Filter
import json
from datetime import datetime

def serialize(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)

c = client.collections.get('KnowledgeBase')

print("Counting documents...")
igdb_count = c.aggregate.over_all(filters=Filter.by_property('source').equal('igdb'), total_count=True).total_count
news_count = c.aggregate.over_all(filters=Filter.by_property('source').equal('news'), total_count=True).total_count
print(f"IGDB docs: {igdb_count}")
print(f"News docs: {news_count}")

igdb_samples = c.query.fetch_objects(limit=5, filters=Filter.by_property('source').equal('igdb'))
print("\n=== IGDB Samples ===")
for obj in igdb_samples.objects:
    p = obj.properties
    print(f"ID: {p.get('article_id')}, Content: {p.get('text', '')[:100]}...")

news_samples = c.query.fetch_objects(limit=5, filters=Filter.by_property('source').equal('news'))
print("\n=== News Samples ===")
for obj in news_samples.objects:
    p = obj.properties
    print(f"ID: {p.get('article_id')}, Content: {p.get('text', '')[:100]}...")

client.close()

# Save JSON safely
with open('db_samples.json', 'w') as f:
    json.dump({
        'igdb_count': igdb_count,
        'news_count': news_count,
        'igdb_samples': [o.properties for o in igdb_samples.objects],
        'news_samples': [o.properties for o in news_samples.objects]
    }, f, default=serialize, indent=2)
print("\nSaved to db_samples.json")