# clear_kb.py
from vector.index_manager import client

# New Weaviate v4 way to delete EVERYTHING
try:
    client.collections.delete("KnowledgeBase")
    print("KB collection DELETED completely!")
except:
    pass

# Recreate empty collection
from vector.index_manager import create_index_if_not_exists
create_index_if_not_exists()
print("Fresh KnowledgeBase created — ready for 100% gaming data!")