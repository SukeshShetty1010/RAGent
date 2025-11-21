# init_schema.py
from vector import index_manager

if __name__ == "__main__":
    print("Ensuring GameKnowledge schema exists...")
    index_manager.create_index_if_not_exists()
    print("Done.")