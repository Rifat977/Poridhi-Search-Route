from qdrant_client import QdrantClient

# Connect to your Qdrant instance
client = QdrantClient(url="http://localhost:6333")

# Get all collection names
collections = client.get_collections().collections

for collection in collections:
    collection_name = collection.name
    print(f"🗑️ Deleting collection: {collection_name}")
    client.delete_collection(collection_name=collection_name)

print("✅ All collections have been removed.")
