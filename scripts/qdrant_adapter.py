from chunk_splitter import ItemWithChunks
from embed_chunks import make_vector

from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import VectorParams, PointStruct
import uuid

qdrant = QdrantClient(url="http://zenodotus.medcontrol.cloud:6333")
collection = "safemobile_docs"

# Create collection if it does not exist
qdrant.delete_collection(collection)
if collection not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=1024, distance=models.Distance.COSINE)
    )


def load_items_to_collection(items: [ItemWithChunks]): 
  size = len(items)
  i = 1
  for item in items:
    print(f"{i}/{size}")
    i += 1
    points = [
      PointStruct(id=str(uuid.uuid4()), vector=make_vector(text), payload={"text": text}) for text in item['chunks']
    ]
    qdrant.upsert(
      collection_name=collection,
      points=points
    )