from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import VectorParams, PointStruct
import json
from ollama import embeddings
from services.qdrant_service import QdrantService

with open('chunks.json', 'r') as f:
  r = json.load(f)
  qdrant = QdrantService()
  qdrant.load_items_to_collection(r)