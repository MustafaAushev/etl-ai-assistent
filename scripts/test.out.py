from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import VectorParams, PointStruct
import json
from ollama import chat
from services.embedder_service import EmbedderService
from services.qdrant_service import QdrantService


QUESTION = ''

qdrant = QdrantService()

hits = qdrant.client.query_points(
  query=EmbedderService.make_vector(QUESTION),
  collection_name=qdrant.collection,
  limit=5,
  score_threshold=0.5,
)

print(hits)
