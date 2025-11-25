from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import VectorParams, PointStruct
import json
from ollama import embeddings

qdrant = QdrantClient(url="http://zenodotus.medcontrol.cloud:6333")
collection = "safemobile_docs"