from parsers.docx_parser import ParsedDocument
from services.embedder_service import EmbedderService

from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import VectorParams, PointStruct
import uuid


class QdrantService:
  def __init__(self) -> None:
    self.client = QdrantClient(url="http://zenodotus.medcontrol.cloud:6333")
    self.collection = 'safemobile_docs'
    self.__init_collection()

  def __init_collection(self):
    self.client.delete_collection(self.collection)
    if self.collection not in [c.name for c in self.client.get_collections().collections]:
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=1024, distance=models.Distance.COSINE)
        )

  def load_items_to_collection(self, doc: ParsedDocument, version: str) -> None: 
    
    items = doc['paragraphs']
    for index, item in enumerate(items):
      print(f"{index}/{len(items)}")
      self.client.upsert(
        collection_name=self.collection,
        points=[
          PointStruct(
            id=str(uuid.uuid4()), 
            vector=EmbedderService.make_vector(f"{item['title']}: {text}"), 
            payload={
              "text": text,
              'paragraph_name': item['title'],
              'document_name': doc['document_name'],
              'document_version': version 
            }
          ) for text in item['chunks']
        ]
      )
      