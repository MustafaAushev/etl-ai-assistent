from chunk_splitter import make_chunks
import json
from qdrant_adapter import load_items_to_collection

with open('output1.json', 'r') as f:
  items = json.load(f)
  items_with_chunks = make_chunks(items)
  load_items_to_collection(items_with_chunks)
