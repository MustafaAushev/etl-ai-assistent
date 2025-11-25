from langchain_text_splitters import RecursiveCharacterTextSplitter

class ChunkableItem:
  title: str
  text: str

class ItemWithChunks:
  title: str
  text: str
  chunks: [str]

def make_chunks(items: [ChunkableItem]) -> [ItemWithChunks]:
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
  return [
    {
      'title': item['title'],
      'text': item['text'],
      'chunks': text_splitter.split_text(item['text'])
    } for item in items
  ]

