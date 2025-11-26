from langchain_text_splitters import RecursiveCharacterTextSplitter
from parsers.docx_parser import ParsedDocument, DocumentParagraph

class ParagraphWithChunks(DocumentParagraph):
  title: str
  text: str
  chunks: [str]

class ParsedDocumentWithChunks(ParsedDocument):
  paragraphs: [ParagraphWithChunks]


class ChunkSplitterService:
  def make_chunks(doc: ParsedDocument) -> ParsedDocumentWithChunks:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
    doc['paragraphs'] = [
      {
        'title': item['title'],
        'text': item['text'],
        'chunks': text_splitter.split_text(item['text'])
      } for item in doc['paragraphs']
    ]
    return doc